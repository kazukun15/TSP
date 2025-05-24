import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import networkx as nx
import osmnx as ox
import packaging.version
from math import radians, sin, cos, sqrt, atan2
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ───────────────────────────────────────
# 定数
# ───────────────────────────────────────
EPSG_PLANE      = 2446
EPSG_WGS84      = 4326
GEOJSON_LOCAL   = "hinanjyo.geojson"
GEOJSON_REMOTE  = "https://raw.githubusercontent.com/<ユーザー名>/<リポジトリ名>/main/hinanjyo.geojson"
DEFAULT_CENTER  = (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
st.title("🏫 避難所最短ルート探すくん")

# ───────────────────────────────────────
# ユーティリティ関数
# ───────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """2点間の大圏距離を km 単位で返す"""
    R = 6371.0
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def load_initial_geojson():
    """ローカル or リモートから GeoJSON を読み込み、EPSG:4326 に変換して返す"""
    src = GEOJSON_LOCAL if os.path.exists(GEOJSON_LOCAL) else GEOJSON_REMOTE
    try:
        gdf = gpd.read_file(src)
        if gdf.crs is None or gdf.crs.to_epsg() != EPSG_PLANE:
            gdf = gdf.set_crs(epsg=EPSG_PLANE, allow_override=True)
        gdf = gdf.to_crs(epsg=EPSG_WGS84)
        gdf = gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry")).reset_index(drop=True)
    except Exception as e:
        st.error(f"初期 GeoJSON 読み込みエラー: {e}")
        return pd.DataFrame(columns=["lat","lon"])

def file_to_df(files):
    """アップロードされたファイル群 (SHP一式/GeoJSON/CSV) から避難所 DataFrame を生成"""
    try:
        # SHP一式
        if any(f.name.endswith(".shp") for f in files):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for f in files:
                    open(os.path.join(td,f.name),"wb").write(f.getvalue())
                shp = next(p for p in os.listdir(td) if p.endswith(".shp"))
                gdf = gpd.read_file(os.path.join(td, shp))
        # GeoJSON/JSON
        elif any(f.name.endswith((".geojson",".json")) for f in files):
            f = next(f for f in files if f.name.endswith((".geojson",".json")))
            gdf = gpd.read_file(f)
        # CSV
        else:
            f = next(f for f in files if f.name.endswith(".csv"))
            df = pd.read_csv(f)
            if not {"lat","lon"}.issubset(df.columns):
                st.error("CSV には必ず lat, lon 列が必要です")
                return pd.DataFrame()
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            if "name" not in df.columns:
                df["name"] = df.index.astype(str)
            return df.dropna(subset=["lat","lon"])
        # CRS 統一→4326
        if gdf.crs is None or gdf.crs.to_epsg() != EPSG_PLANE:
            gdf = gdf.set_crs(epsg=EPSG_PLANE, allow_override=True)
        gdf = gdf.to_crs(epsg=EPSG_WGS84)
        gdf = gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"]=gdf.geometry.x; gdf["lat"]=gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry"))
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame()

def create_road_distance_matrix(locs, mode="drive"):
    """
    locs: list of (lat, lon)
    mode: "drive" or "walk"
    returns: (distance_matrix [km], graph G or None, nearest_node_ids)
    """
    n = len(locs)
    # bbox + padding でグラフ取得
    ver = packaging.version.parse(ox.__version__)
    G = None
    for pad in [0.01, 0.03, 0.07]:
        north = max(lat for lat,_ in locs) + pad
        south = min(lat for lat,_ in locs) - pad
        east  = max(lon for _,lon in locs) + pad
        west  = min(lon for _,lon in locs) - pad
        try:
            if ver < packaging.version.parse("2.0.0"):
                G = ox.graph_from_bbox(north, south, east, west, network_type=mode)
            else:
                G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type=mode)
            if G.nodes:
                break
        except:
            G = None
    # nearest nodes
    nodes = []
    for lat,lon in locs:
        if G:
            try:
                nid = ox.nearest_nodes(G, lon, lat)
            except:
                nid = None
        else:
            nid = None
        nodes.append(nid)
    # 距離行列
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j: continue
            ni, nj = nodes[i], nodes[j]
            if G and ni is not None and nj is not None:
                try:
                    d_m = nx.shortest_path_length(G, ni, nj, weight="length")
                    mat[i,j] = d_m/1000.0
                    continue
                except:
                    pass
            # フォールバック: Haversine
            lat1,lon1 = locs[i]
            lat2,lon2 = locs[j]
            mat[i,j] = haversine(lat1, lon1, lat2, lon2)
    return mat, G, nodes

def solve_tsp(distance_matrix):
    """OR-Tools で TSP を解く。返り値は巡回経路のノードインデックスリスト"""
    size = len(distance_matrix)
    mgr  = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def cost_cb(f,t):
        return int(distance_matrix[mgr.IndexToNode(f), mgr.IndexToNode(t)] * 1e5)
    idx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 2
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    if not sol:
        return []
    route = []
    cur = routing.Start(0)
    while not routing.IsEnd(cur):
        route.append(mgr.IndexToNode(cur))
        cur = sol.Value(routing.NextVar(cur))
    route.append(route[0])
    return route

# ───────────────────────────────────────
# セッション状態
# ───────────────────────────────────────
if "shelters"   not in st.session_state:
    st.session_state.shelters = load_initial_geojson()
if "route"      not in st.session_state:
    st.session_state.route = []
if "road_path"  not in st.session_state:
    st.session_state.road_path = []

# ───────────────────────────────────────
# サイドバー: データ追加
# ───────────────────────────────────────
st.sidebar.header("避難所データ追加")
uploaded = st.sidebar.file_uploader(
    "SHP/GeoJSON/CSV をアップロード",
    type=["shp","geojson","json","csv"], accept_multiple_files=True
)
if uploaded:
    df_new = file_to_df(uploaded)
    if not df_new.empty:
        st.session_state.shelters = pd.concat(
            [st.session_state.shelters, df_new],
            ignore_index=True
        )
        st.sidebar.success(f"{len(df_new)} 件追加")

with st.sidebar.form("manual_add"):
    lat = st.number_input("緯度", value=DEFAULT_CENTER[0], format="%f")
    lon = st.number_input("経度", value=DEFAULT_CENTER[1], format="%f")
    name= st.text_input("避難所名", "新規避難所")
    if st.form_submit_button("追加"):
        row = pd.DataFrame([{"lat":lat,"lon":lon,"name":name}])
        st.session_state.shelters = pd.concat(
            [st.session_state.shelters, row], ignore_index=True
        )

if st.sidebar.button("リセット"):
    st.session_state.shelters = load_initial_geojson()
    st.session_state.route    = []
    st.session_state.road_path= []

# ───────────────────────────────────────
# サイドバー: TSP 設定
# ───────────────────────────────────────
df = st.session_state.shelters.copy()
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

label_cols = list(df.columns)
label_col  = st.sidebar.selectbox("ラベル列を選択", label_cols,
                 index=label_cols.index("name") if "name" in label_cols else 0)
buffer_m = st.sidebar.slider("バッファ距離 (m)", 0, 3000, 0, 100)
map_styles = {
    "Light":"light","Dark":"dark",
    "Streets":"mapbox://styles/mapbox/streets-v12",
    "Satellite":"mapbox://styles/mapbox/satellite-streets-v12"
}
map_style   = st.sidebar.selectbox("地図スタイル", list(map_styles.keys()))
mode_option = st.sidebar.selectbox("道路種別", ["drive","walk"],
                 format_func=lambda x:{"drive":"自動車","walk":"徒歩"}[x])

# ───────────────────────────────────────
# メイン: 施設選択
# ───────────────────────────────────────
st.markdown("<span style='font-size:14px;'>📋 巡回施設を選択</span>",
            unsafe_allow_html=True)
labels = [
    f"{row[label_col]} ({row['lat']:.5f},{row['lon']:.5f})"
    for _,row in df.iterrows()
]
idx_map = {i:labels[i] for i in range(len(labels))}
selected = st.multiselect(
    "選択", options=list(idx_map.keys()),
    format_func=lambda x: idx_map[x], key="sel"
)

# ───────────────────────────────────────
# TSP 実行
# ───────────────────────────────────────
if st.sidebar.button("TSP 計算"):
    if len(selected) < 2:
        st.sidebar.warning("2か所以上選択してください")
    else:
        sub = df.iloc[selected].reset_index(drop=True)
        locs = list(zip(sub["lat"], sub["lon"]))
        dist_mat, G, nodes = create_road_distance_matrix(locs, mode_option)
        route = solve_tsp(dist_mat)
        if not route:
            st.sidebar.error("TSP 解が見つかりませんでした")
        else:
            st.session_state.route = [ selected[i] for i in route if i < len(selected) ]
            # ─── 経路座標生成 ───
            path = []
            for k in range(len(route)-1):
                i, j = route[k], route[k+1]
                start_n = nodes[i] if i < len(nodes) else None
                end_n   = nodes[j] if j < len(nodes) else None
                coords = []
                if G and start_n is not None and end_n is not None:
                    try:
                        seg = nx.shortest_path(G, start_n, end_n, weight="length")
                        coords = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in seg]
                    except:
                        coords = []
                if not coords:
                    lon1, lat1 = sub.loc[i, ["lon","lat"]]
                    lon2, lat2 = sub.loc[j, ["lon","lat"]]
                    coords = [[lon1, lat1], [lon2, lat2]]
                if not path:
                    path.extend(coords)
                else:
                    path.extend(coords[1:])
            st.session_state.road_path = path
            total = sum(dist_mat[route[i],route[i+1]] for i in range(len(route)-1))
            st.sidebar.success(f"総距離: {total:.2f} km")

# ───────────────────────────────────────
# 地図描画
# ───────────────────────────────────────
# 動的センタリング
if not df.empty:
    src = df.iloc[selected] if selected else df
    center_lat = float(src["lat"].mean())
    center_lon = float(src["lon"].mean())
    span_lat = src["lat"].max() - src["lat"].min()
    span_lon = src["lon"].max() - src["lon"].min()
    if span_lat>0.1 or span_lon>0.1: zoom = 10
    elif span_lat>0.05 or span_lon>0.05: zoom = 11.5
    elif len(src)>1: zoom = 13.2
    else: zoom = 14
else:
    center_lat, center_lon = DEFAULT_CENTER
    zoom = 13.2

layers = []
# バッファ
if buffer_m>0:
    layers.append(pdk.Layer(
        "CircleLayer", data=df.iloc[selected] if selected else df,
        get_position='[lon,lat]', get_radius=buffer_m,
        get_fill_color=[0,0,255,50]
    ))
# ポイント
layers.append(pdk.Layer(
    "ScatterplotLayer", data=df,
    get_position='[lon,lat]', get_radius=30,
    get_color=[255,0,0,180]
))
# ラベル
layers.append(pdk.Layer(
    "TextLayer", data=df,
    get_position='[lon,lat]', get_text=label_col,
    get_size=16, get_color=[0,0,0,200],
    get_alignment_baseline="'bottom'"
))
# 経路
rp = st.session_state.road_path
if rp and len(rp)>1:
    layers.append(pdk.Layer(
        "PathLayer", data=pd.DataFrame({"path":[rp]}),
        get_path="path", get_color=[255,60,60,200],
        width_scale=10, width_min_pixels=3
    ))

view = pdk.ViewState(
    latitude=center_lat, longitude=center_lon,
    zoom=zoom, pitch=45
)
st.pydeck_chart(
    pdk.Deck(
        map_style=map_styles[map_style],
        initial_view_state=view,
        layers=layers,
        tooltip={"text": f"{{{label_col}}}"}
    ),
    use_container_width=True
)

# ───────────────────────────────────────
# データ一覧 & 巡回順
# ───────────────────────────────────────
with st.expander("📋 避難所一覧・巡回順"):
    st.dataframe(df)
    if st.session_state.route:
        order = [ df.iloc[i][label_col] for i in st.session_state.route ]
        st.write("巡回順:", order)

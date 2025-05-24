import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import osmnx as ox
import networkx as nx
import packaging.version
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ───────────────────────────────────────
# 設定
# ───────────────────────────────────────
GEOJSON_LOCAL_PATH = "hinanjyo.geojson"
GEOJSON_RAW_URL   = "https://raw.githubusercontent.com/<ユーザー名>/<リポジトリ名>/main/hinanjyo.geojson"
KAMIJIMA_CENTER   = (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
st.markdown("""
    <style>
    @media (max-width: 800px) {
        .block-container { padding-left:0.5rem; padding-right:0.5rem; }
        .stButton button { font-size:1.1em; padding:0.6em 1.2em; }
    }
    </style>
""", unsafe_allow_html=True)
st.title("🏫 避難所最短ルート探すくん")

# ───────────────────────────────────────
# GeoJSON/SHP/CSV 読み込みユーティリティ
# ───────────────────────────────────────
def load_initial_geojson():
    """ローカル or GitHub から初期 GeoJSON を読み込み、EPSG:2446→4326変換後、全プロパティ＋lat/lon列を返す"""
    try:
        src = GEOJSON_LOCAL_PATH if os.path.exists(GEOJSON_LOCAL_PATH) else GEOJSON_RAW_URL
        gdf = gpd.read_file(src)
        # 投影を2446に固定→4326に変換
        if gdf.crs is None or gdf.crs.to_epsg() != 2446:
            gdf = gdf.set_crs(epsg=2446, allow_override=True)
        gdf = gdf.to_crs(epsg=4326)
        # Point のみ
        gdf = gdf[gdf.geometry.type == "Point"].copy()
        # lat/lon列追加
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        # geometryは不要
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"初期GeoJSON読み込みエラー: {e}")
        return pd.DataFrame()

def file_to_df(uploaded_files):
    """アップロードされた SHP一式 / GeoJSON / CSV を全プロパティ＋lat/lon DataFrameに変換"""
    try:
        # SHP一式
        if any(f.name.endswith(".shp") for f in uploaded_files):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for f in uploaded_files:
                    open(os.path.join(td, f.name),"wb").write(f.getvalue())
                shp = next(p for p in os.listdir(td) if p.endswith(".shp"))
                gdf = gpd.read_file(os.path.join(td, shp))
        # GeoJSON/JSON
        elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
            f = next(f for f in uploaded_files if f.name.endswith((".geojson",".json")))
            gdf = gpd.read_file(f)
        # CSV
        else:
            f = next(f for f in uploaded_files if f.name.endswith(".csv"))
            df_csv = pd.read_csv(f)
            if not {"lat","lon"}.issubset(df_csv.columns):
                st.warning("CSVには必ずlat, lon列が必要です")
                return pd.DataFrame()
            df_csv["lat"] = pd.to_numeric(df_csv["lat"], errors="coerce")
            df_csv["lon"] = pd.to_numeric(df_csv["lon"], errors="coerce")
            if "name" not in df_csv.columns:
                df_csv["name"] = df_csv.index.astype(str)
            return df_csv.dropna(subset=["lat","lon"])
        # CRS統一→4326
        if gdf.crs is None or gdf.crs.to_epsg()!=2446:
            gdf = gdf.set_crs(epsg=2446, allow_override=True)
        gdf = gdf.to_crs(epsg=4326)
        # Pointのみ & lat/lon列追加
        gdf = gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame()

# ───────────────────────────────────────
# 道路ネットワーク距離行列＋ノード取得
# ───────────────────────────────────────
def create_road_distance_matrix(locs, mode="drive"):
    """OSMnxで道路ネットを取得、距離行列＋Graph＋nearest node listを返す"""
    import numpy as np
    ver = packaging.version.parse(ox.__version__)
    lats = [float(p[0]) for p in locs]
    lons = [float(p[1]) for p in locs]
    for pad in [0.01,0.03,0.07]:
        try:
            if ver < packaging.version.parse("2.0.0"):
                G = ox.graph_from_bbox(
                    max(lats)+pad, min(lats)-pad,
                    max(lons)+pad, min(lons)-pad,
                    network_type=mode)
            else:
                bbox = (max(lats)+pad, min(lats)-pad, max(lons)+pad, min(lons)-pad)
                G = ox.graph_from_bbox(bbox=bbox, network_type=mode)
            if not G.nodes:
                continue
            nodes = []
            for lat,lon in locs:
                try:
                    nodes.append(ox.nearest_nodes(G, lon, lat))
                except:
                    nodes.append(None)
            n = len(locs)
            mat = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i==j: continue
                    ni, nj = nodes[i], nodes[j]
                    if ni is not None and nj is not None:
                        try:
                            mat[i,j] = nx.shortest_path_length(G, ni, nj, weight="length")/1000
                        except:
                            mat[i,j] = np.inf
                    else:
                        mat[i,j] = np.inf
            return mat, G, nodes
        except:
            continue
    st.warning("道路ネット取得失敗→直線距離TSPに切替")
    n = len(locs)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                mat[i,j] = np.linalg.norm(np.array(locs[i]) - np.array(locs[j]))
    return mat, None, []

# ───────────────────────────────────────
# OR-Tools で TSP 解決
# ───────────────────────────────────────
def solve_tsp(distance_matrix):
    mgr = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def cb(f,t):
        return int(distance_matrix[mgr.IndexToNode(f), mgr.IndexToNode(t)]*1e5)
    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 1
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    route = []
    if sol:
        cur = routing.Start(0)
        while not routing.IsEnd(cur):
            route.append(mgr.IndexToNode(cur))
            cur = sol.Value(routing.NextVar(cur))
        route.append(route[0])
    return route

# ───────────────────────────────────────
# セッションステート初期化
# ───────────────────────────────────────
if "shelters" not in st.session_state:
    st.session_state["shelters"] = load_initial_geojson()
if "route" not in st.session_state:
    st.session_state["route"] = []
if "road_path" not in st.session_state:
    st.session_state["road_path"] = []

# ───────────────────────────────────────
# サイドバー：データ追加
# ───────────────────────────────────────
st.sidebar.header("避難所データ追加")
st.sidebar.info("SHP/GeoJSON/CSV → 一括アップロード可")
up = st.sidebar.file_uploader(
    "ファイルを選択", 
    type=["shp","shx","dbf","prj","geojson","json","csv"],
    accept_multiple_files=True
)
if up:
    newdf = file_to_df(up)
    if not newdf.empty:
        st.session_state["shelters"] = pd.concat([st.session_state["shelters"], newdf], ignore_index=True)
        st.success(f"{len(newdf)} 件追加")

with st.sidebar.form("manual_add"):
    st.write("手動追加")
    lat = st.number_input("緯度", value=KAMIJIMA_CENTER[0], format="%f")
    lon = st.number_input("経度", value=KAMIJIMA_CENTER[1], format="%f")
    name= st.text_input("避難所名","新規避難所")
    if st.form_submit_button("追加"):
        st.session_state["shelters"] = pd.concat([
            st.session_state["shelters"],
            pd.DataFrame([{"lat":lat,"lon":lon,"name":name}])
        ], ignore_index=True)
if st.sidebar.button("リセット"):
    st.session_state["shelters"] = load_initial_geojson()
    st.session_state["route"] = []
    st.session_state["road_path"] = []

# ───────────────────────────────────────
# サイドバー：TSP設定
# ───────────────────────────────────────
df = st.session_state["shelters"].copy()
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

# 全カラムをラベル候補に
label_opts = list(df.columns)
label_col  = st.sidebar.selectbox("ラベル列を選択", label_opts, index=label_opts.index("name") if "name" in label_opts else 0)

# バッファ距離
buffer_radius = st.sidebar.slider("バッファ距離(m)", min_value=0, max_value=3000, value=0, step=100)

# 地図スタイル
map_styles = {
    "light":"light","dark":"dark",
    "ストリート":"mapbox://styles/mapbox/streets-v12",
    "衛星":"mapbox://styles/mapbox/satellite-streets-v12"
}
map_style = st.sidebar.selectbox("地図スタイル", list(map_styles.keys()))

# 道路種別
mode = st.sidebar.selectbox("道路種別", ["drive","walk"], format_func=lambda x: {"drive":"自動車","walk":"徒歩"}[x])

# ───────────────────────────────────────
# メイン：施設選択
# ───────────────────────────────────────
st.markdown("<span style='font-size:14px;'>📋 巡回施設を選択</span>", unsafe_allow_html=True)
display = [
    f"{row[label_col]} ({row['lat']:.5f},{row['lon']:.5f})"
    for _,row in df.iterrows()
]
idx_map = {i:n for i,n in enumerate(display)}
sel = st.multiselect(
    "選択", options=list(idx_map.keys()),
    format_func=lambda x: idx_map[x], key="sel"
)

if st.sidebar.button("TSP 計算"):
    if len(sel) < 2:
        st.warning("２か所以上選択してください")
    else:
        sub = df.iloc[sel].reset_index(drop=True)
        locs = list(zip(sub["lat"], sub["lon"]))
        with st.spinner("計算中…"):
            mat, G, nodes = create_road_distance_matrix(locs, mode=mode)
            route = solve_tsp(mat)
            # セッションに「元DFのインデックス」で保存
            st.session_state["route"] = [ sel[i] for i in route if i < len(sel) ]
            # 路線座標作成
            path = []
            if G is not None:
                for i in range(len(route)-1):
                    seg = nx.shortest_path(G, nodes[route[i]], nodes[route[i+1]], weight="length")
                    coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in seg]
                    path += coords if i==0 else coords[1:]
            else:
                for i in range(len(route)-1):
                    path += [
                        [ sub.loc[route[i],"lat"], sub.loc[route[i],"lon"] ],
                        [ sub.loc[route[i+1],"lat"], sub.loc[route[i+1],"lon"] ]
                    ]
            st.session_state["road_path"] = path
            total = sum(mat[route[i],route[i+1]] for i in range(len(route)-1))
            st.success(f"総距離: {total:.2f} km")

# ───────────────────────────────────────
# 地図描画
# ───────────────────────────────────────
st.markdown("<span style='font-size:14px;'>🗺️ 地図表示</span>", unsafe_allow_html=True)
layers = []

# バッファ
if buffer_radius>0:
    layers.append(pdk.Layer(
        "CircleLayer", data=df.iloc[sel] if sel else df,
        get_position='[lon,lat]', get_radius=buffer_radius,
        get_fill_color=[0,0,255,60]
    ))

# ポイント
layers.append(pdk.Layer(
    "ScatterplotLayer", data=df,
    get_position='[lon,lat]', get_radius=30,
    get_color=[255,0,0,180]
))
# ラベル（ポイント上）
layers.append(pdk.Layer(
    "TextLayer", data=df,
    get_position='[lon,lat]', get_text=label_col,
    get_size=16, get_color=[0,0,0,200],
    get_alignment_baseline="'bottom'"
))

# TSP経路
rp = st.session_state["road_path"]
if rp and len(rp)>1:
    layers.append(pdk.Layer(
        "PathLayer", data=pd.DataFrame({"path":[rp]}),
        get_path="path", get_color=[255,60,60,200],
        width_scale=10, width_min_pixels=3
    ))

view = pdk.ViewState(
    latitude=KAMIJIMA_CENTER[0], longitude=KAMIJIMA_CENTER[1],
    zoom=13.2, pitch=45
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
# データ一覧 & 巡回順 Expander
# ───────────────────────────────────────
with st.expander("📋 避難所データ一覧・巡回順"):
    st.dataframe(df)
    if st.session_state["route"]:
        order = [ df.iloc[i][label_col] for i in st.session_state["route"] ]
        st.write("巡回順:", order)

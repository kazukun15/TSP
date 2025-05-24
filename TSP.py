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

# ── 設定 ─────────────────────────────────────────────────────────
GEOJSON_LOCAL_PATH = "hinanjyo.geojson"
GEOJSON_RAW_URL   = "https://raw.githubusercontent.com/<ユーザー名>/<リポジトリ名>/<ブランチ名>/hinanjyo.geojson"
KAMIJIMA_CENTER   = (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
# モバイルでも余白を抑えるCSS
st.markdown("""
    <style>
    @media (max-width: 800px) {
        .block-container { padding-left:0.4rem; padding-right:0.4rem; }
        .stButton button { font-size:1.12em; padding:0.7em 1.3em; }
    }
    </style>
""", unsafe_allow_html=True)
st.title("🏫 避難所最短ルート探すくん")

# ── ヘルパー関数 ─────────────────────────────────────────────────
def load_initial_geojson():
    """ローカル or GitHub から初期 GeoJSON を読む"""
    try:
        path = GEOJSON_LOCAL_PATH if os.path.exists(GEOJSON_LOCAL_PATH) else GEOJSON_RAW_URL
        gdf = gpd.read_file(path)
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"] = gdf.index.astype(str)
        return gdf.reset_index(drop=True)
    except Exception as e:
        st.error(f"初期GeoJSON読み込みエラー: {e}")
        return gpd.GeoDataFrame(columns=["lat","lon","name"])

def file_to_df(uploaded_files):
    """SHP/GeoJSON/CSV を DataFrame(lat, lon, name) に変換"""
    try:
        # SHP一式
        if any(f.name.endswith(".shp") for f in uploaded_files):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for f in uploaded_files:
                    open(os.path.join(td,f.name),"wb").write(f.getvalue())
                shp = [os.path.join(td,f.name) for f in uploaded_files if f.name.endswith(".shp")][0]
                gdf = gpd.read_file(shp)
        # GeoJSON/JSON
        elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
            f = [f for f in uploaded_files if f.name.endswith((".geojson",".json"))][0]
            gdf = gpd.read_file(f)
        # CSV
        else:
            f = [f for f in uploaded_files if f.name.endswith(".csv")][0]
            df = pd.read_csv(f)
            if not {"lat","lon"}.issubset(df.columns):
                st.warning("CSVにはlat, lon列が必要です")
                return pd.DataFrame(columns=["lat","lon","name"])
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            if "name" not in df.columns:
                df["name"] = df.index.astype(str)
            return df.dropna(subset=["lat","lon"])[["lat","lon","name"]]
        # GeoDataFrame → DataFrame
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"] = gdf.index.astype(str)
        return gdf[["lat","lon","name"]]
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame(columns=["lat","lon","name"])

def create_road_distance_matrix(locs, mode="drive"):
    """OSM から道路ネットワーク取得 → 距離行列、ノードIDリスト返却"""
    import numpy as np
    version = packaging.version.parse(ox.__version__)
    lats = [float(p[0]) for p in locs]
    lons = [float(p[1]) for p in locs]
    for pad in [0.01,0.03,0.07]:
        try:
            if version < packaging.version.parse("2.0.0"):
                G = ox.graph_from_bbox(max(lats)+pad, min(lats)-pad,
                                       max(lons)+pad, min(lons)-pad,
                                       network_type=mode)
            else:
                bbox = (max(lats)+pad, min(lats)-pad, max(lons)+pad, min(lons)-pad)
                G = ox.graph_from_bbox(bbox=bbox, network_type=mode)
            if not G.nodes:
                continue
            # nearest_nodes
            node_ids = []
            for lat, lon in locs:
                try:
                    node_ids.append(ox.nearest_nodes(G, lon, lat))
                except:
                    node_ids.append(None)
            # distance matrix
            n = len(locs)
            mat = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i==j:
                        mat[i,j] = 0
                    elif node_ids[i] is not None and node_ids[j] is not None:
                        try:
                            mat[i,j] = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight="length")/1000
                        except:
                            mat[i,j] = float("inf")
                    else:
                        mat[i,j] = float("inf")
            return mat, G, node_ids
        except:
            continue
    # フォールバック：直線距離
    st.warning("道路ネットワーク取得失敗 → 直線距離でTSP")
    n = len(locs)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                mat[i,j] = np.linalg.norm(np.array(locs[i]) - np.array(locs[j]))
    return mat, None, []

def solve_tsp(distance_matrix):
    """OR-Tools で TSP 解く"""
    size = len(distance_matrix)
    mgr  = pywrapcp.RoutingIndexManager(size,1,0)
    routing = pywrapcp.RoutingModel(mgr)
    def cb(f,t):
        return int(distance_matrix[mgr.IndexToNode(f), mgr.IndexToNode(t)]*100000)
    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 1
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    route=[]
    if sol:
        cur = routing.Start(0)
        while not routing.IsEnd(cur):
            route.append(mgr.IndexToNode(cur))
            cur = sol.Value(routing.NextVar(cur))
        route.append(route[0])
    return route

# ── セッションステート初期化 ────────────────────────────────────────
if "shelters" not in st.session_state:
    st.session_state["shelters"] = load_initial_geojson()
if "route" not in st.session_state:
    st.session_state["route"] = []
if "road_path" not in st.session_state:
    st.session_state["road_path"] = []

# ── サイドバー：データ追加 ─────────────────────────────────────────
st.sidebar.header("避難所データ追加")
st.sidebar.info("SHP/GeoJSON/CSV → 一括アップロード可")
up = st.sidebar.file_uploader(
    "ファイルを選択", 
    type=["shp","shx","dbf","prj","geojson","json","csv"],
    accept_multiple_files=True
)
if up:
    new_df = file_to_df(up)
    if not new_df.empty:
        st.session_state["shelters"] = pd.concat([st.session_state["shelters"], new_df], ignore_index=True)
        st.success(f"{len(new_df)} 件追加")

with st.sidebar.form("manual_add"):
    st.write("手動で追加")
    lat = st.number_input("緯度", value=KAMIJIMA_CENTER[0], format="%f")
    lon = st.number_input("経度", value=KAMIJIMA_CENTER[1], format="%f")
    name= st.text_input("避難所名", "新規避難所")
    if st.form_submit_button("追加"):
        st.session_state["shelters"] = pd.concat([
            st.session_state["shelters"],
            pd.DataFrame([{"lat":lat,"lon":lon,"name":name}])
        ], ignore_index=True)
if st.sidebar.button("リセット"):
    st.session_state["shelters"] = load_initial_geojson()
    st.session_state["route"] = []
    st.session_state["road_path"] = []

# ── サイドバー：UI設定 ─────────────────────────────────────────
df = st.session_state["shelters"].copy()
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

# 全カラムをラベル候補に
label_opts = list(df.columns)
label_col = st.sidebar.selectbox("ラベル列を選択", label_opts, index=label_opts.index("name") if "name" in label_opts else 0)

# バッファ距離
buffer_radius = st.sidebar.slider("バッファ距離 (m)", min_value=0, max_value=3000, value=0, step=100)

# 地図スタイル
map_style_dict = {
    "light":"light","dark":"dark",
    "ストリート":"mapbox://styles/mapbox/streets-v12",
    "衛星":"mapbox://styles/mapbox/satellite-streets-v12"
}
map_style = st.sidebar.selectbox("地図スタイル", list(map_style_dict.keys()))

# 道路種別 & TSP ボタン
mode = st.sidebar.selectbox("道路種別", ["drive","walk"], format_func=lambda x: {"drive":"自動車","walk":"徒歩"}[x])
if st.sidebar.button("TSP 計算"):
    # 選択が後ろで処理

    pass

# ── メイン：施設選択 ─────────────────────────────────────────
st.markdown("<span style='font-size:14px;'>📋 巡回施設を選択</span>", unsafe_allow_html=True)
display = [
    f"{row[label_col]} ({row['lat']:.5f},{row['lon']:.5f})"
    for _,row in df.iterrows()
]
idx_map = {i:name for i,name in enumerate(display)}
sel = st.multiselect(
    "対象を複数選択", 
    options=list(idx_map.keys()),
    format_func=lambda x: idx_map[x],
    key="sel"
)

# ── TSP 計算実行 ─────────────────────────────────────────
if st.sidebar.button("計算スタート"):
    if len(sel)<2:
        st.warning("2か所以上選択してください。")
    else:
        sub = df.iloc[sel].reset_index(drop=True)
        locs = list(zip(sub["lat"], sub["lon"]))
        with st.spinner("ルート計算中…"):
            distmat, G, node_ids = create_road_distance_matrix(locs, mode=mode)
            route = solve_tsp(distmat)
            st.session_state["route"] = [ sel[i] for i in route if i < len(sel) ]
            # 線形経路座標
            path=[]
            if G is not None:
                for i in range(len(route)-1):
                    seg = nx.shortest_path(G, node_ids[route[i]], node_ids[route[i+1]], weight="length")
                    coords = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in seg]
                    path += coords if i==0 else coords[1:]
            else:
                for i in range(len(route)-1):
                    path.append([ sub.loc[route[i],"lon"], sub.loc[route[i],"lat"] ])
                    path.append([ sub.loc[route[i+1],"lon"], sub.loc[route[i+1],"lat"] ])
            st.session_state["road_path"] = path
            total = sum(distmat[route[i],route[i+1]] for i in range(len(route)-1))
            st.success(f"総距離: {total:.2f} km")

# ── 地図描画 ─────────────────────────────────────────────────────
st.markdown("<span style='font-size:14px;'>🗺️ 地図表示</span>", unsafe_allow_html=True)

layers = []
# バッファ円
if buffer_radius>0:
    layers.append(pdk.Layer(
        "CircleLayer",
        data=df.iloc[sel] if sel else df,
        get_position='[lon, lat]',
        get_radius=buffer_radius,
        get_fill_color=[0,0,255,50],
        pickable=False
    ))
# ポイント
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon,lat]',
    get_color='[255,0,0,200]',
    get_radius=50,
    pickable=True
))
# テキスト
layers.append(pdk.Layer(
    "TextLayer",
    data=df,
    get_position='[lon,lat]',
    get_text=label_col,
    get_size=16,
    get_color=[0,0,0,200],
    get_alignment_baseline="'bottom'",
    pickable=False
))
# TSP 経路
rp = st.session_state["road_path"]
if rp and len(rp)>1:
    layers.append(pdk.Layer(
        "PathLayer",
        data=pd.DataFrame({"path":[rp]}),
        get_path="path",
        get_color=[255,0,0,200],
        width_scale=10,
        width_min_pixels=4,
        pickable=False
    ))

view = pdk.ViewState(
    latitude=KAMIJIMA_CENTER[0], longitude=KAMIJIMA_CENTER[1],
    zoom=13.3, pitch=45, bearing=0
)
st.pydeck_chart(
    pdk.Deck(
        map_style=map_style_dict[map_style],
        initial_view_state=view,
        layers=layers,
        tooltip={"text": f"{{{label_col}}}"}
    ),
    use_container_width=True
)

# ── データ一覧 expander ─────────────────────────────────────────
with st.expander("📋 避難所データ一覧・巡回順"):
    st.dataframe(df)
    if st.session_state["route"]:
        st.write(
            "巡回順:",
            [ df.iloc[i][label_col] for i in st.session_state["route"] ]
        )

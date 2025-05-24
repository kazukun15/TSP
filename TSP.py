import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import osmnx as ox
import networkx as nx
import packaging.version
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

KAMIJIMA_CENTER = (34.25754417840102, 133.20446981161595)
st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")

st.markdown("""
    <style>
    @media (max-width: 800px) {
        .block-container { padding-left: 0.4rem; padding-right: 0.4rem; }
        .stButton button { font-size: 1.12em; padding: 0.7em 1.3em; }
    }
    </style>
""", unsafe_allow_html=True)
st.title("🏫 避難所最短ルート探すくん")

def guess_name_col(df):
    for cand in ["name", "NAME", "名称", "避難所", "施設名", "address", "住所"]:
        if cand in df.columns:
            return cand
    obj_cols = [c for c in df.columns if df[c].dtype == 'O']
    if obj_cols:
        return obj_cols[0]
    return df.columns[0] if not df.empty else "name"

def file_to_df(uploaded_files):
    try:
        if any(f.name.endswith(".shp") for f in uploaded_files):
            import tempfile, os
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    with open(os.path.join(temp_dir, file.name), "wb") as out:
                        out.write(file.getvalue())
                shp_path = [os.path.join(temp_dir, f.name) for f in uploaded_files if f.name.endswith(".shp")][0]
                gdf = gpd.read_file(shp_path)
        elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
            geojson_file = [f for f in uploaded_files if f.name.endswith((".geojson",".json"))][0]
            gdf = gpd.read_file(geojson_file)
        elif any(f.name.endswith(".csv") for f in uploaded_files):
            csv_file = [f for f in uploaded_files if f.name.endswith(".csv")][0]
            df = pd.read_csv(csv_file)
            if not set(["lat","lon"]).issubset(df.columns):
                st.warning("lat, lon 列が必要です")
                return pd.DataFrame(columns=["lat", "lon", "name"])
            for c in ["lat", "lon"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        else:
            st.warning("SHP/GeoJSON/CSVのみ対応です")
            return pd.DataFrame(columns=["lat", "lon", "name"])

        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        if "geometry" not in gdf.columns or gdf.empty:
            st.warning("ジオメトリ情報がありません")
            return pd.DataFrame(columns=["lat", "lon", "name"])
        if not (gdf.geometry.type == "Point").any():
            st.warning("Point型ジオメトリのみ対応です")
            return pd.DataFrame(columns=["lat", "lon", "name"])
        gdf = gdf[gdf.geometry.type == "Point"]

        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"] = gdf.index.astype(str)
        gdf["lat"] = pd.to_numeric(gdf["lat"], errors="coerce")
        gdf["lon"] = pd.to_numeric(gdf["lon"], errors="coerce")
        gdf = gdf.dropna(subset=["lat", "lon"])
        return gdf.reset_index(drop=True)
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame(columns=["lat", "lon", "name"])

def create_road_distance_matrix(locs, mode="drive"):
    import networkx as nx
    import numpy as np
    import osmnx as ox
    lats = [float(p[0]) for p in locs]
    lons = [float(p[1]) for p in locs]
    version = packaging.version.parse(ox.__version__)
    for pad in [0.01, 0.03, 0.07]:
        try:
            if version < packaging.version.parse("2.0.0"):
                G = ox.graph_from_bbox(
                    max(lats) + pad, min(lats) - pad,
                    max(lons) + pad, min(lons) - pad,
                    network_type=mode
                )
            else:
                bbox = (max(lats) + pad, min(lats) - pad, max(lons) + pad, min(lons) - pad)
                G = ox.graph_from_bbox(bbox=bbox, network_type=mode)
            if len(G.nodes) == 0:
                continue
            node_ids = []
            for lat, lon in locs:
                try:
                    node_id = ox.nearest_nodes(G, lon, lat)
                    node_ids.append(node_id)
                except Exception:
                    node_ids.append(None)
            n = len(locs)
            mat = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        mat[i, j] = 0
                    elif node_ids[i] is not None and node_ids[j] is not None:
                        try:
                            mat[i, j] = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight='length') / 1000
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            mat[i, j] = float('inf')
                    else:
                        mat[i, j] = float('inf')
            return mat, G, node_ids
        except Exception:
            continue
    st.warning("道路ネットワークを取得できませんでした。直線距離でTSP巡回します。")
    n = len(locs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0
            else:
                mat[i, j] = np.linalg.norm(np.array(locs[i]) - np.array(locs[j]))
    return mat, None, []

def solve_tsp(distance_matrix):
    size = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 100000)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 1
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(search_parameters)
    route = []
    if solution:
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(route[0])
    return route

def load_initial_geojson(filepath):
    try:
        gdf = gpd.read_file(filepath)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        if "geometry" not in gdf.columns or gdf.empty:
            return pd.DataFrame(columns=["lat", "lon", "name"])
        gdf = gdf[gdf.geometry.type == "Point"]
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"] = gdf.index.astype(str)
        gdf["lat"] = pd.to_numeric(gdf["lat"], errors="coerce")
        gdf["lon"] = pd.to_numeric(gdf["lon"], errors="coerce")
        gdf = gdf.dropna(subset=["lat", "lon"])
        return gdf.reset_index(drop=True)
    except Exception as e:
        st.error(f"初期GeoJSON読み込みエラー: {e}")
        return pd.DataFrame(columns=["lat", "lon", "name"])

if "shelters" not in st.session_state:
    geojson_path = "hinanjyo.geojson"
    st.session_state["shelters"] = load_initial_geojson(geojson_path)

if "label_col" not in st.session_state:
    st.session_state["label_col"] = "name"
if "map_style" not in st.session_state:
    st.session_state["map_style"] = "light"
if "ox_mode" not in st.session_state:
    st.session_state["ox_mode"] = "drive"
if "route" not in st.session_state:
    st.session_state["route"] = []
if "road_path" not in st.session_state:
    st.session_state["road_path"] = []

st.sidebar.header("避難所データ追加 (SHP/GeoJSON/CSV)")
st.sidebar.info(
    "スマホ利用時は『ファイルアプリ（Googleドライブ等）→ダウンロードして選択』推奨です。\n"
    "SHPは全ファイル一括（shp, shx, dbf, prj等）、GeoJSON, CSVもOK。"
)
uploaded_files = st.sidebar.file_uploader(
    "全ファイル一括選択可（SHP一式, GeoJSON, CSV混在OK）",
    type=["shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "csv"],
    accept_multiple_files=True
)
if uploaded_files:
    gdf = file_to_df(uploaded_files)
    if not gdf.empty:
        gdf = gdf[[c for c in gdf.columns if c in ["lat", "lon"] or gdf[c].dtype == 'O']].copy()
        st.session_state["shelters"] = pd.concat([st.session_state["shelters"], gdf], ignore_index=True)
        st.success(f"{len(gdf)}件の避難所を追加しました")
        st.session_state["label_col"] = guess_name_col(st.session_state["shelters"])

with st.sidebar.form(key="manual_add"):
    st.write("避難所を手動で追加")
    lat = st.number_input("緯度", value=KAMIJIMA_CENTER[0], format="%f")
    lon = st.number_input("経度", value=KAMIJIMA_CENTER[1], format="%f")
    name = st.text_input("避難所名", "新しい避難所")
    add_btn = st.form_submit_button("追加")
    if add_btn:
        st.session_state["shelters"] = pd.concat([
            st.session_state["shelters"],
            pd.DataFrame([{"lat": lat, "lon": lon, "name": str(name)}])
        ], ignore_index=True)

if st.sidebar.button("すべて削除"):
    st.session_state["shelters"] = load_initial_geojson("hinanjyo.geojson")
    st.session_state["route"] = []
    st.session_state["road_path"] = []
    st.session_state["label_col"] = "name"

csv_export = st.session_state["shelters"].to_csv(index=False)
st.sidebar.download_button("避難所CSVをダウンロード", csv_export, file_name="shelters.csv", mime="text/csv")

# サイドバー：道路種別・TSPルート計算まとめてフォーム
with st.sidebar.form("tsp_form"):
    st.markdown("---")
    st.header("TSPルート計算")
    mode_disp = st.selectbox("道路種別", ["車（drive推奨）", "徒歩（歩道のみ）"], index=0, key="sb_mode")
    st.session_state["ox_mode"] = "drive" if "車" in mode_disp else "walk"
    tsp_btn = st.form_submit_button("道路でTSP最短巡回ルート計算")

shelters_df = st.session_state["shelters"].copy()
shelters_df["lat"] = pd.to_numeric(shelters_df["lat"], errors="coerce")
shelters_df["lon"] = pd.to_numeric(shelters_df["lon"], errors="coerce")
label_candidates = [c for c in shelters_df.columns if shelters_df[c].dtype == "O"]
if len(label_candidates) == 0:
    label_candidates = ["name"]
st.session_state["label_col"] = st.selectbox(
    "地図ラベルに使う列を選んでください（おすすめ：名称）",
    label_candidates,
    index=label_candidates.index(st.session_state["label_col"]) if st.session_state["label_col"] in label_candidates else 0
)

map_style_dict = {
    "light": "light",
    "dark": "dark",
    "ストリート": "mapbox://styles/mapbox/streets-v12",
    "衛星写真": "mapbox://styles/mapbox/satellite-streets-v12",
    "アウトドア": "mapbox://styles/mapbox/outdoors-v12",
    "ナビ風": "mapbox://styles/mapbox/navigation-night-v1"
}
style_name = st.selectbox(
    "地図背景スタイル",
    list(map_style_dict.keys()),
    index=list(map_style_dict.keys()).index(st.session_state.get("map_style", "light"))
)
st.session_state["map_style"] = style_name

shelters_df = shelters_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

# --- 巡回施設選択（小さい見出し） ---
st.markdown(
    "<span style='font-size:14px;'>📋 巡回施設の選択</span>",
    unsafe_allow_html=True
)
if not shelters_df.empty:
    display_names = [
        f"{row[st.session_state['label_col']]} ({row['lat']:.5f},{row['lon']:.5f})"
        for _, row in shelters_df.iterrows()
    ]
    idx_to_name = {i: name for i, name in enumerate(display_names)}
    st.multiselect(
        "巡回対象にする施設を選択（複数選択可）",
        options=list(idx_to_name.keys()),
        format_func=lambda x: idx_to_name[x],
        key="multiselect_tsp"
    )
else:
    st.info("避難所データをまずアップロード・追加してください。")

# --- TSPボタンが押されたら処理 ---
if tsp_btn:
    selected = st.session_state.get("multiselect_tsp", [])
    if not selected or len(selected) < 2:
        st.warning("最低2か所以上の避難所を選択してください。")
        st.session_state["road_path"] = []
    else:
        df = shelters_df.iloc[selected].reset_index(drop=True)
        locs = list(zip(df["lat"], df["lon"]))
        with st.spinner("OSM道路情報を取得＆巡回ルートを計算中...（通信状況により数秒かかります）"):
            distmat, G, node_ids = create_road_distance_matrix(locs, mode=st.session_state["ox_mode"])
            if np.any(np.isinf(distmat)):
                st.error("一部の避難所間で道路がつながっていません。別の組合せで試してください。")
                st.session_state["road_path"] = []
            else:
                route = solve_tsp(distmat)
                route = [i for i in route if i < len(node_ids)]
                st.session_state["route"] = [selected[i] for i in route if i < len(selected)]
                total = sum([distmat[route[i], route[i+1]] for i in range(len(route)-1) if route[i]<len(distmat) and route[i+1]<len(distmat)])
                # 実際の経路ラインも取得
                full_path = []
                if G is not None and node_ids:
                    for i in range(len(route)-1):
                        try:
                            if node_ids[route[i]] is not None and node_ids[route[i+1]] is not None:
                                seg = nx.shortest_path(G, node_ids[route[i]], node_ids[route[i+1]], weight='length')
                                seg_coords = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in seg]
                                if i != 0:
                                    seg_coords = seg_coords[1:]
                                full_path.extend(seg_coords)
                        except Exception as e:
                            st.error(f"経路描画エラー: {e}")
                            continue
                else:
                    # 直線距離の場合は点列を直線でつなぐ
                    for i in range(len(route)-1):
                        full_path.append([df.loc[route[i], "lon"], df.loc[route[i], "lat"]])
                        full_path.append([df.loc[route[i+1], "lon"], df.loc[route[i+1], "lat"]])
                st.session_state["road_path"] = full_path
                st.success(f"巡回ルート計算完了！総距離: {total:.2f} km（道路距離）")

# --- 地図（小さい見出し） ---
st.markdown(
    "<span style='font-size:14px;'>🗺️ 地図（全避難所ラベル付き・TSP道路ルート表示）</span>",
    unsafe_allow_html=True
)

layer_pts = pdk.Layer(
    "ScatterplotLayer",
    data=shelters_df,
   

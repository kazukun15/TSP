import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import tempfile
import os
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="避難所3D最適ルートアプリ", layout="wide")
st.title("🏫 避難所3D最適ルートアプリ（SHP/GeoJSON/CSV対応）")

# -------------------------
# 共通関数
# -------------------------
def guess_name_col(df):
    for cand in ["name", "NAME", "名称", "避難所", "施設名"]:
        if cand in df.columns:
            return cand
    return df.columns[0]

def file_to_df(uploaded_files):
    # 複数ファイルまとめて扱う
    if any(f.name.endswith(".shp") for f in uploaded_files):
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as out:
                    out.write(file.getvalue())
            shp_path = [os.path.join(temp_dir, f.name) for f in uploaded_files if f.name.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)
            if gdf.geometry.iloc[0].geom_type == "Point":
                gdf["lat"] = gdf.geometry.y
                gdf["lon"] = gdf.geometry.x
            else:
                st.warning("Point以外は非対応です")
                return pd.DataFrame(columns=["lat", "lon", "name"])
            name_col = guess_name_col(gdf)
            return gdf[["lat","lon",name_col]].rename(columns={name_col: "name"})
    elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
        # GeoJSON
        geojson_file = [f for f in uploaded_files if f.name.endswith((".geojson",".json"))][0]
        gdf = gpd.read_file(geojson_file)
        if gdf.geometry.iloc[0].geom_type == "Point":
            gdf["lat"] = gdf.geometry.y
            gdf["lon"] = gdf.geometry.x
        else:
            st.warning("Point以外は非対応です")
            return pd.DataFrame(columns=["lat", "lon", "name"])
        name_col = guess_name_col(gdf)
        return gdf[["lat","lon",name_col]].rename(columns={name_col: "name"})
    elif any(f.name.endswith(".csv") for f in uploaded_files):
        csv_file = [f for f in uploaded_files if f.name.endswith(".csv")][0]
        df = pd.read_csv(csv_file)
        if not set(["lat","lon"]).issubset(df.columns):
            st.warning("lat, lon 列が必要です")
            return pd.DataFrame(columns=["lat", "lon", "name"])
        name_col = guess_name_col(df)
        return df[["lat","lon",name_col]].rename(columns={name_col: "name"})
    else:
        st.warning("SHP/GeoJSON/CSVのみ対応です")
        return pd.DataFrame(columns=["lat", "lon", "name"])

def create_distance_matrix(locations):
    n = len(locations)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
    return mat

def solve_tsp(distance_matrix):
    size = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node]*10000)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    route = []
    if solution:
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(route[0])
    return route

# -------------------------
# セッション管理
# -------------------------
if "shelters" not in st.session_state:
    st.session_state["shelters"] = pd.DataFrame(columns=["lat", "lon", "name"])
if "route" not in st.session_state:
    st.session_state["route"] = []

# -------------------------
# ファイルアップロード
# -------------------------
st.sidebar.header("避難所データ追加 (SHP/GeoJSON/CSV)")

uploaded_files = st.sidebar.file_uploader(
    "全ファイル一括選択可能です（例: SHP一式, GeoJSON, CSV混在OK）",
    type=["shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "csv"],
    accept_multiple_files=True
)
if uploaded_files:
    df = file_to_df(uploaded_files)
    if not df.empty:
        st.session_state["shelters"] = pd.concat([st.session_state["shelters"], df], ignore_index=True).drop_duplicates(subset=["lat","lon","name"], keep="first")
        st.success(f"{len(df)}件の避難所を追加しました")

# 手動追加
with st.sidebar.form(key="manual_add"):
    st.write("避難所を手動で追加")
    lat = st.number_input("緯度", value=34.2832, format="%f")
    lon = st.number_input("経度", value=133.1831, format="%f")
    name = st.text_input("避難所名", "新しい避難所")
    add_btn = st.form_submit_button("追加")
    if add_btn:
        st.session_state["shelters"] = pd.concat([
            st.session_state["shelters"],
            pd.DataFrame([{"lat": lat, "lon": lon, "name": name}])
        ], ignore_index=True)

# 全削除
if st.sidebar.button("すべて削除"):
    st.session_state["shelters"] = pd.DataFrame(columns=["lat", "lon", "name"])
    st.session_state["route"] = []

# CSVエクスポート
csv_export = st.session_state["shelters"].to_csv(index=False)
st.sidebar.download_button("避難所CSVをダウンロード", csv_export, file_name="shelters.csv", mime="text/csv")

# -------------------------
# TSPルート計算
# -------------------------
st.sidebar.header("巡回ルート計算")
if st.sidebar.button("最短ルート計算（TSP）"):
    df = st.session_state["shelters"]
    if len(df) < 2:
        st.sidebar.warning("2か所以上の避難所が必要です")
    else:
        locs = list(zip(df["lat"], df["lon"]))
        distmat = create_distance_matrix(locs)
        route = solve_tsp(distmat)
        st.session_state["route"] = route
        total = sum([distmat[route[i], route[i+1]] for i in range(len(route)-1)])
        st.sidebar.success(f"総距離: {total:.2f} km（直線距離）")

# -------------------------
# 3D地図表示
# -------------------------
st.header("🗺️ 3D地図ビュー")
df = st.session_state["shelters"]
route = st.session_state["route"]

layer_pts = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_color='[0, 150, 255, 200]',
    get_radius=150,
    pickable=True,
)
layers = [layer_pts]

if route and len(route) > 1:
    coords = [[df.iloc[i]["lon"], df.iloc[i]["lat"]] for i in route]
    layer_line = pdk.Layer(
        "LineLayer",
        data=pd.DataFrame({"start": coords[:-1], "end": coords[1:]}),
        get_source_position="start",
        get_target_position="end",
        get_width=5,
        get_color=[255, 40, 40, 180],
    )
    layers.append(layer_line)

view = pdk.ViewState(
    latitude=float(df["lat"].mean()) if len(df) > 0 else 34.2832,
    longitude=float(df["lon"].mean()) if len(df) > 0 else 133.1831,
    zoom=13,
    pitch=45,
    bearing=0,
)

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view,
    tooltip={"text": "{name}"}
))

with st.expander("避難所リスト/巡回順"):
    st.dataframe(df)
    if route:
        st.write("巡回順（0起点）:", route)

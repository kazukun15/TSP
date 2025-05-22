import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import tempfile
import os
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# 上島町役場の座標
KAMIJIMA_CENTER = (34.2224, 133.2005)

st.set_page_config(page_title="避難所TSPルート（上島町役場中心）", layout="wide")
st.title("🏫 避難所TSPルートアプリ（中心：上島町役場）")

# --------------- 共通関数 ---------------
def guess_name_col(df):
    for cand in ["name", "NAME", "名称", "避難所", "施設名"]:
        if cand in df.columns:
            return cand
    return df.columns[0]

def file_to_df(uploaded_files):
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
            df = gdf[["lat","lon",name_col]].rename(columns={name_col: "name"})
            df["name"] = df["name"].astype(str)
            return df
    elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
        geojson_file = [f for f in uploaded_files if f.name.endswith((".geojson",".json"))][0]
        gdf = gpd.read_file(geojson_file)
        if gdf.geometry.iloc[0].geom_type == "Point":
            gdf["lat"] = gdf.geometry.y
            gdf["lon"] = gdf.geometry.x
        else:
            st.warning("Point以外は非対応です")
            return pd.DataFrame(columns=["lat", "lon", "name"])
        name_col = guess_name_col(gdf)
        df = gdf[["lat","lon",name_col]].rename(columns={name_col: "name"})
        df["name"] = df["name"].astype(str)
        return df
    elif any(f.name.endswith(".csv") for f in uploaded_files):
        csv_file = [f for f in uploaded_files if f.name.endswith(".csv")][0]
        df = pd.read_csv(csv_file)
        if not set(["lat","lon"]).issubset(df.columns):
            st.warning("lat, lon 列が必要です")
            return pd.DataFrame(columns=["lat", "lon", "name"])
        name_col = guess_name_col(df)
        df = df[["lat","lon",name_col]].rename(columns={name_col: "name"})
        df["name"] = df["name"].astype(str)
        return df
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

# --------------- セッション管理 ---------------
if "shelters" not in st.session_state:
    # 初期値は上島町役場だけ
    st.session_state["shelters"] = pd.DataFrame([
        {"lat": KAMIJIMA_CENTER[0], "lon": KAMIJIMA_CENTER[1], "name": "上島町役場"}
    ])
if "selected" not in st.session_state:
    st.session_state["selected"] = []
if "route" not in st.session_state:
    st.session_state["route"] = []

# --------------- ファイルアップロード ---------------
st.sidebar.header("避難所データ追加 (SHP/GeoJSON/CSV)")
uploaded_files = st.sidebar.file_uploader(
    "全ファイル一括選択可（SHP一式, GeoJSON, CSV混在OK）",
    type=["shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "csv"],
    accept_multiple_files=True
)
if uploaded_files:
    df = file_to_df(uploaded_files)
    if not df.empty:
        st.session_state["shelters"] = pd.concat([st.session_state["shelters"], df], ignore_index=True).drop_duplicates(subset=["lat","lon","name"], keep="first")
        st.session_state["shelters"]["name"] = st.session_state["shelters"]["name"].astype(str)
        st.success(f"{len(df)}件の避難所を追加しました")

# 手動追加
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
        st.session_state["shelters"]["name"] = st.session_state["shelters"]["name"].astype(str)

# 全削除
if st.sidebar.button("すべて削除"):
    st.session_state["shelters"] = pd.DataFrame([
        {"lat": KAMIJIMA_CENTER[0], "lon": KAMIJIMA_CENTER[1], "name": "上島町役場"}
    ])
    st.session_state["selected"] = []
    st.session_state["route"] = []

# CSVエクスポート
csv_export = st.session_state["shelters"].to_csv(index=False)
st.sidebar.download_button("避難所CSVをダウンロード", csv_export, file_name="shelters.csv", mime="text/csv")

# --------------- 選択避難所チェックUI ---------------
st.header("📋 避難所リストから計算対象を選択")
shelters_df = st.session_state["shelters"].copy()
# 型変換
shelters_df["lat"] = pd.to_numeric(shelters_df["lat"], errors="coerce")
shelters_df["lon"] = pd.to_numeric(shelters_df["lon"], errors="coerce")
shelters_df["name"] = shelters_df["name"].astype(str)
shelters_df = shelters_df.dropna(subset=["lat", "lon"])

if not shelters_df.empty:
    select_labels = [f"{row['name']} ({row['lat']:.5f},{row['lon']:.5f})" for _, row in shelters_df.iterrows()]
    selected_labels = st.multiselect(
        "巡回したい避難所に✔を入れてください（順序は自動で最適化されます）",
        options=select_labels,
        default=[select_labels[i] for i in st.session_state["selected"]] if st.session_state["selected"] else select_labels
    )
    selected_idx = [select_labels.index(lab) for lab in selected_labels]
    st.session_state["selected"] = selected_idx
else:
    st.info("避難所データをまずアップロード・追加してください。")

# --------------- TSPルート計算 ---------------
st.header("🚩 最短巡回ルート計算・地図表示")
if st.button("選択避難所でTSP最短巡回ルート計算"):
    selected = st.session_state["selected"]
    if not selected or len(selected) < 2:
        st.warning("最低2か所以上の避難所を選択してください。")
    else:
        df = shelters_df.iloc[selected].reset_index(drop=True)
        locs = list(zip(df["lat"], df["lon"]))
        distmat = create_distance_matrix(locs)
        route = solve_tsp(distmat)
        st.session_state["route"] = [selected[i] for i in route]
        total = sum([distmat[route[i], route[i+1]] for i in range(len(route)-1)])
        st.success(f"巡回ルート計算完了！総距離: {total:.2f} km（直線距離）")

# --------------- 3D地図表示 ---------------
df = shelters_df
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

if route and len(route) > 1 and all(i < len(df) for i in route):
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

# 常に上島町役場中心で表示
view = pdk.ViewState(
    latitude=KAMIJIMA_CENTER[0],
    longitude=KAMIJIMA_CENTER[1],
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
    if route and all(i < len(df) for i in route):
        st.write("巡回順（0起点）:", [df.iloc[i]['name'] for i in route])

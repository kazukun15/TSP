import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="避難所3D巡回ルートアプリ", layout="wide")
st.title("🏫 避難所3D巡回ルートアプリ")

# --- セッション初期化 ---
if "shelters" not in st.session_state:
    st.session_state["shelters"] = pd.DataFrame(columns=["lat", "lon", "name"])
if "route" not in st.session_state:
    st.session_state["route"] = []

# --- 避難所データ入力 ---
st.sidebar.header("避難所データ")
# CSVアップロード（ヘッダ：lat,lon,name）
uploaded = st.sidebar.file_uploader("CSVアップロード (lat,lon,name)", type="csv")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if set(["lat", "lon"]).issubset(df.columns):
            st.session_state["shelters"] = df
            st.success("CSV読込成功")
        else:
            st.warning("lat, lon 列が必要です")
    except Exception as e:
        st.error(f"CSVエラー: {e}")

# 手動追加
with st.sidebar.form(key="manual"):
    st.write("避難所を手動で追加")
    lat = st.number_input("緯度", value=34.2832, format="%f")
    lon = st.number_input("経度", value=133.1831, format="%f")
    name = st.text_input("避難所名", "新しい避難所")
    submit = st.form_submit_button("追加")
    if submit:
        st.session_state["shelters"] = pd.concat([
            st.session_state["shelters"],
            pd.DataFrame([{"lat": lat, "lon": lon, "name": name}])
        ], ignore_index=True)

# 削除ボタン
if st.sidebar.button("すべての避難所を削除"):
    st.session_state["shelters"] = pd.DataFrame(columns=["lat", "lon", "name"])
    st.session_state["route"] = []

# --- 巡回路最適化（TSP） ---
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
        return int(distance_matrix[from_node][to_node]*10000)  # 距離の丸め
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

# --- ルート計算ボタン ---
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

# --- 3D地図表示 ---
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
    pitch=45,  # 3D感
    bearing=0,
)

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view,
    tooltip={"text": "{name}"}
))

# --- データ表示 ---
with st.expander("避難所リスト/順序を見る"):
    st.dataframe(df)
    if route:
        st.write("巡回順（0起点）:", route)

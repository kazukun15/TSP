import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import geopandas as gpd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import tempfile
import os
import numpy as np

ox.config(log_console=True, use_cache=True)

# 初期設定
def initialize():
    if "shelters" not in st.session_state:
        st.session_state["shelters"] = []
    if "route" not in st.session_state:
        st.session_state["route"] = []
    if "calculated" not in st.session_state:
        st.session_state["calculated"] = False

# マップ生成
def create_map():
    return folium.Map(location=[34.2832, 133.1831], zoom_start=13)

# 避難所表示
def display_shelters(map_obj, shelters):
    for shelter in shelters:
        folium.Marker(
            [shelter["lat"], shelter["lon"]],
            popup=shelter["name"],
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(map_obj)

# データ読み込み
def load_shelters(files):
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
        shp_file = next(f for f in files if f.name.endswith('.shp'))
        gdf = gpd.read_file(os.path.join(temp_dir, shp_file.name))
        shelters = [
            {"lat": geom.y, "lon": geom.x, "name": row.get("name", "避難所")}
            for geom, row in zip(gdf.geometry, gdf.to_dict('records'))
        ]
    return shelters

# ネットワーク取得（広めに確保）
@st.cache_data
def get_network(shelters, mode):
    lats = [s["lat"] for s in shelters]
    lons = [s["lon"] for s in shelters]
    padding = 0.005
    G = ox.graph_from_bbox(
        north=max(lats)+padding, south=min(lats)-padding,
        east=max(lons)+padding, west=min(lons)-padding,
        network_type=mode
    )
    return G

# 距離行列作成（堅牢に）
def create_distance_matrix(G, shelters):
    nodes = [ox.nearest_nodes(G, s["lon"], s["lat"]) for s in shelters]
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    dist = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')
                except Exception:
                    dist = float('inf')
                dist_matrix[i][j] = dist
    return dist_matrix, nodes

# OR-ToolsでTSP解決（高精度）
def solve_tsp(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = solution.Value(routing.NextVar(index))
        route.append(route[0])  # return to start
    return route

# Streamlit UI構成
def main():
    st.set_page_config(layout="wide", page_title="避難所巡回ルート最適化アプリ")
    initialize()

    st.title("🗺️ 避難所巡回ルート最適化アプリ")

    tabs = st.tabs(["📍 地図表示", "📂 データアップロード", "➕ 避難所追加", "🚴 ルート計算"])

    # 地図表示
    with tabs[0]:
        map_obj = create_map()
        display_shelters(map_obj, st.session_state["shelters"])
        if st.session_state["calculated"]:
            route_coords = [
                (st.session_state["shelters"][i]["lat"], st.session_state["shelters"][i]["lon"])
                for i in st.session_state["route"]
            ]
            folium.PolyLine(route_coords, color="red", weight=3).add_to(map_obj)
        st_folium(map_obj, width=800, height=500)

    # データアップロード
    with tabs[1]:
        files = st.file_uploader("避難所データ（SHP）をアップロードしてください", accept_multiple_files=True, type=["shp","shx","dbf","prj"])
        if st.button("データ読込"):
            if files:
                shelters = load_shelters(files)
                st.session_state["shelters"].extend(shelters)
                st.success(f"{len(shelters)}件の避難所を追加しました。")
            else:
                st.error("ファイルをアップロードしてください。")

    # 手動追加
    with tabs[2]:
        lat = st.number_input("緯度", value=34.2832, format="%f")
        lon = st.number_input("経度", value=133.1831, format="%f")
        name = st.text_input("避難所名", "新規避難所")
        if st.button("避難所を追加"):
            st.session_state["shelters"].append({"lat": lat, "lon": lon, "name": name})
            st.success("避難所を追加しました。")

    # TSP計算
    with tabs[3]:
        mode = st.selectbox("移動手段", ["walk", "bike", "drive"], format_func=lambda x: {"walk":"徒歩","bike":"自転車","drive":"自動車"}[x])
        if st.button("ルート計算"):
            if len(st.session_state["shelters"]) < 2:
                st.error("最低2つの避難所が必要です。")
            else:
                with st.spinner("計算中…"):
                    G = get_network(st.session_state["shelters"], mode)
                    dist_matrix, nodes = create_distance_matrix(G, st.session_state["shelters"])
                    route = solve_tsp(dist_matrix)
                    st.session_state["route"] = route
                    st.session_state["calculated"] = True
                    route_length = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                    st.success(f"ルート計算完了（総距離: {route_length/1000:.2f} km）")

if __name__ == "__main__":
    main()

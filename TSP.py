import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, sin, cos, sqrt, atan2
import pydeck as pdk
import time

# 定数
EARTH_RADIUS_KM = 6371
KAMIJIMA_CENTER = (34.2124, 132.9994)

def haversine(lat1, lon1, lat2, lon2):
    """2点間の球面距離（km）を計算"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c

def create_distance_matrix(locs, progress_bar, status_text):
    """地点リストから距離行列を作成"""
    n = len(locs)
    mat = np.full((n, n), np.inf)
    G = None
    nodes = [None] * n

    # OSMnxグラフの取得
    status_text.text("OSM道路ネットワークを取得中...")
    try:
        lats, lons = zip(*locs)
        north, south = max(lats) + 0.01, min(lats) - 0.01
        east, west = max(lons) + 0.01, min(lons) - 0.01
        G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
        nodes = [ox.nearest_nodes(G, lon, lat) for lat, lon in locs]
    except Exception as e:
        st.warning(f"道路ネットワークの取得に失敗しました。直線距離を使用します。詳細: {e}")

    # 距離行列の作成
    status_text.text("距離行列を作成中...")
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                if G is not None and nodes[i] is not None and nodes[j] is not None:
                    try:
                        length = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')
                        mat[i][j] = length / 1000  # m → km
                    except:
                        mat[i][j] = haversine(*locs[i], *locs[j])
                else:
                    mat[i][j] = haversine(*locs[i], *locs[j])
        progress_bar.progress((i + 1) / n)
        time.sleep(0.01)  # UI更新のための短いスリープ

    return mat, G, nodes

def solve_tsp(distance_matrix):
    """距離行列からTSP経路を計算"""
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(route[0])
        return route
    else:
        return None

def visualize_route(df, path):
    """経路をpydeckで可視化"""
    view = pdk.ViewState(
        latitude=df["lat"].mean() if not df.empty else KAMIJIMA_CENTER[0],
        longitude=df["lon"].mean() if not df.empty else KAMIJIMA_CENTER[1],
        zoom=13,
        pitch=45
    )

    route_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path}],
        get_path="path",
        get_color=[0, 128, 255],
        width_scale=10,
        width_min_pixels=2
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=30,
        get_fill_color=[255, 0, 0],
        pickable=True
    )

    tooltip = {"html": "<b>避難所</b>: {name}", "style": {"color": "white"}}
    st.pydeck_chart(pdk.Deck(layers=[route_layer, scatter_layer], initial_view_state=view, tooltip=tooltip))

# Streamlitアプリケーション
st.title("避難所最短ルート探すくん")

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください（列名: lat, lon, name）", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not {"lat", "lon"}.issubset(df.columns):
            st.error("CSVファイルに 'lat' および 'lon' 列が必要です。")
        else:
            if "name" not in df.columns:
                df["name"] = [f"避難所{i+1}" for i in range(len(df))]
            locs = df[["lat", "lon"]].values.tolist()

            progress_bar = st.progress(0)
            status_text = st.empty()

            distance_matrix, G, nodes = create_distance_matrix(locs, progress_bar, status_text)

            status_text.text("TSP経路を計算中...")
            route = solve_tsp(distance_matrix)
            if route is None:
                st.warning("TSP計算に失敗しました。")
            else:
                total_distance = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                st.success(f"総距離: {total_distance:.2f} km")

                # 経路の座標リストを作成
                path = []
                for i in range(len(route) - 1):
                    start_idx = route[i]
                    end_idx = route[i + 1]
                    if G is not None and nodes[start_idx] is not None and nodes[end_idx] is not None:
                        try:
                            shortest_path = nx.shortest_path(G, nodes[start_idx], nodes[end_idx], weight='length')
                            segment = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in shortest_path]
                            if path:
                                path.extend(segment[1:])  # 重複を避ける
                            else:
                                path.extend(segment)
                            continue
                        except:
                            pass
                    # フォールバックとして直線を使用
                    path.extend([
                        [df.loc[start_idx, "lon"], df.loc[start_idx, "lat"]],
                        [df.loc[end_idx, "lon"], df.loc[end_idx, "lat"]]
                    ])

                visualize_route(df, path)

            progress_bar.empty()
            status_text.empty()
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
else:
    st.info("CSVファイルをアップロードしてください。")

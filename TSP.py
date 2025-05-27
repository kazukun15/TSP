import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import networkx as nx
import osmnx as ox
from math import radians, sin, cos, sqrt, atan2
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Constants
EPSG_WGS84 = 4326
GEOJSON_LOCAL = "hinanjyo.geojson"
DEFAULT_CENTER = (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
st.title("🏫 避難所最短ルート探すくん")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

@st.cache_data
def load_initial_geojson():
    if os.path.exists(GEOJSON_LOCAL):
        gdf = gpd.read_file(GEOJSON_LOCAL).to_crs(EPSG_WGS84)
        gdf = gdf[gdf.geometry.type == "Point"].copy()
        gdf["lon"], gdf["lat"] = gdf.geometry.x, gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry"))
    return pd.DataFrame(columns=["lat", "lon", "name"])

@st.cache_data
def file_to_df(files):
    try:
        f = files[0]
        ext = f.name.split('.')[-1].lower()
        if ext in ["shp", "geojson", "json"]:
            gdf = gpd.read_file(f).to_crs(EPSG_WGS84)
            gdf = gdf[gdf.geometry.type == "Point"].copy()
            gdf["lon"], gdf["lat"] = gdf.geometry.x, gdf.geometry.y
            return pd.DataFrame(gdf.drop(columns="geometry"))
        elif ext == "csv":
            df = pd.read_csv(f)
            if not {"lat", "lon"}.issubset(df.columns):
                st.error("CSVにはlat, lon列が必要です")
                return pd.DataFrame()
            return df.dropna(subset=["lat", "lon"])
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame()

@st.cache_data
def create_road_distance_matrix(locs, mode="drive"):
    pad = 0.03
    north, south = max(lat for lat, _ in locs) + pad, min(lat for lat, _ in locs) - pad
    east, west = max(lon for _, lon in locs) + pad, min(lon for _, lon in locs) - pad
    G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type=mode)

    nodes = [ox.nearest_nodes(G, lon, lat) for lat, lon in locs]
    n = len(nodes)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    mat[i, j] = nx.shortest_path_length(G, nodes[i], nodes[j], weight="length") / 1000
                except nx.NetworkXNoPath:
                    mat[i, j] = haversine(locs[i][0], locs[i][1], locs[j][0], locs[j][1])
    return mat, G, nodes

@st.cache_data
def solve_tsp(dist_mat):
    n = len(dist_mat)
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def distance_callback(from_index, to_index):
        return int(dist_mat[mgr.IndexToNode(from_index), mgr.IndexToNode(to_index)] * 1e5)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 10
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(params)
    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(mgr.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(route[0])
    return route

# Session states
if "shelters" not in st.session_state:
    st.session_state.shelters = load_initial_geojson()

uploaded_files = st.sidebar.file_uploader("避難所データ追加", type=["shp", "geojson", "json", "csv"], accept_multiple_files=True)
if uploaded_files:
    df_new = file_to_df(uploaded_files)
    if not df_new.empty:
        st.session_state.shelters = pd.concat([st.session_state.shelters, df_new], ignore_index=True)

st.sidebar.header("経路計算")
mode = st.sidebar.selectbox("移動手段", ["drive", "walk"], format_func=lambda x: {"drive": "自動車", "walk": "徒歩"}[x])

if st.sidebar.button("最短経路計算"):
    locs = list(zip(st.session_state.shelters.lat, st.session_state.shelters.lon))
    dist_mat, G, nodes = create_road_distance_matrix(locs, mode)
    route = solve_tsp(dist_mat)
    total_distance = sum(dist_mat[route[i], route[i+1]] for i in range(len(route)-1))
    st.sidebar.success(f"総距離: {total_distance:.2f} km")

    path_coords = [[locs[i][1], locs[i][0]] for i in route]
    layer = pdk.Layer("PathLayer", data=[{"path": path_coords}], get_path="path", get_color=[255, 0, 0], width_scale=20, width_min_pixels=3)
    view_state = pdk.ViewState(latitude=DEFAULT_CENTER[0], longitude=DEFAULT_CENTER[1], zoom=13)
    st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=[layer]))

st.dataframe(st.session_state.shelters)

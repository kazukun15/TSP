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

# ===== Utility Functions =====
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
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"] = gdf.index.astype(str)
        return pd.DataFrame(gdf[["name", "lat", "lon"]])
    return pd.DataFrame(columns=["name", "lat", "lon"])

@st.cache_data
def file_to_df(files):
    try:
        f = files[0]
        ext = f.name.split('.')[-1].lower()
        if ext in ["shp", "geojson", "json"]:
            gdf = gpd.read_file(f).to_crs(EPSG_WGS84)
            gdf = gdf[gdf.geometry.type == "Point"].copy()
            gdf["lon"] = gdf.geometry.x
            gdf["lat"] = gdf.geometry.y
            if "name" not in gdf.columns:
                gdf["name"] = gdf.index.astype(str)
            return pd.DataFrame(gdf[["name", "lat", "lon"]])
        elif ext == "csv":
            df = pd.read_csv(f)
            if not {"lat", "lon"}.issubset(df.columns):
                st.error("CSVにはlat, lon列が必要です")
                return pd.DataFrame()
            if "name" not in df.columns:
                df["name"] = df.index.astype(str)
            return df[["name", "lat", "lon"]].dropna(subset=["lat", "lon"])
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame()

@st.cache_data
def create_road_distance_matrix(locs, mode="drive"):
    pad = 0.03
    north = max(lat for lat, _ in locs) + pad
    south = min(lat for lat, _ in locs) - pad
    east  = max(lon for _, lon in locs) + pad
    west  = min(lon for _, lon in locs) - pad
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
    return mat, G

@st.cache_data
def solve_tsp(dist_mat):
    n = len(dist_mat)
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def distance_callback(from_index, to_index):
        return int(dist_mat[mgr.IndexToNode(from_index), mgr.IndexToNode(to_index)] * 1e5)
    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 10
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    route = []
    if sol:
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(mgr.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        route.append(route[0])
    return route

# ===== Main App =====
# Load or initialize shelters
if "shelters" not in st.session_state:
    st.session_state.shelters = load_initial_geojson()

# Sidebar: Data upload
uploaded = st.sidebar.file_uploader("避難所データ追加 (SHP/GeoJSON/CSV)", type=["shp","geojson","json","csv"], accept_multiple_files=True)
if uploaded:
    df_new = file_to_df(uploaded)
    if not df_new.empty:
        st.session_state.shelters = pd.concat([st.session_state.shelters, df_new], ignore_index=True)
        st.sidebar.success(f"{len(df_new)}件追加されました")

# Sidebar: Select mode and shelters
mode = st.sidebar.selectbox("移動手段を選択", ["drive","walk"], format_func=lambda x: {"drive":"自動車","walk":"徒歩"}[x])
choices = st.sidebar.multiselect(
    "巡回対象の避難所を選択",
    options=list(range(len(st.session_state.shelters))),
    format_func=lambda i: st.session_state.shelters.loc[i, 'name'],
    help="2つ以上選択してください"
)

# Calculate route
if st.sidebar.button("最短経路を計算"):
    if len(choices) < 2:
        st.sidebar.warning("2つ以上の避難所を選択してください")
    else:
        df_sel = st.session_state.shelters.loc[choices].reset_index(drop=True)
        locs = list(zip(df_sel['lat'], df_sel['lon']))
        dist_mat, G = create_road_distance_matrix(locs, mode)
        route = solve_tsp(dist_mat)
        # Compute distance
        total = sum(dist_mat[route[i], route[i+1]] for i in range(len(route)-1))
        st.sidebar.success(f"総距離: {total:.2f} km")
        # Prepare map layers
        route_coords = [[locs[i][1], locs[i][0]] for i in route]
        center_lat = df_sel['lat'].mean()
        center_lon = df_sel['lon'].mean()
        view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, pitch=45)
        layers = [
            pdk.Layer(
                "ScatterplotLayer", data=df_sel,
                get_position='[lon,lat]', get_radius=100,
                get_color=[255, 69, 0], pickable=True
            ),
            pdk.Layer(
                "PathLayer", data=[{"path": route_coords}],
                get_path='path', width_scale=20, width_min_pixels=4,
                get_color=[30, 144, 255], cap_style="round", joint_style="round"
            )
        ]
        st.pydeck_chart(pdk.Deck(
            initial_view_state=view,
            map_style='mapbox://styles/mapbox/streets-v11',
            layers=layers,
            tooltip={"text":"{name}"}
        ))

# Show table of shelters
st.header("避難所一覧")
st.dataframe(st.session_state.shelters)

# Optionally show selected order
if 'route' in locals() and route:
    order = [df_sel.loc[idx, 'name'] for idx in route]
    st.write("巡回順:", order)

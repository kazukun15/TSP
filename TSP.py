import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import networkx as nx
import osmnx as ox
import packaging.version
from math import radians, sin, cos, sqrt, atan2
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Constants
EPSG_PLANE = 2446
EPSG_WGS84 = 4326
GEOJSON_LOCAL = "hinanjyo.geojson"
GEOJSON_REMOTE = "https://raw.githubusercontent.com/<username>/<repo>/main/hinanjyo.geojson"
DEFAULT_CENTER = (34.25754417840102, 133.20446981161595)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def two_opt(route, dist):
    best = route[:]
    n = len(best)
    def cost(r): return sum(dist[r[i], r[i+1]] for i in range(n - 1))
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_r = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if cost(new_r) < cost(best):
                    best = new_r
                    improved = True
        route = best
    return best

def load_geojson():
    src = GEOJSON_LOCAL if os.path.exists(GEOJSON_LOCAL) else GEOJSON_REMOTE
    try:
        gdf = gpd.read_file(src).to_crs(epsg=EPSG_WGS84)
        gdf = gdf[gdf.geometry.type == "Point"].copy()
        gdf["lon"], gdf["lat"] = gdf.geometry.x, gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry"))
    except Exception as e:
        st.error(f"GeoJSON読み込みエラー: {e}")
        return pd.DataFrame(columns=["lat", "lon"])

@st.cache_data
def file_to_df(files):
    try:
        ext = files[0].name.split('.')[-1].lower()
        if ext in ["shp", "geojson", "json"]:
            gdf = gpd.read_file(files[0]).to_crs(epsg=EPSG_WGS84)
            gdf = gdf[gdf.geometry.type == "Point"].copy()
            gdf["lon"], gdf["lat"] = gdf.geometry.x, gdf.geometry.y
            return pd.DataFrame(gdf.drop(columns="geometry"))
        elif ext == "csv":
            df = pd.read_csv(files[0])
            if not {"lat", "lon"}.issubset(df.columns):
                st.error("CSVにlat,lon列が必要")
                return pd.DataFrame()
            return df.dropna(subset=["lat", "lon"])
    except Exception as e:
        st.error(f"ファイルエラー: {e}")
        return pd.DataFrame()

@st.cache_data
def create_road_matrix(locs, mode="drive"):
    pad = 0.05
    north, south, east, west = max(lat for lat, _ in locs) + pad, min(lat for lat, _ in locs) - pad, max(lon for _, lon in locs) + pad, min(lon for _, lon in locs) - pad
    G = ox.graph_from_bbox(north, south, east, west, network_type=mode)
    nodes = [ox.nearest_nodes(G, lon, lat) for lat, lon in locs]
    mat = np.array([[nx.shortest_path_length(G, nodes[i], nodes[j], weight="length") / 1000.0 if i != j else 0 for j in range(len(nodes))] for i in range(len(nodes))])
    return mat, G, nodes

@st.cache_data
def solve_tsp(dist_mat):
    n = len(dist_mat)
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def cb(f, t): return int(dist_mat[mgr.IndexToNode(f), mgr.IndexToNode(t)] * 1e5)
    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 10
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    return [mgr.IndexToNode(routing.Start(0))] + [mgr.IndexToNode(sol.Value(routing.NextVar(i))) for i in range(n - 1)] if sol else []

# Main app logic
if __name__ == '__main__':
    st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
    st.title("🏫 避難所最短ルート探すくん")

    # Load and manage data
    df = load_geojson()
    uploaded_files = st.sidebar.file_uploader("避難所データ追加", type=["shp", "geojson", "json", "csv"], accept_multiple_files=True)
    if uploaded_files:
        df = pd.concat([df, file_to_df(uploaded_files)], ignore_index=True)

    st.map(df)

    if st.sidebar.button("リセット"):
        st.experimental_rerun()

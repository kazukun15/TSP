import streamlit as st
import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, sin, cos, sqrt, atan2

# ハーバーサイン公式
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径 (km)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# 距離行列生成
def create_road_distance_matrix(locs, mode="drive"):
    n = len(locs)
    ver = ox.__version__
    G = None
    for pad in [0.01, 0.03, 0.07]:
        north, south = max(lat for lat,_ in locs)+pad, min(lat for lat,_ in locs)-pad
        east, west   = max(lon for _,lon in locs)+pad, min(lon for _,lon in locs)-pad
        try:
            if ver < "2.0.0":
                G = ox.graph_from_bbox(north,south,east,west,network_type=mode)
            else:
                G = ox.graph_from_bbox(bbox=(north,south,east,west),network_type=mode)
            if G and G.nodes: break
        except: G = None

    nodes = []
    for lat, lon in locs:
        try: nid = ox.nearest_nodes(G, lon, lat) if G else None
        except: nid = None
        nodes.append(nid)

    mat = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i==j: mat[i, j]=0
            else:
                ni, nj = nodes[i], nodes[j]
                if G and ni is not None and nj is not None:
                    try:
                        d = nx.shortest_path_length(G, ni, nj, weight="length")
                        mat[i, j] = d / 1000.0
                        continue
                    except: pass
                mat[i, j] = haversine(locs[i][0], locs[i][1], locs[j][0], locs[j][1])
    return mat, G, nodes

# inf埋めとTSPソルバー
def solve_tsp(mat):
    max_val = np.nanmax(np.where(np.isinf(mat), -np.inf, mat))
    safe_mat = np.where(np.isinf(mat), max_val*1.5, mat)
    n=len(safe_mat)
    mgr=pywrapcp.RoutingIndexManager(n,1,0)
    routing=pywrapcp.RoutingModel(mgr)
    def cb(f,t): return int(safe_mat[mgr.IndexToNode(f),mgr.IndexToNode(t)]*1e5)
    idx=routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds=2
    params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol=routing.SolveWithParameters(params)
    if not sol: return []
    route=[]; cur=routing.Start(0)
    while not routing.IsEnd(cur):
        route.append(mgr.IndexToNode(cur))
        cur=sol.Value(routing.NextVar(cur))
    route.append(route[0])
    return route

# メインアプリ
st.title("避難所最短ルート探すくん（上島町対応版）")
uploaded_file = st.file_uploader("CSVをアップロード (lat, lon 列必須)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "lat" in df.columns and "lon" in df.columns:
        locs = df[["lat","lon"]].values.tolist()
        mat, G, nodes = create_road_distance_matrix(locs)
        route = solve_tsp(mat)
        st.write("計算経路インデックス:", route)

        # 距離計算
        total_dist = 0
        for i in range(len(route)-1):
            total_dist += mat[route[i], route[i+1]]
        st.write(f"総距離: {total_dist:.2f} km")

        # 地図表示用 PathLayer 座標生成
        path=[]
        if G:
            for i in range(len(route)-1):
                ni, nj = nodes[route[i]], nodes[route[i+1]]
                if ni and nj:
                    try:
                        sp = nx.shortest_path(G, ni, nj, weight="length")
                        seg = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in sp]
                        path.extend(seg if not path else seg[1:])
                        continue
                    except: pass
                # fallback: 直線
                path.extend([
                    [df.loc[route[i],"lon"], df.loc[route[i],"lat"]],
                    [df.loc[route[i+1],"lon"], df.loc[route[i+1],"lat"]]
                ])
        else:
            for idx in route:
                path.append([df.loc[idx,"lon"], df.loc[idx,"lat"]])

        import pydeck as pdk
        view_state = pdk.ViewState(latitude=np.mean(df["lat"]),
                                   longitude=np.mean(df["lon"]),
                                   zoom=13, pitch=45)
        layer = pdk.Layer("PathLayer", data=[{"path":path}],
                           get_path="path", get_width=5, get_color=[255,0,0],
                           width_min_pixels=2)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state)
        st.pydeck_chart(r)
    else:
        st.error("lat, lon 列が見つかりません。")

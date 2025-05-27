import streamlit as st
import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, sin, cos, sqrt, atan2
import pydeck as pdk
import time

# 定数
EARTH_RADIUS_KM = 6371
EPSG_WGS84 = 4326
KAMIJIMA_CENTER = (34.2124, 132.9994)

def haversine(lat1, lon1, lat2, lon2):
    """2点間の球面距離（km）"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return EARTH_RADIUS_KM * c

def create_distance_matrix(locs, log_box, prog_bar, mode="drive"):
    """地点リストlocsから距離行列を作成（道路or直線）"""
    n = len(locs)
    ver = ox.__version__
    G = None
    log_box.text("OSM道路ネット取得中…")
    for pad in [0.01, 0.03, 0.07]:
        north, south = max(lat for lat,_ in locs)+pad, min(lat for lat,_ in locs)-pad
        east, west   = max(lon for _,lon in locs)+pad, min(lon for _,lon in locs)-pad
        try:
            G = ox.graph_from_bbox((north,south,east,west), network_type=mode) if ver>="2" else \
                ox.graph_from_bbox(north,south,east,west,network_type=mode)
            if G and G.nodes: break
        except: G = None
    log_box.text("最寄りノード探索中…")
    nodes = [ox.nearest_nodes(G, lon, lat) if G else None for lat, lon in locs]
    mat = np.full((n, n), np.inf)
    log_box.text("距離行列作成中…")
    for i in range(n):
        for j in range(n):
            if i==j: mat[i,j]=0
            else:
                ni, nj = nodes[i], nodes[j]
                if G and ni and nj:
                    try:
                        mat[i,j]=nx.shortest_path_length(G, ni, nj, weight="length")/1000
                        continue
                    except: pass
                mat[i,j]=haversine(*locs[i], *locs[j])
        prog_bar.progress((i+1)/n)
    return mat, G, nodes

def solve_tsp(mat):
    """距離行列からTSP経路を計算"""
    safe_mat = np.where(np.isinf(mat), np.nanmax(mat)*1.5, mat)
    n=len(safe_mat)
    mgr=pywrapcp.RoutingIndexManager(n,1,0)
    routing=pywrapcp.RoutingModel(mgr)
    cb=lambda f,t: int(safe_mat[mgr.IndexToNode(f),mgr.IndexToNode(t)]*1e5)
    idx=routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds=3
    params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol=routing.SolveWithParameters(params)
    if not sol: return []
    route=[]; cur=routing.Start(0)
    while not routing.IsEnd(cur):
        route.append(mgr.IndexToNode(cur))
        cur=sol.Value(routing.NextVar(cur))
    route.append(route[0])
    return route

def visualize_route(df, path, route):
    """pydeck PathLayer描画"""
    color = [0, 128, 255]  # 青
    view = pdk.ViewState(
        latitude=df["lat"].mean() if not df.empty else KAMIJIMA_CENTER[0],
        longitude=df["lon"].mean() if not df.empty else KAMIJIMA_CENTER[1],
        zoom=13, pitch=45
    )
    layer = pdk.Layer(
        "PathLayer",
        data=[{"path":path}],
        get_path="path",
        get_color=color,
        width_scale=10,
        width_min_pixels=2
    )
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=30,
        get_fill_color=[255, 0, 0],
        pickable=True
    )
    tooltip={"html": "<b>避難所</b>: {name}", "style": {"color": "white"}}
    st.pydeck_chart(pdk.Deck(layers=[layer, scatter], initial_view_state=view, tooltip=tooltip))

# UI
st.title("避難所最短ルート探すくん（全機能版）")
uploaded = st.file_uploader("CSVをアップロード（lat, lon, name列）", type=["csv"])
if uploaded:
    try:
        df=pd.read_csv(uploaded)
        if not {"lat","lon"}.issubset(df.columns):
            st.error("CSVにlat, lon列がありません。")
        else:
            if "name" not in df.columns:
                df["name"] = [f"避難所{i+1}" for i in range(len(df))]
            locs=df[["lat","lon"]].values.tolist()
            log_box=st.empty()
            prog_bar=st.progress(0)
            mat, G, nodes = create_distance_matrix(locs, log_box, prog_bar)
            log_box.text("TSP計算中…")
            route=solve_tsp(mat)
            if not route:
                st.warning("TSP計算失敗。経路が見つかりません。")
            else:
                total_dist=sum(mat[route[i],route[i+1]] for i in range(len(route)-1))
                st.success(f"総距離: {total_dist:.2f} km")
                # PathLayer用座標
                path=[]
                for i in range(len(route)-1):
                    ni, nj = nodes[route[i]], nodes[route[i+1]]
                    if G and ni and nj:
                        try:
                            sp = nx.shortest_path(G, ni, nj, weight="length")
                            seg = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in sp]
                            path.extend(seg if not path else seg[1:])
                            continue
                        except: pass
                    # fallback直線
                    path.extend([
                        [df.loc[route[i],"lon"], df.loc[route[i],"lat"]],
                        [df.loc[route[i+1],"lon"], df.loc[route[i+1],"lat"]]
                    ])
                visualize_route(df, path, route)
            prog_bar.empty()
    except Exception as e:
        st.error(f"ファイル読込エラー: {e}")
else:
    st.info("CSVをアップロードしてください。")

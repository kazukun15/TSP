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

# 定数
EPSG_PLANE    = 2446
EPSG_WGS84    = 4326
GEOJSON_LOCAL = "hinanjyo.geojson"
GEOJSON_REMOTE= "https://raw.githubusercontent.com/<ユーザー名>/<リポジトリ名>/main/hinanjyo.geojson"
DEFAULT_CENTER= (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
st.title("🏫 避難所最短ルート探すくん")

def haversine(lat1, lon1, lat2, lon2):
    R=6371.0
    φ1,λ1,φ2,λ2=map(radians,[lat1,lon1,lat2,lon2])
    dφ, dλ = φ2-φ1, λ2-λ1
    a = sin(dφ/2)**2+cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R*2*atan2(sqrt(a),sqrt(1-a))

def two_opt(route, dist):
    best=route[:]; n=len(best)
    def cost(r): return sum(dist[r[i],r[i+1]] for i in range(n-1))
    improved=True
    while improved:
        improved=False
        for i in range(1,n-2):
            for j in range(i+1,n-1):
                new_r=best[:i]+best[i:j+1][::-1]+best[j+1:]
                if cost(new_r)<cost(best):
                    best=new_r; improved=True
        route=best
    return best

def load_initial_geojson():
    src=GEOJSON_LOCAL if os.path.exists(GEOJSON_LOCAL) else GEOJSON_REMOTE
    try:
        gdf=gpd.read_file(src)
        if gdf.crs is None or gdf.crs.to_epsg()!=EPSG_PLANE:
            gdf=gdf.set_crs(epsg=EPSG_PLANE,allow_override=True)
        gdf=gdf.to_crs(epsg=EPSG_WGS84)
        gdf=gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"],gdf["lat"]=gdf.geometry.x,gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry"))
    except:
        return pd.DataFrame(columns=["lat","lon"])

def file_to_df(files):
    try:
        if any(f.name.endswith(".shp") for f in files):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for f in files: open(os.path.join(td,f.name),"wb").write(f.getvalue())
                shp=next(p for p in os.listdir(td) if p.endswith(".shp"))
                gdf=gpd.read_file(os.path.join(td,shp))
        elif any(f.name.endswith((".geojson",".json")) for f in files):
            f=next(f for f in files if f.name.endswith((".geojson",".json")))
            gdf=gpd.read_file(f)
        else:
            f=next(f for f in files if f.name.endswith(".csv"))
            df=pd.read_csv(f)
            if not {"lat","lon"}.issubset(df.columns):
                st.error("CSV に lat,lon 列が必要です"); return pd.DataFrame()
            df["lat"],df["lon"]=pd.to_numeric(df["lat"],errors="coerce"),pd.to_numeric(df["lon"],errors="coerce")
            if "name" not in df: df["name"]=df.index.astype(str)
            return df.dropna(subset=["lat","lon"])
        if gdf.crs is None or gdf.crs.to_epsg()!=EPSG_PLANE:
            gdf=gdf.set_crs(epsg=EPSG_PLANE,allow_override=True)
        gdf=gdf.to_crs(epsg=EPSG_WGS84)
        gdf=gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"],gdf["lat"]=gdf.geometry.x,gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry"))
    except Exception as e:
        st.error(f"ファイル読み込み失敗: {e}")
        return pd.DataFrame()

def create_road_distance_matrix(locs,mode="drive"):
    n=len(locs)
    ver=packaging.version.parse(ox.__version__)
    G=None
    for pad in [0.01,0.03,0.07]:
        north, south = max(lat for lat,_ in locs)+pad, min(lat for lat,_ in locs)-pad
        east, west   = max(lon for _,lon in locs)+pad, min(lon for _,lon in locs)-pad
        try:
            if ver<packaging.version.parse("2.0.0"):
                G=ox.graph_from_bbox(north,south,east,west,network_type=mode)
            else:
                G=ox.graph_from_bbox(bbox=(north,south,east,west),network_type=mode)
            if G.nodes: break
        except: G=None
    nodes=[]
    for lat,lon in locs:
        if G:
            try: nid=ox.nearest_nodes(G,lon,lat)
            except: nid=None
        else: nid=None
        nodes.append(nid)
    mat=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j: continue
            ni,nj=nodes[i],nodes[j]
            if G and ni is not None and nj is not None:
                try:
                    d=nx.shortest_path_length(G,ni,nj,weight="length")
                    mat[i,j]=d/1000.0; continue
                except: pass
            lat1,lon1=locs[i]; lat2,lon2=locs[j]
            mat[i,j]=haversine(lat1,lon1,lat2,lon2)
    return mat,G,nodes

def solve_tsp(dist_mat):
    n=len(dist_mat)
    mgr=pywrapcp.RoutingIndexManager(n,1,0)
    routing=pywrapcp.RoutingModel(mgr)
    def cb(f,t): return int(dist_mat[mgr.IndexToNode(f),mgr.IndexToNode(t)]*1e5)
    idx=routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds=1
    params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sol=routing.SolveWithParameters(params)
    if not sol: return []
    route=[]; cur=routing.Start(0)
    while not routing.IsEnd(cur):
        route.append(mgr.IndexToNode(cur)); cur=sol.Value(routing.NextVar(cur))
    route.append(route[0])
    return two_opt(route, dist_mat)

# セッション
if "shelters"   not in st.session_state: st.session_state.shelters  = load_initial_geojson()
if "route"      not in st.session_state: st.session_state.route     = []
if "road_path"  not in st.session_state: st.session_state.road_path = []

# サイドバー: データ追加
st.sidebar.header("避難所データ追加")
up=st.sidebar.file_uploader("SHP/GeoJSON/CSV→一括",type=["shp","geojson","json","csv"],accept_multiple_files=True)
if up:
    df_new=file_to_df(up)
    if not df_new.empty:
        st.session_state.shelters=pd.concat([st.session_state.shelters,df_new],ignore_index=True)
        st.sidebar.success(f"{len(df_new)} 件追加")
with st.sidebar.form("manual"):
    lat=st.number_input("緯度",value=DEFAULT_CENTER[0],format="%f")
    lon=st.number_input("経度",value=DEFAULT_CENTER[1],format="%f")
    name=st.text_input("名前","新規避難所")
    if st.form_submit_button("追加"):
        st.session_state.shelters=pd.concat([
            st.session_state.shelters,
            pd.DataFrame([{"lat":lat,"lon":lon,"name":name}])
        ],ignore_index=True)
if st.sidebar.button("リセット"):
    st.session_state.shelters=load_initial_geojson()
    st.session_state.route=[]
    st.session_state.road_path=[]

# サイドバー: TSP 設定
df=st.session_state.shelters.copy()
df["lat"],df["lon"]=pd.to_numeric(df["lat"],errors="coerce"),pd.to_numeric(df["lon"],errors="coerce")
label_cols=list(df.columns)
label_col=st.sidebar.selectbox("ラベル列",label_cols,index=label_cols.index("name") if "name" in label_cols else 0)
buffer_m=st.sidebar.slider("バッファ(m)",0,3000,0,100)
map_styles={"Light":"light","Dark":"dark","Streets":"mapbox://styles/mapbox/streets-v12","Satellite":"mapbox://styles/mapbox/satellite-streets-v12"}
map_style=st.sidebar.selectbox("地図スタイル",list(map_styles.keys()))
mode_opt=st.sidebar.selectbox("道路種別",["drive","walk"],format_func=lambda x:{"drive":"自動車","walk":"徒歩"}[x])

# 施設選択 & TSP
st.markdown("<span style='font-size:14px;'>📋 巡回施設を選択</span>",unsafe_allow_html=True)
labels=[f"{r[label_col]}({r['lat']:.5f},{r['lon']:.5f})" for _,r in df.iterrows()]
idx_map={i:labels[i] for i in range(len(labels))}
sel=st.multiselect("選択",options=list(idx_map.keys()),format_func=lambda x:idx_map[x],key="sel")
if st.sidebar.button("TSP 計算"):
    if len(sel)<2:
        st.sidebar.warning("2か所以上選択して下さい")
    else:
        sub=df.iloc[sel].reset_index(drop=True)
        locs=list(zip(sub["lat"],sub["lon"]))
        dist_mat,G,nodes=create_road_distance_matrix(locs,mode_opt)
        route=solve_tsp(dist_mat)
        if not route:
            st.sidebar.error("解が見つかりませんでした")
        else:
            st.session_state.route=[sel[i] for i in route]
            # 経路座標生成
            rp=[]
            for k in range(len(route)-1):
                i,j=route[k],route[k+1]
                ni,nj=nodes[i] if i<len(nodes) else None,nodes[j] if j<len(nodes) else None
                seg=[]
                if G and ni is not None and nj is not None:
                    try:
                        path_nodes=nx.shortest_path(G,ni,nj,weight="length")
                        seg=[[G.nodes[n]["x"],G.nodes[n]["y"]] for n in path_nodes]
                    except: seg=[]
                if not seg:
                    lon1,lat1=sub.loc[i,["lon","lat"]]; lon2,lat2=sub.loc[j,["lon","lat"]]
                    seg=[[lon1,lat1],[lon2,lat2]]
                if not rp: rp.extend(seg)
                else: rp.extend(seg[1:])
            st.session_state.road_path=rp
            total=sum(dist_mat[route[k],route[k+1]] for k in range(len(route)-1))
            st.sidebar.success(f"総距離: {total:.2f} km")

# 地図描画 + 先進的アニメーション
if not df.empty:
    ds=df.iloc[sel] if sel else df
    clat,clon=float(ds["lat"].mean()),float(ds["lon"].mean())
    slat,slon=ds["lat"].max()-ds["lat"].min(),ds["lon"].max()-ds["lon"].min()
    zoom=10 if slat>0.1 or slon>0.1 else 11.5 if slat>0.05 or slon>0.05 else 14 if len(ds)==1 else 13.2
else:
    clat,clon=DEFAULT_CENTER; zoom=13.2

layers=[]
if buffer_m>0:
    layers.append(pdk.Layer("CircleLayer",data=df.iloc[sel] if sel else df,
        get_position='[lon,lat]',get_radius=buffer_m,get_fill_color=[0,0,255,50]))
layers.append(pdk.Layer("ScatterplotLayer",data=df,
    get_position='[lon,lat]',get_radius=30,get_color=[255,0,0,180]))
layers.append(pdk.Layer("TextLayer",data=df,
    get_position='[lon,lat]',get_text=label_col,get_size=16,get_color=[0,0,0,200],
    get_alignment_baseline="'bottom'"))

rp=st.session_state.road_path
if rp and len(rp)>1:
    # 先進的アニメーション: TripsLayer を使って動く経路表示
    timestamps=list(range(len(rp)))
    df_trips=pd.DataFrame({"path":[rp],"timestamps":[timestamps]})
    t_max=max(timestamps)
    current_time=st.slider("経路アニメーション",0,t_max,0,1,key="anim_time")
    layers.append(pdk.Layer("TripsLayer",data=df_trips,
        get_path="path",get_timestamps="timestamps",
        get_color=[255,60,60],opacity=0.8,width_min_pixels=4,
        trail_length=20,current_time=current_time))

view=pdk.ViewState(latitude=clat,longitude=clon,zoom=zoom,pitch=45)
st.pydeck_chart(pdk.Deck(map_style=map_styles[map_style],
                         initial_view_state=view,layers=layers,
                         tooltip={"text":f"{{{label_col}}}"}),
                use_container_width=True)

with st.expander("📋 避難所一覧・巡回順"):
    st.dataframe(df)
    if st.session_state.route:
        order=[df.iloc[i][label_col] for i in st.session_state.route]
        st.write("巡回順:", order)

import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import osmnx as ox
import networkx as nx
import packaging.version
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ── 設定 ─────────────────────────────────────────────────────────
GEOJSON_LOCAL_PATH = "hinanjyo.geojson"
GEOJSON_RAW_URL   = "https://raw.githubusercontent.com/<ユーザー名>/<リポジトリ名>/<ブランチ名>/hinanjyo.geojson"
KAMIJIMA_CENTER   = (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
st.markdown("""
    <style>
    @media (max-width: 800px) {
        .block-container { padding-left:0.4rem; padding-right:0.4rem; }
        .stButton button { font-size:1.12em; padding:0.7em 1.3em; }
    }
    </style>
""", unsafe_allow_html=True)
st.title("🏫 避難所最短ルート探すくん")

# ── ユーティリティ関数 ─────────────────────────────────────────
def guess_name_col(df):
    for cand in ["name","NAME","名称","避難所","施設名","address","住所"]:
        if cand in df.columns:
            return cand
    return df.columns[0] if not df.empty else "name"

def file_to_df(uploaded_files):
    try:
        if any(f.name.endswith(".shp") for f in uploaded_files):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for f in uploaded_files:
                    open(os.path.join(td,f.name),"wb").write(f.getvalue())
                shp = [os.path.join(td,f.name) for f in uploaded_files if f.name.endswith(".shp")][0]
                gdf = gpd.read_file(shp)
        elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
            f = [f for f in uploaded_files if f.name.endswith((".geojson",".json"))][0]
            gdf = gpd.read_file(f)
        elif any(f.name.endswith(".csv") for f in uploaded_files):
            f = [f for f in uploaded_files if f.name.endswith(".csv")][0]
            df = pd.read_csv(f)
            if not {"lat","lon"}<=set(df.columns):
                st.warning("CSVには必ずlat, lon列が必要です")
                return pd.DataFrame(columns=["lat","lon","name"])
            df["lat"]=pd.to_numeric(df["lat"],errors="coerce")
            df["lon"]=pd.to_numeric(df["lon"],errors="coerce")
            return df.dropna(subset=["lat","lon"])
        else:
            st.warning("対応形式：SHP一式/GeoJSON/CSVのみ")
            return pd.DataFrame(columns=["lat","lon","name"])
        # GeoDataFrame の EPSG を4326に統一
        if gdf.crs is None:
            gdf.set_crs(epsg=4326,inplace=True)
        elif gdf.crs.to_epsg()!=4326:
            gdf = gdf.to_crs(epsg=4326)
        # Point 型のみ
        gdf = gdf[gdf.geometry.type=="Point"]
        gdf["lon"]=gdf.geometry.x; gdf["lat"]=gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"]=gdf.index.astype(str)
        return gdf[["lat","lon","name"]].dropna()
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame(columns=["lat","lon","name"])

def load_initial_geojson():
    try:
        path = GEOJSON_LOCAL_PATH if os.path.exists(GEOJSON_LOCAL_PATH) else GEOJSON_RAW_URL
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326,inplace=True)
        elif gdf.crs.to_epsg()!=4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.type=="Point"]
        gdf["lon"]=gdf.geometry.x; gdf["lat"]=gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"]=gdf.index.astype(str)
        return gdf[["lat","lon","name"]].dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"初期GeoJSON読み込みエラー: {e}")
        return pd.DataFrame(columns=["lat","lon","name"])

def create_road_distance_matrix(locs, mode="drive"):
    import numpy as np
    version = packaging.version.parse(ox.__version__)
    lats=[float(p[0]) for p in locs]; lons=[float(p[1]) for p in locs]
    for pad in [0.01,0.03,0.07]:
        try:
            if version<packaging.version.parse("2.0.0"):
                G=ox.graph_from_bbox(max(lats)+pad, min(lats)-pad,
                                     max(lons)+pad, min(lons)-pad,
                                     network_type=mode)
            else:
                bbox=(max(lats)+pad, min(lats)-pad, max(lons)+pad, min(lons)-pad)
                G=ox.graph_from_bbox(bbox=bbox, network_type=mode)
            if not G.nodes:
                continue
            node_ids=[]
            for lat,lon in locs:
                try:
                    node_ids.append(ox.nearest_nodes(G,lon,lat))
                except:
                    node_ids.append(None)
            n=len(locs); mat=np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i==j: continue
                    ni, nj = node_ids[i], node_ids[j]
                    if ni is not None and nj is not None:
                        try:
                            mat[i,j]=nx.shortest_path_length(G,ni,nj,weight="length")/1000
                        except:
                            mat[i,j]=float("inf")
                    else:
                        mat[i,j]=float("inf")
            return mat, G, node_ids
        except:
            continue
    st.warning("道路データ取得失敗→直線距離TSPに切替")
    n=len(locs); mat=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                mat[i,j]=np.linalg.norm(np.array(locs[i])-np.array(locs[j]))
    return mat,None,[]

def solve_tsp(distance_matrix):
    mgr=pywrapcp.RoutingIndexManager(len(distance_matrix),1,0)
    routing=pywrapcp.RoutingModel(mgr)
    def cb(fi,ti):
        return int(distance_matrix[mgr.IndexToNode(fi),mgr.IndexToNode(ti)]*100000)
    idx_cb=routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx_cb)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds=1
    params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol=routing.SolveWithParameters(params)
    route=[]
    if sol:
        idx=routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(mgr.IndexToNode(idx))
            idx=sol.Value(routing.NextVar(idx))
        route.append(route[0])
    return route

# ── セッションステート初期化 ─────────────────────────────────────────
defaults={
    "shelters": load_initial_geojson(),
    "label_col":"name",
    "map_style":"light",
    "ox_mode":"drive",
    "road_path":[],
    "route":[]
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

# ── サイドバー：データ追加 & TSPフォーム ─────────────────────────
st.sidebar.header("避難所データ追加 (SHP/GeoJSON/CSV)")
st.sidebar.info("スマホ→ファイルアプリ→選択でOK\nSHP一式 or GeoJSON/CSV")
up=st.sidebar.file_uploader("アップロード",type=["shp","shx","dbf","prj","geojson","json","csv"],accept_multiple_files=True)
if up:
    gdf=file_to_df(up)
    if not gdf.empty:
        st.session_state["shelters"]=pd.concat([st.session_state["shelters"],gdf],ignore_index=True)
        st.success(f"{len(gdf)}件追加")
        st.session_state["label_col"]=guess_name_col(st.session_state["shelters"])
with st.sidebar.form("manual_add"):
    lat=st.number_input("緯度",value=KAMIJIMA_CENTER[0],format="%f")
    lon=st.number_input("経度",value=KAMIJIMA_CENTER[1],format="%f")
    nm=st.text_input("避難所名","新しい避難所")
    if st.form_submit_button("手動追加"):
        st.session_state["shelters"]=pd.concat([
            st.session_state["shelters"],
            pd.DataFrame([{"lat":lat,"lon":lon,"name":nm}])
        ],ignore_index=True)
if st.sidebar.button("リセット"):
    st.session_state["shelters"]=load_initial_geojson()
    st.session_state["road_path"]=[]
    st.session_state["route"]=[]
    st.session_state["label_col"]="name"
download=st.session_state["shelters"].to_csv(index=False)
st.sidebar.download_button("CSVダウンロード",download,"shelters.csv","text/csv")

with st.sidebar.form("tsp_form"):
    st.markdown("---")
    st.header("TSPルート計算")
    mode_disp=st.selectbox("道路種別",["車（drive）","徒歩（walk）"])
    st.session_state["ox_mode"]="drive" if "車" in mode_disp else "walk"
    calc_btn=st.form_submit_button("計算スタート")

# ── メイン画面：施設選択・計算・地図描画 ─────────────────────────────────
df=st.session_state["shelters"].copy()
df["lat"]=pd.to_numeric(df["lat"],errors="coerce")
df["lon"]=pd.to_numeric(df["lon"],errors="coerce")

# ★ すべてのカラムをラベル選択肢に
label_opts=list(df.columns)
st.session_state["label_col"]=st.selectbox(
    "ラベル列", label_opts,
    index=label_opts.index(st.session_state["label_col"])
)

# 地図スタイル辞書
map_style_dict={
    "light":"light","dark":"dark",
    "ストリート":"mapbox://styles/mapbox/streets-v12",
    "衛星写真":"mapbox://styles/mapbox/satellite-streets-v12",
    "アウトドア":"mapbox://styles/mapbox/outdoors-v12",
    "ナビ風":"mapbox://styles/mapbox/navigation-night-v1"
}
st.session_state["map_style"]=st.selectbox(
    "地図背景スタイル", list(map_style_dict.keys()),
    index=list(map_style_dict.keys()).index(st.session_state["map_style"])
)

# 施設リスト表示と選択
display=[f"{r[st.session_state['label_col']]}({r['lat']:.5f},{r['lon']:.5f})" for _,r in df.iterrows()]
idx_map={i:n for i,n in enumerate(display)}
st.markdown("<span style='font-size:14px;'>📋 巡回施設の選択</span>",unsafe_allow_html=True)
sel=st.multiselect("選択",options=list(idx_map),format_func=lambda x:idx_map[x],key="sel")

# 計算実行
if calc_btn:
    if len(sel)<2:
        st.warning("2か所以上選択してください")
    else:
        sub=df.iloc[sel].reset_index(drop=True)
        locs=list(zip(sub["lat"],sub["lon"]))
        with st.spinner("計算中…"):
            distmat,G,node_ids=create_road_distance_matrix(locs,mode=st.session_state["ox_mode"])
            route=solve_tsp(distmat)
            st.session_state["route"]=[sel[i] for i in route if i<len(sel)]
            # 経路描画座標
            path=[]
            if G and node_ids:
                for i in range(len(route)-1):
                    seg=nx.shortest_path(G,node_ids[route[i]],node_ids[route[i+1]],weight="length")
                    coords=[[G.nodes[n]["x"],G.nodes[n]["y"]] for n in seg]
                    path+=coords if i==0 else coords[1:]
            else:
                for i in range(len(route)-1):
                    path.append([sub.loc[route[i],"lon"],sub.loc[route[i],"lat"]])
                    path.append([sub.loc[route[i+1],"lon"],sub.loc[route[i+1],"lat"]])
            st.session_state["road_path"]=path
            total=sum(distmat[route[i],route[i+1]] for i in range(len(route)-1))
            st.success(f"計算完了！総距離: {total:.2f} km")

# 地図描画
st.markdown("<span style='font-size:14px;'>🗺️ 地図（ラベル＋TSPルート）</span>",unsafe_allow_html=True)
layer_pts=pdk.Layer("ScatterplotLayer",data=df,get_position='[lon,lat]',get_color='[0,150,255,200]',get_radius=40,radius_min_pixels=1,radius_max_pixels=6,pickable=True)
layer_text=pdk.Layer("TextLayer",data=df,get_position='[lon,lat]',get_text=st.session_state["label_col"],get_size=15,get_color=[20,20,40,180],get_alignment_baseline="'bottom'",pickable=False)
layers=[layer_pts,layer_text]
rp=st.session_state["road_path"]
if rp and len(rp)>1:
    layers.append(pdk.Layer("PathLayer",data=pd.DataFrame({"path":[rp]}),get_path="path",get_color=[255,60,60,200],width_scale=10,width_min_pixels=4,pickable=False))
view=pdk.ViewState(latitude=KAMIJIMA_CENTER[0],longitude=KAMIJIMA_CENTER[1],zoom=13.3,pitch=45,bearing=0)
st.pydeck_chart(pdk.Deck(map_style=map_style_dict[st.session_state["map_style"]],layers=layers,initial_view_state=view,tooltip={"text":f"{{{st.session_state['label_col']}}}"}),use_container_width=True)

# データ一覧 expander
if not df.empty:
    with st.expander("📋 避難所一覧・巡回順"):
        st.dataframe(df)
        if st.session_state["route"]:
            st.write("巡回順:",[df.iloc[i][st.session_state["label_col"]] for i in st.session_state["route"]])
else:
    st.info("避難所データがありません")

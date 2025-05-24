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

# ───────────────────────────────────────
# 設定
# ───────────────────────────────────────
GEOJSON_LOCAL_PATH = "hinanjyo.geojson"
GEOJSON_RAW_URL   = "https://raw.githubusercontent.com/<ユーザー名>/<リポジトリ名>/main/hinanjyo.geojson"
KAMIJIMA_CENTER   = (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所最短ルート探すくん", layout="wide")
st.markdown("""
    <style>
    @media (max-width: 800px) {
        .block-container { padding-left:0.5rem; padding-right:0.5rem; }
        .stButton button { font-size:1.1em; padding:0.6em 1.2em; }
    }
    </style>
""", unsafe_allow_html=True)
st.title("🏫 避難所最短ルート探すくん")

# ───────────────────────────────────────
# GeoJSON/SHP/CSV 読み込みユーティリティ
# ───────────────────────────────────────
def load_initial_geojson():
    try:
        src = GEOJSON_LOCAL_PATH if os.path.exists(GEOJSON_LOCAL_PATH) else GEOJSON_RAW_URL
        gdf = gpd.read_file(src)
        if gdf.crs is None or gdf.crs.to_epsg() != 2446:
            gdf = gdf.set_crs(epsg=2446, allow_override=True)
        gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"初期GeoJSON読み込みエラー: {e}")
        return pd.DataFrame()

def file_to_df(uploaded_files):
    try:
        if any(f.name.endswith(".shp") for f in uploaded_files):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for f in uploaded_files:
                    open(os.path.join(td,f.name),"wb").write(f.getvalue())
                shp = next(p for p in os.listdir(td) if p.endswith(".shp"))
                gdf = gpd.read_file(os.path.join(td,shp))
        elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
            f = next(f for f in uploaded_files if f.name.endswith((".geojson",".json")))
            gdf = gpd.read_file(f)
        else:
            f = next(f for f in uploaded_files if f.name.endswith(".csv"))
            df_csv = pd.read_csv(f)
            if not {"lat","lon"}.issubset(df_csv.columns):
                st.warning("CSVには必ず lat, lon 列が必要です")
                return pd.DataFrame()
            df_csv["lat"] = pd.to_numeric(df_csv["lat"],errors="coerce")
            df_csv["lon"] = pd.to_numeric(df_csv["lon"],errors="coerce")
            if "name" not in df_csv.columns:
                df_csv["name"] = df_csv.index.astype(str)
            return df_csv.dropna(subset=["lat","lon"])
        if gdf.crs is None or gdf.crs.to_epsg()!=2446:
            gdf = gdf.set_crs(epsg=2446, allow_override=True)
        gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.type=="Point"].copy()
        gdf["lon"]=gdf.geometry.x; gdf["lat"]=gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry"))
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return pd.DataFrame()

# ───────────────────────────────────────
# 道路ネットワーク距離行列＋ノード取得（プログレス付）
# ───────────────────────────────────────
def create_road_distance_matrix(locs, mode="drive", log_box=None, prog_bar=None):
    import numpy as np
    ver = packaging.version.parse(ox.__version__)
    lats = [float(p[0]) for p in locs]; lons = [float(p[1]) for p in locs]
    # ネットワーク取得
    for pad in [0.01,0.03,0.07]:
        try:
            if ver < packaging.version.parse("2.0.0"):
                G = ox.graph_from_bbox(max(lats)+pad, min(lats)-pad,
                                       max(lons)+pad, min(lons)-pad,
                                       network_type=mode)
            else:
                bbox=(max(lats)+pad, min(lats)-pad, max(lons)+pad, min(lons)-pad)
                G = ox.graph_from_bbox(bbox=bbox, network_type=mode)
            if G.nodes: break
        except Exception:
            continue
    else:
        log_box.warning("道路ネット取得失敗→直線距離でTSP")
        G=None

    # nearest_nodes
    n = len(locs)
    node_ids = []
    log_box.text("Nearest nodes を計算中…")
    for i,(lat,lon) in enumerate(locs):
        if G:
            try: node_ids.append(ox.nearest_nodes(G, lon, lat))
            except: node_ids.append(None)
        else:
            node_ids.append(None)
        prog_bar.progress(int((i+1)/n*20))  # 0-20%
    # 距離行列構築
    mat = np.zeros((n,n))
    log_box.text("距離行列を計算中…")
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if G and node_ids[i] and node_ids[j]:
                try:
                    d = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight="length")/1000
                except:
                    d = np.inf
            else:
         

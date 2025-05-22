import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import tempfile
import os
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# デザイン推奨：地図中心
KAMIJIMA_CENTER = (34.25754417840102, 133.20446981161595)

st.set_page_config(page_title="避難所TSPラベル地図", layout="wide")
st.title("🏫 避難所TSPルートアプリ（地図デザイン＆ラベル選択＆切替）")

def guess_name_col(df):
    # 一般的にラベルにしたい順で選ぶ
    for cand in ["name", "NAME", "名称", "避難所", "施設名", "address", "住所"]:
        if cand in df.columns:
            return cand
    # 次にobject型（str系）から選ぶ
    obj_cols = [c for c in df.columns if df[c].dtype == 'O']
    if obj_cols:
        return obj_cols[0]
    return df.columns[0]

def file_to_df(uploaded_files):
    if any(f.name.endswith(".shp") for f in uploaded_files):
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as out:
                    out.write(file.getvalue())
            shp_path = [os.path.join(temp_dir, f.name) for f in uploaded_files if f.name.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)
    elif any(f.name.endswith((".geojson",".json")) for f in uploaded_files):
        geojson_file = [f for f in uploaded_files if f.name.endswith((".geojson",".json"))][0]
        gdf = gpd.read_file(geojson_file)
    elif any(f.name.endswith(".csv") for f in uploaded_files):
        csv_file = [f for f in uploaded_files if f.name.endswith(".csv")][0]
        df = pd.read_csv(csv_file)
        if not set(["lat","lon"]).issubset(df.columns):
            st.warning("lat, lon 列が必要です")
            return pd.DataFrame(columns=["lat", "lon", "name"])
        for c in ["lat", "lon"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    else:
        st.warning("SHP/GeoJSON/CSVのみ対応です")
        return pd.DataFrame(columns=["lat", "lon", "name"])

    # EPSG自動変換
    crs_was_set = False
    if gdf.crs is None:
        st.warning("座標系情報がありません。EPSG:4326として扱います。")
        gdf.set_crs(epsg=4326, inplace=True)
        crs_was_set = True
    elif gdf.crs.to_epsg() != 4326:
        st.info(f"座標系が {gdf.crs} → EPSG:4326 に自動変換します")
        gdf = gdf.to_crs(epsg=4326)
        crs_was_set = True

    # (確認用)CRSを表示
    if crs_was_set:
        st.write(f"現在のCRS: {gdf.crs}")

    if gdf.geometry.iloc[0].geom_type != "Point":
        st.warning("Point型ジオメトリのみ対応です")
        return pd.DataFrame(columns=["lat", "lon", "name"])
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf

def create_distance_matrix(locations):
    n = len(locations)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
    return mat

def solve_tsp(distance_matrix):
    size = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node]*10000)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    route = []
    if solution:
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(route[0])
    return route

# セッション管理
if "shelters" not in st.session_state:
    st.session_state["shelters"] = pd.DataFrame([
        {"lat": KAMIJIMA_CENTER[0], "lon": KAMIJIMA_CENTER[1], "name": "上島町役場"}
    ])
if "selected" not in st.session_state:
    st.session_state["selected"] = []
if "route" not in st.session_state:
    st.session_state["route"] = []
if "label_col" not in st.session_state:
    st.session_state["label_col"] = "name"
if "map_style" not in st.session_state:
    st.session_state["map_style"] = "light"

# ファイルアップロード
st.sidebar.header("避難所データ追加 (SHP/GeoJSON/CSV)")
uploaded_files = st.sidebar.file_uploader(
    "全ファイル一括選択可（SHP一式, GeoJSON, CSV混在OK）",
    type=["shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "csv"],
    accept_multiple_files=True
)
if uploaded_files:
    gdf = file_to_df(uploaded_files)
    if not gdf.empty:
        gdf = gdf[[c for c in gdf.columns if c in ["lat", "lon"] or gdf[c].dtype == 'O']].copy()
        # ラベル候補列を抽出
        st.session_state["shelters"] = pd.concat([st.session_state["shelters"], gdf], ignore_index=True)
        st.success(f"{len(gdf)}件の避難所を追加しました")
        # 追加直後は推奨カラムで上書き
        st.session_state["label_col"] = guess_name_col(st.session_state["shelters"])

# 手動追加
with st.sidebar.form(key="manual_add"):
    st.write("避難所を手動で追加")
    lat = st.number_input("緯度", value=KAMIJIMA_CENTER[0], format="%f")
    lon = st.number_input("経度", value=KAMIJIMA_CENTER[1], format="%f")
    name = st.text_input("避難所名", "新しい避難所")
    add_btn = st.form_submit_button("追加")
    if add_btn:
        st.session_state["shelters"] = pd.concat([
            st.session_state["shelters"],
            pd.DataFrame([{"lat": lat, "lon": lon, "name": str(name)}])
        ], ignore_index=True)

if st.sidebar.button("すべて削除"):
    st.session_state["shelters"] = pd.DataFrame([
        {"lat": KAMIJIMA_CENTER[0], "lon": KAMIJIMA_CENTER[1], "name": "上島町役場"}
    ])
    st.session_state["selected"] = []
    st.session_state["route"] = []
    st.session_state["label_col"] = "name"

csv_export = st.session_state["shelters"].to_csv(index=False)
st.sidebar.download_button("避難所CSVをダウンロード", csv_export, file_name="shelters.csv", mime="text/csv")

# --------------------
st.header("📋 避難所リストから計算対象とラベル・地図を選択")
shelters_df = st.session_state["shelters"].copy()
shelters_df["lat"] = pd.to_numeric(shelters_df["lat"], errors="coerce")
shelters_df["lon"] = pd.to_numeric(shelters_df["lon"], errors="coerce")
# ラベル候補（object型カラムのみ選択肢に）
label_candidates = [c for c in shelters_df.columns if shelters_df[c].dtype == "O"]
if len(label_candidates) == 0:
    label_candidates = ["name"]
st.session_state["label_col"] = st.selectbox(
    "地図ラベルに使う列を選んでください（おすすめ：名称）",
    label_candidates,
    index=label_candidates.index(st.session_state["label_col"]) if st.session_state["label_col"] in label_candidates else 0
)

# 地図スタイル選択
map_style_dict = {
    "light": "light",
    "dark": "dark",
    "ストリート": "mapbox://styles/mapbox/streets-v12",
    "衛星写真": "mapbox://styles/mapbox/satellite-streets-v12",
    "アウトドア": "mapbox://styles/mapbox/outdoors-v12",
    "ナビ風": "mapbox://styles/mapbox/navigation-night-v1"
}
style_name = st.selectbox(
    "地図背景スタイル",
    list(map_style_dict.keys()),
    index=list(map_style_dict.keys()).index(st.session_state.get("map_style", "light"))
)
st.session_state["map_style"] = style_name

# カラム欠損除去
shelters_df = shelters_df.dropna(subset=["lat", "lon"])

if not shelters_df.empty:
    select_labels = [f"{row[st.session_state['label_col']]} ({row['lat']:.5f},{row['lon']:.5f})" for _, row in shelters_df.iterrows()]
    selected_labels = st.multiselect(
        "巡回したい避難所（順序自動最適化）",
        options=select_labels,
        default=[select_labels[i] for i in st.session_state["selected"]] if st.session_state["selected"] else select_labels
    )
    selected_idx = [select_labels.index(lab) for lab in selected_labels]
    st.session_state["selected"] = selected_idx
else:
    st.info("避難所データをまずアップロード・追加してください。")

st.header("🚩 最短巡回ルート計算・地図表示")
if st.button("選択避難所でTSP最短巡回ルート計算"):
    selected = st.session_state["selected"]
    if not selected or len(selected) < 2:
        st.warning("最低2か所以上の避難所を選択してください。")
    else:
        df = shelters_df.iloc[selected].reset_index(drop=True)
        locs = list(zip(df["lat"], df["lon"]))
        distmat = create_distance_matrix(locs)
        route = solve_tsp(distmat)
        st.session_state["route"] = [selected[i] for i in route]
        total = sum([distmat[route[i], route[i+1]] for i in range(len(route)-1)])
        st.success(f"巡回ルート計算完了！総距離: {total:.2f} km（直線距離）")

df = shelters_df
route = st.session_state["route"]

# -------------------- pydeck: ラベルも表示 -------------------
layer_pts = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_color='[0, 150, 255, 200]',
    get_radius=150,
    radius_min_pixels=2,
    pickable=True,
)

layer_text = pdk.Layer(
    "TextLayer",
    data=df,
    get_position='[lon, lat]',
    get_text=st.session_state["label_col"],
    get_size=18,
    get_color=[20, 20, 40, 230],
    get_angle=0,
    get_alignment_baseline="'bottom'",
    pickable=False,
)

layers = [layer_pts, layer_text]

if route and len(route) > 1 and all(i < len(df) for i in route):
    coords = [[df.iloc[i]["lon"], df.iloc[i]["lat"]] for i in route]
    layer_line = pdk.Layer(
        "LineLayer",
        data=pd.DataFrame({"start": coords[:-1], "end": coords[1:]}),
        get_source_position="start",
        get_target_position="end",
        get_width=6,
        get_color=[255, 50, 50, 180],
    )
    layers.append(layer_line)

view = pdk.ViewState(
    latitude=KAMIJIMA_CENTER[0],
    longitude=KAMIJIMA_CENTER[1],
    zoom=13.3,
    pitch=45,
    bearing=0,
)

# MapboxのAPI Keyが必要な場合は下記のように環境変数でセットして下さい
# st.secrets["MAPBOX_KEY"] = "xxxx"
st.pydeck_chart(pdk.Deck(
    map_style=map_style_dict[st.session_state["map_style"]],
    layers=layers,
    initial_view_state=view,
    tooltip={"text": f"{{{st.session_state['label_col']}}}"}
))

with st.expander("避難所リスト/巡回順"):
    st.dataframe(df)
    if route and all(i < len(df) for i in route):
        st.write("巡回順（0起点）:", [df.iloc[i][st.session_state["label_col"]] for i in route])

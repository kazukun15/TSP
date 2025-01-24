import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import geopandas as gpd
import tempfile
import os
import numpy as np

def create_map(center=[34.2832, 133.1831], zoom=13):
    """地図を初期化"""
    return folium.Map(location=center, zoom_start=zoom)

def add_shelter(map_obj, shelters):
    """避難所をマップに追加"""
    for shelter in shelters:
        folium.Marker(
            location=[shelter["lat"], shelter["lon"]],
            popup=shelter["name"]
        ).add_to(map_obj)

def load_shelters(files):
    """SHP関連ファイルから避難所データを読み込む"""
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
        shp_path = next(os.path.join(temp_dir, file.name) for file in files if file.name.endswith('.shp'))
        gdf = gpd.read_file(shp_path)
        return [
            {
                "lat": row.geometry.y,
                "lon": row.geometry.x,
                "name": row.get("name", "Unnamed Shelter")
            }
            for _, row in gdf.iterrows()
        ]

# ─────────────────────────────────────────────────────────────
# ★ 修正箇所：OSMnxバージョンに合わせて「位置引数のみ」を使う
# ─────────────────────────────────────────────────────────────
def get_network_from_shelters(shelters, mode="walking"):
    """避難所の範囲を考慮してネットワークを取得（OSMnx 0.x 向け位置引数形式）"""
    mode_type = {"walking": "walk", "bicycle": "bike", "car": "drive"}[mode]

    # 緯度と経度の範囲を計算
    lats = [s["lat"] for s in shelters]
    lons = [s["lon"] for s in shelters]
    north, south, east, west = max(lats), min(lats), max(lons), min(lons)

    # 位置引数のみを使用 (古いバージョンでエラーにならない形)
    G = ox.graph_from_bbox(north, south, east, west, network_type=mode_type)
    return G

def calculate_route_osm(shelters, mode="walking"):
    """OSMネットワークを利用して最短経路を計算"""
    G = get_network_from_shelters(shelters, mode)
    nodes = [ox.nearest_nodes(G, shelter["lon"], shelter["lat"]) for shelter in shelters]

    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                try:
                    distance = nx.shortest_path_length(
                        G, source=nodes[i], target=nodes[j], weight="length"
                    )
                    distance_matrix[i][j] = distance
                except nx.NetworkXNoPath:
                    distance_matrix[i][j] = float("inf")

    path, total_distance = solve_tsp_with_network(distance_matrix)
    return path, total_distance, G, nodes

def solve_tsp_with_network(distance_matrix):
    from scipy.optimize import linear_sum_assignment
    num_nodes = distance_matrix.shape[0]
    cost_matrix = distance_matrix.copy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    path = list(row_ind) + [row_ind[0]]  # 巡回経路を閉じる
    total_distance = sum(cost_matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind)))
    return path, total_distance

def initialize_session_state():
    if "shelters" not in st.session_state:
        st.session_state.shelters = []
    if "route" not in st.session_state:
        st.session_state.route = []
    if "route_calculated" not in st.session_state:
        st.session_state.route_calculated = False

def main():
    st.set_page_config(page_title="避難所セールスマン問題解決アプリ", layout="wide")
    initialize_session_state()

    st.title("避難所セールスマン問題解決アプリ（修正版）")

    st.sidebar.header("操作メニュー")
    action = st.sidebar.radio(
        "操作を選択してください：",
        options=["地図を表示", "データをアップロード", "手動で避難所を追加"],
    )

    # 開始ボタン
    if st.sidebar.button("開始ボタン"):
        if len(st.session_state.shelters) < 2:
            st.sidebar.error("避難所が2つ以上必要です。")
        else:
            mode = st.sidebar.radio(
                "移動手段を選択してください",
                options=["walking", "bicycle", "car"],
                format_func=lambda x: {"walking": "徒歩", "bicycle": "自転車", "car": "自動車"}[x]
            )
            try:
                path, total_distance, G, nodes = calculate_route_osm(st.session_state.shelters, mode)
                # (lat, lon)形式に変換
                st.session_state.route = [
                    (G.nodes[nodes[i]]["y"], G.nodes[nodes[i]]["x"]) for i in path
                ]
                st.session_state.route_calculated = True
                st.sidebar.success(f"ルート計算が完了しました！総距離: {total_distance / 1000:.2f} km")
            except Exception as e:
                st.sidebar.error(f"エラーが発生しました: {e}")

    # 地図を表示
    if action == "地図を表示":
        st.header("地図を表示")
        map_obj = create_map()
        add_shelter(map_obj, st.session_state.shelters)
        if st.session_state.route_calculated:
            for i in range(len(st.session_state.route) - 1):
                folium.PolyLine(
                    locations=[
                        (st.session_state.route[i][0], st.session_state.route[i][1]),
                        (st.session_state.route[i + 1][0], st.session_state.route[i + 1][1]),
                    ],
                    color="red",
                    weight=2.5,
                    opacity=0.8,
                ).add_to(map_obj)
        st_folium(map_obj, width=800, height=600)

    elif action == "データをアップロード":
        st.header("避難所データのアップロード")
        uploaded_files = st.file_uploader(
            "SHPファイルをアップロード", 
            type=["shp", "shx", "dbf", "prj"],
            accept_multiple_files=True
        )
        if st.button("データを読み込む"):
            if uploaded_files:
                shelters = load_shelters(uploaded_files)
                st.session_state.shelters.extend(shelters)
                st.success(f"{len(shelters)} 件の避難所が追加されました！")
            else:
                st.error("ファイルをアップロードしてください。")

    elif action == "手動で避難所を追加":
        st.header("避難所の手動追加")
        lat = st.text_input("緯度を入力", value="34.2832")
        lon = st.text_input("経度を入力", value="133.1831")
        name = st.text_input("避難所名", value="新しい避難所")
        if st.button("避難所を追加"):
            try:
                lat = float(lat)
                lon = float(lon)
                st.session_state.shelters.append({"lat": lat, "lon": lon, "name": name})
                st.success(f"避難所 '{name}' を追加しました！")
            except ValueError:
                st.error("緯度または経度の入力が正しくありません。数値を入力してください。")

if __name__ == "__main__":
    main()

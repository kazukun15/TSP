import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import tempfile
import os
import osmnx as ox
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

KAMIJIMA_CENTER = (34.25754417840102, 133.20446981161595)
st.set_page_config(page_title="避難所TSP（スマホ・PC両対応）", layout="wide")
st.markdown("""
    <style>
    /* スマホ時はマージン減らしてフォームを縦積み */
    @media (max-width: 800px) {
        .block-container { padding-left: 0.4rem; padding-right: 0.4rem; }
        label, .stTextInput > label { font-size: 1.1em; }
        .stButton button { font-size: 1.15em; padding: 0.75em 1.4em; }
    }
    </style>
""", unsafe_allow_html=True)
st.title("🏫 避難所TSPルートアプリ（スマホ・PC両対応）")

def guess_name_col(df):
    for cand in ["name", "NAME", "名称", "避難所", "施設名", "address", "住所"]:
        if cand in df.columns:
            return cand
    obj_cols = [c for c in df.columns if df[c].dtype == 'O']
    if obj_cols:
        return obj_cols[0]
    return df.columns[0]

def file_to_df(uploaded_files):
    try:
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
            csv_file

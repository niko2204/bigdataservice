from pathlib import Path
import sys
import folium
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.data_service import load_features

st.title("🗺️ 지도 분석")
df = load_features()
metric = st.selectbox("원 크기 지표", ["유동인구", "총인구", "카페수", "음식점수"])
m = folium.Map(location=[34.804, 126.414], zoom_start=13, tiles="CartoDB positron")
max_value = max(df[metric].max(), 1)
for _, row in df.iterrows():
    folium.CircleMarker(
        [row["위도"], row["경도"]], radius=5 + 18 * row[metric] / max_value,
        tooltip=f"{row['행정동명']} | {metric}: {row[metric]:,.0f}",
        color="#2563eb", fill=True, fill_opacity=.55,
    ).add_to(m)
st_folium(m, use_container_width=True, height=610)


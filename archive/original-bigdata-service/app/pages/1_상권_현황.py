from pathlib import Path
import sys
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.data_service import load_features

st.title("📊 상권 현황")
df = load_features()
metric = st.selectbox("비교 지표", ["총인구", "20대인구", "30대인구", "유동인구", "카페수", "음식점수", "주차장수"])
ordered = df.sort_values(metric, ascending=False)
st.plotly_chart(px.bar(ordered, x="행정동명", y=metric, color=metric, title=f"행정동별 {metric}"), use_container_width=True)
st.dataframe(ordered, use_container_width=True, hide_index=True)


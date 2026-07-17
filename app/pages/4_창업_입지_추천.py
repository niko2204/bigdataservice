from pathlib import Path
import sys
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.data_service import load_features
from src.models.location_score import calculate_scores

st.title("🎯 창업 입지 추천")
business = st.selectbox("창업 업종", ["카페", "음식점", "편의점"])
target_age = st.radio("주요 고객", ["20대", "30대"], horizontal=True)
st.subheader("중요도 설정")
cols = st.columns(5)
labels = ["잠재고객", "유동인구", "경쟁완화", "접근성", "비용효율"]
weights = {label: cols[i].slider(label, 0, 10, 5) for i, label in enumerate(labels)}
result = calculate_scores(load_features(), business, target_age, weights)
top = result.head(5)
st.plotly_chart(px.bar(top, x="행정동명", y="적합도", color="적합도", range_y=[0, 100], title="추천 상위 지역"), use_container_width=True)
st.dataframe(top[["순위", "행정동명", "적합도", "추천이유"]].style.format({"적합도": "{:.1f}"}), use_container_width=True, hide_index=True)


from pathlib import Path
import sys
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.data_service import load_features

st.title("🏪 업종 분석")
df = load_features()
business = st.selectbox("업종", ["카페", "음식점", "편의점"])
col = {"카페": "카페수", "음식점": "음식점수", "편의점": "편의점수"}[business]
df["점포당인구"] = (df["총인구"] / (df[col] + 1)).round(1)
left, right = st.columns(2)
left.plotly_chart(px.scatter(df, x=col, y="유동인구", size="총인구", hover_name="행정동명", title="경쟁 점포와 유동인구"), use_container_width=True)
right.plotly_chart(px.bar(df.sort_values("점포당인구", ascending=False), x="행정동명", y="점포당인구", title=f"{business} 점포당 인구"), use_container_width=True)


from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.data_service import load_features

st.set_page_config(page_title="목포 창업 입지 추천", page_icon="📍", layout="wide")
data = load_features()

st.title("📍 목포 상권 빅데이터 서비스")
st.caption("지역 생활인구·상권 데이터 기반 창업 입지 추천 교육용 프로젝트")

c1, c2, c3, c4 = st.columns(4)
c1.metric("분석 행정동", f"{len(data)}개")
c2.metric("총인구", f"{data['총인구'].sum():,}명")
c3.metric("등록 카페", f"{data['카페수'].sum():,}개")
c4.metric("총 유동인구 지수", f"{data['유동인구'].sum():,}")

st.info("왼쪽 페이지 메뉴에서 상권 현황, 지도, 입지 추천, AI 보고서를 확인하세요.")
fig = px.bar(data.sort_values("유동인구", ascending=False), x="행정동명", y="유동인구",
             color="카페수", title="행정동별 유동인구와 카페 수")
st.plotly_chart(fig, use_container_width=True)

st.warning("이 서비스의 샘플 수치는 수업용 합성 데이터이며 실제 창업 판단에 사용할 수 없습니다.")


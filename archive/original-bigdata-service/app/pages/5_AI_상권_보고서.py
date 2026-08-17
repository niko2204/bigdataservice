from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.data_service import load_features
from src.models.location_score import calculate_scores

st.title("🤖 AI 상권 분석 보고서")
business = st.selectbox("업종", ["카페", "음식점", "편의점"])
result = calculate_scores(load_features(), business)
place = st.selectbox("분석 지역", result["행정동명"].tolist())
row = result[result["행정동명"] == place].iloc[0]
report = f"""# {place} {business} 상권 분석 보고서

## 요약
{place}의 {business} 입지 적합도는 **{row['적합도']:.1f}점**, 비교 지역 중 **{int(row['순위'])}위**입니다.

## 핵심 지표
- 총인구: {int(row['총인구']):,}명
- 20대 인구: {int(row['20대인구']):,}명
- 유동인구 지수: {int(row['유동인구']):,}
- 기존 카페: {int(row['카페수'])}개
- 주차장: {int(row['주차장수'])}개

## 추천 근거
{row['추천이유']}합니다. 실제 의사결정 전에는 임대료, 시간대별 매출, 보행량, 폐업률을 추가 조사해야 합니다.

> 이 보고서는 합성 교육 데이터를 사용한 규칙 기반 예시입니다.
"""
st.markdown(report)
st.download_button("보고서 다운로드", report, file_name=f"{place}_{business}_report.md")


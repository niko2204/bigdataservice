# Course 09 학습노트: Streamlit 데이터 서비스

## 1. 학습목표

권장 학습시간은 Streamlit 실행모델 60분, 화면 구현 150분, 사용성 점검 60분이다. 학습 후 사용자 입력–계산–시각화 흐름을 구현하고 캐시, 상태, 오류, 빈 결과와 다운로드를 처리할 수 있어야 한다.

## 2. Streamlit 실행모델

위젯 값이 바뀔 때 Python 스크립트가 위에서 아래로 다시 실행된다. 따라서 느린 데이터 로딩과 모델 학습은 캐시하고, 사용자 세션에서 유지할 값은 `st.session_state`를 사용한다.

```python
import streamlit as st
import pandas as pd

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data("data/sample/location_features.csv")
```

## 3. 화면을 질문 중심으로 설계하기

페이지마다 하나의 핵심 질문을 정한다.

- 홈: 현재 데이터의 범위와 핵심 현황은?
- 상권 현황: 지역별 지표는 어떻게 다른가?
- 업종 분석: 경쟁과 잠재수요가 함께 좋은 지역은?
- 지도: 공간적으로 어디에 몰려 있는가?
- 추천: 내 조건에서 어떤 지역이 상위인가?
- 보고서: 추천 근거와 한계는 무엇인가?

## 4. 입력과 검증

```python
business = st.selectbox("업종", ["카페", "음식점", "편의점"])
min_population = st.number_input("최소 인구", min_value=0, value=5000, step=1000)
filtered = df[df["총인구"] >= min_population]

if filtered.empty:
    st.warning("조건을 만족하는 지역이 없습니다. 최소 인구를 낮춰보세요.")
    st.stop()
```

위젯 기본값은 의미 있는 결과가 나오도록 정한다. 입력 단위, 범위, 영향도 설명을 도움말로 제공한다.

## 5. 추천 화면의 설명가능성

총점만 보여주면 사용자가 판단을 검증할 수 없다. 지표별 점수와 가중치, 장점·약점, 기준일을 함께 표시한다.

```python
st.metric("입지 적합도", f"{score:.1f}점")
st.caption("잠재고객 25% · 유동인구 25% · 경쟁완화 20% · 접근성 15% · 비용효율 15%")
```

사용자가 가중치를 모두 0으로 설정했을 때 기본값으로 돌아가는지 또는 오류를 표시하는지 정책을 명확히 한다.

## 6. 차트와 지도

차트에는 제목, 축 단위, 범례와 데이터 출처를 표시한다. 색상만으로 상태를 구분하지 않고 텍스트·순위를 함께 제공한다. 지도는 데이터가 많을 때 클러스터링 또는 샘플링을 적용한다.

## 7. 다운로드와 재현성

```python
csv = result.to_csv(index=False).encode("utf-8-sig")
st.download_button("추천 결과 CSV", csv, "recommendation.csv", "text/csv")
```

다운로드 결과에는 사용한 입력 조건, 가중치, 데이터 기준일을 포함해야 같은 결과를 재현할 수 있다.

## 8. 오류 처리

```python
try:
    result = calculate_scores(df, business, target_age, weights)
except (KeyError, ValueError) as exc:
    st.error(f"분석을 수행할 수 없습니다: {exc}")
    st.stop()
```

개발 중 모든 예외를 `except Exception`으로 숨기면 오류 원인을 찾기 어렵다. 예상 가능한 오류만 처리하고 로그에는 상세 정보를 남긴다.

## 9. 사용성·접근성 점검

- 처음 보는 사용자가 페이지 목적을 이해하는가?
- 필터가 결과에 어떤 영향을 주는지 보이는가?
- 차트를 읽지 못해도 표와 설명으로 핵심을 알 수 있는가?
- 색상 대비와 글자 크기가 충분한가?
- 키보드로 주요 조작이 가능한가?
- 합성 데이터와 실제 데이터가 명확히 구분되는가?

## 10. 단계별 실습

1. 홈 한 페이지에서 데이터 로딩과 지표 3개를 구현한다.
2. selectbox 하나로 차트를 갱신한다.
3. 빈 결과와 잘못된 입력을 처리한다.
4. 추천 함수와 화면 코드를 분리한다.
5. 다중 페이지 구조로 확장한다.
6. CSV 다운로드에 조건·기준일을 포함한다.
7. 다른 팀 학생 2명에게 과업을 주고 성공 여부·시간·오류를 기록한다.

## 11. 실행 확인

```bash
streamlit run app/Home.py
```

변경 후 홈, 모든 페이지, 필터 극단값, 빈 결과, 다운로드를 순서대로 시험한다.


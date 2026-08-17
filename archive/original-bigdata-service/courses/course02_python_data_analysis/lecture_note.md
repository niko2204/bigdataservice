# Course 02 학습노트: Python과 pandas 데이터 분석

## 1. 학습목표와 선수지식

Python의 변수, 조건문, 반복문, 함수와 리스트를 알고 있다고 가정한다. 권장 학습시간은 이론 70분, 코드 따라하기 90분, 개인 Lab 120분이다.

학습 후 학생은 다음을 할 수 있어야 한다.

- 관측 단위와 변수의 의미를 구분한다.
- pandas로 데이터 구조와 품질을 진단한다.
- 필터링, 파생변수, 그룹 집계를 수행한다.
- 평균·분산·표준점수의 수학적 의미를 코드와 연결한다.
- 분석 결과를 단위와 함께 해석한다.

## 2. 표 형태 데이터의 구조

DataFrame의 한 행은 하나의 관측, 한 열은 하나의 변수다. `location_features.csv`에서 관측 단위는 행정동이고, `stores.csv`에서는 개별 점포다. 관측 단위가 다른 표를 곧바로 합치면 중복 집계가 발생할 수 있다.

```python
import pandas as pd

df = pd.read_csv("data/sample/location_features.csv")
print(df.shape)       # 행 수, 열 수
print(df.columns)     # 변수 이름
print(df.dtypes)      # 저장 자료형
display(df.head(3))   # 실제 값과 단위 추정
```

데이터를 받으면 바로 그래프를 그리지 말고 다음 네 질문에 답한다.

1. 한 행은 무엇을 뜻하는가?
2. 각 열의 단위와 기준 시점은 무엇인가?
3. 식별자 또는 병합 키는 무엇인가?
4. 결측, 중복, 불가능한 값이 있는가?

## 3. 선택·필터·정렬

```python
columns = ["행정동명", "20대인구", "유동인구", "카페수"]
view = df.loc[:, columns]

condition = (df["20대인구"] >= 1500) & (df["카페수"] <= 30)
candidates = df.loc[condition, columns].sort_values("유동인구", ascending=False)
display(candidates)
```

Python의 `and` 대신 Series 조건에는 `&`, `or` 대신 `|`를 사용하고 각 조건을 괄호로 감싼다.

## 4. 파생변수와 0으로 나누기

점포당 인구는 `인구/점포수`지만 점포가 0개일 수 있다. 분모에 무조건 1을 더하면 계산은 되지만 지표 의미가 바뀐다. 목적에 맞는 정책을 명시해야 한다.

```python
import numpy as np

df["카페당20대인구"] = np.where(
    df["카페수"] > 0,
    df["20대인구"] / df["카페수"],
    np.nan,
)
```

점포 0개 지역을 “경쟁이 전혀 없는 기회”로 볼지, “시장 형성이 안 된 지역”으로 볼지는 데이터만으로 결정되지 않는다.

## 5. 그룹 집계

`groupby`는 분할–적용–결합 과정이다.

```python
stores = pd.read_csv("data/sample/stores.csv")
summary = (
    stores.groupby(["행정동명", "상권업종대분류명"], as_index=False)
    .agg(
        점포수=("상가업소번호", "nunique"),
        업종수=("상권업종소분류명", "nunique"),
    )
)
display(summary)
```

`size()`는 결측과 관계없이 행을 세고, `count()`는 지정 열의 결측이 아닌 값을 센다. 점포 식별자가 있다면 `nunique()`가 중복에 더 안전하다.

## 6. 평균·분산·표준점수

표본평균과 표본분산은 다음과 같다.

$$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2$$

표본분산에서 `n-1`을 사용하는 것은 표본평균을 사용하면서 줄어든 자유도를 보정하기 위해서다. 서로 단위가 다른 변수를 비교할 때 z점수를 사용할 수 있다.

$$z_i=\frac{x_i-\bar{x}}{s}$$

```python
x = df["유동인구"]
df["유동인구_z"] = (x - x.mean()) / x.std(ddof=1)
```

z점수는 상대적 위치를 나타낼 뿐 “좋음”을 뜻하지 않는다. 경쟁점포의 높은 z점수는 오히려 불리할 수 있다.

## 7. 메서드 체이닝과 검증

```python
top5 = (
    df.assign(잠재고객비율=df["20대인구"] / df["총인구"])
      .loc[:, ["행정동명", "잠재고객비율", "유동인구"]]
      .sort_values(["잠재고객비율", "유동인구"], ascending=False)
      .head(5)
)

assert len(top5) == 5
assert top5["잠재고객비율"].between(0, 1).all()
```

좋은 분석 코드는 결과를 만들 뿐 아니라 결과가 기대 범위에 있는지 검증한다.

## 8. 단계별 실습

1. 원본 shape, 열, 자료형, 결측 수를 출력한다.
2. 유동인구 상위 5개 행정동을 찾는다.
3. 카페당 20대 인구를 계산하고 분모 0 정책을 기록한다.
4. 상위 지역이 2단계 결과와 다른 이유를 설명한다.
5. Lab 00과 Lab 02에서 손계산과 라이브러리 결과를 비교한다.

## 9. 자주 하는 실수

- 숫자 열이 문자열인데 평균을 계산한다: `pd.to_numeric(..., errors="coerce")`로 변환 결과를 점검한다.
- 원본 DataFrame을 반복 수정해 재실행 결과가 달라진다: `raw`, `clean`, `analysis`를 구분한다.
- 상관관계를 원인으로 해석한다: 관찰과 해석을 별도 문장으로 쓴다.
- 단위 없이 “높다”고 쓴다: 값, 단위, 비교 기준을 함께 쓴다.


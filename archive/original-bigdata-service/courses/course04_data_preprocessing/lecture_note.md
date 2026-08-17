# Course 04 학습노트: 데이터 품질, 전처리와 통합

## 1. 학습목표

전처리는 “분석이 실행되도록 값을 바꾸는 일”이 아니라 데이터 생성 과정과 분석 목적을 바탕으로 품질 문제를 처리하고 그 영향을 기록하는 과정이다. 권장 학습시간은 이론 80분, Lab 01 120분, 통합 실습 90분이다.

## 2. 품질의 여섯 관점

| 관점 | 질문 | 예시 검사 |
|---|---|---|
| 완전성 | 필요한 값이 있는가? | 열별 결측률 |
| 유일성 | 같은 관측이 반복되는가? | 식별자 중복 |
| 유효성 | 허용 범위·형식인가? | 위도, 음수 인구 |
| 일관성 | 서로 모순되지 않는가? | 총인구 < 연령인구 합 |
| 정확성 | 실제 대상을 반영하는가? | 표본 원문 대조 |
| 적시성 | 분석 시점에 적절한가? | 기준월 차이 |

```python
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_n": df.isna().sum(),
    "missing_rate": df.isna().mean(),
    "unique_n": df.nunique(dropna=False),
})
display(quality)
```

## 3. 결측값

결측 발생 메커니즘을 생각한다.

- MCAR: 다른 값과 무관하게 우연히 결측
- MAR: 관찰된 다른 변수와 관련된 결측
- MNAR: 결측된 값 자체와 관련된 결측

무조건 평균으로 대체하면 분산이 작아지고 변수 간 관계가 왜곡될 수 있다. 삭제, 전체 중앙값, 그룹 중앙값, 모델 기반 대체를 비교한다.

```python
overall = df["월임대료"].fillna(df["월임대료"].median())
by_group = df["월임대료"].fillna(
    df.groupby("업종")["월임대료"].transform("median")
)
```

처리 후 결측 수만 확인하지 말고 평균, 중앙값, 표준편차와 분포 변화를 비교한다.

## 4. 중복

완전 중복과 식별자 중복은 다르다.

```python
exact_duplicates = df.duplicated(keep=False)
id_duplicates = stores.duplicated("상가업소번호", keep=False)
```

같은 점포가 여러 기준월에 존재하면 중복이 아니라 시계열 관측일 수 있다. 관측 단위와 기준시점을 먼저 정의한다.

## 5. 이상값

IQR 방법은 다음 경계 밖의 값을 후보로 표시한다.

$$IQR=Q_3-Q_1,\quad [Q_1-1.5IQR,\;Q_3+1.5IQR]$$

```python
q1, q3 = df["월매출"].quantile([.25, .75])
iqr = q3 - q1
mask = ~df["월매출"].between(q1 - 1.5*iqr, q3 + 1.5*iqr)
display(df.loc[mask])
```

이상값 처리 선택지는 원자료 확인, 수정, 제외, 상·하한 절단, 로그변환, 강건 통계 사용이다. 제외할 때는 규칙과 제외 행을 기록한다.

## 6. 자료형과 단위

```python
df["유동인구"] = pd.to_numeric(df["유동인구"], errors="coerce")
df["기준월"] = pd.to_datetime(df["기준월"], format="%Y%m")
df["행정동명"] = df["행정동명"].str.strip().str.replace(r"\s+", " ", regex=True)
```

`errors="coerce"`는 변환 실패를 결측으로 바꾸므로 변환 전후 결측 증가량을 반드시 검사한다.

## 7. 데이터 통합과 카디널리티

병합 전 관계를 정의한다.

- 1:1: 행정동별 인구와 행정동별 집계 점포
- 1:N: 행정동 경계와 개별 점포
- N:M: 그대로 병합하면 행이 곱으로 증가하므로 보통 사전 집계 필요

```python
store_count = stores.groupby("행정동명", as_index=False).agg(점포수=("상가업소번호", "nunique"))
merged = population.merge(store_count, on="행정동명", how="left", validate="one_to_one", indicator=True)
print(merged["_merge"].value_counts())
```

`validate`와 `indicator`는 조용히 발생하는 잘못된 병합을 찾는 중요한 장치다.

## 8. 전처리 파이프라인

함수는 입력을 직접 변경하지 않고 새 DataFrame을 반환하도록 작성한다.

```python
def clean_population(raw: pd.DataFrame) -> pd.DataFrame:
    clean = raw.copy()
    clean.columns = clean.columns.str.strip()
    clean["행정동명"] = clean["행정동명"].str.strip()
    clean["총인구"] = pd.to_numeric(clean["총인구"], errors="coerce")
    clean = clean.drop_duplicates(["기준월", "행정동명"])
    if clean["총인구"].isna().any():
        raise ValueError("총인구 변환 실패 또는 결측이 있습니다.")
    return clean
```

## 9. 단계별 실습과 완료 기준

1. 원본 품질표를 만든다.
2. 결측 처리 2개 방법의 통계 변화를 비교한다.
3. IQR 이상값 후보를 표시하고 삭제 여부를 결정한다.
4. 행정동 이름을 표준화한다.
5. 점포를 행정동 단위로 집계한 후 인구와 1:1 병합한다.
6. 병합 전후 행 수, 미매칭 키, 결측률을 보고서에 남긴다.
7. 같은 코드를 재실행해 같은 결과가 생성되는지 확인한다.


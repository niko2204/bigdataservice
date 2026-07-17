from __future__ import annotations

import pandas as pd


def clean_region_name(series: pd.Series) -> pd.Series:
    """행정동 이름의 앞뒤 공백과 반복 공백을 제거한다."""
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def clean_stores(data: pd.DataFrame) -> pd.DataFrame:
    required = ["상가업소번호", "상호명", "상권업종소분류명", "행정동명", "위도", "경도"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")
    df = data.drop_duplicates("상가업소번호").copy()
    df["행정동명"] = clean_region_name(df["행정동명"])
    df[["위도", "경도"]] = df[["위도", "경도"]].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["행정동명", "위도", "경도"])


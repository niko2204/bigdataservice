"""학번에 따라 재현 가능하지만 서로 다른 실습 데이터를 만든다."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def student_seed(student_id: str) -> int:
    """학번 문자열을 NumPy에서 사용할 수 있는 정수 시드로 변환한다."""
    normalized = str(student_id).strip()
    if len(normalized) < 4:
        raise ValueError("학번을 4자리 이상 입력하세요.")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_student_dataset(student_id: str, rows: int = 180) -> pd.DataFrame:
    """상권 분석용 개인별 합성 데이터에 결측값과 이상값을 주입한다."""
    if rows < 60:
        raise ValueError("rows는 60 이상이어야 합니다.")
    rng = np.random.default_rng(student_seed(student_id))
    regions = np.array(["용당", "목원", "동명", "하당", "신흥", "부주"])
    business = np.array(["카페", "음식점", "편의점"])
    region = rng.choice(regions, rows, p=[.12, .14, .13, .20, .21, .20])
    category = rng.choice(business, rows, p=[.35, .45, .20])
    population = rng.normal(12_000, 2_800, rows).clip(3_000, 22_000)
    floating = population * rng.uniform(.75, 1.65, rows) + rng.normal(0, 1_300, rows)
    competitors = rng.poisson(np.where(category == "음식점", 20, 13), rows)
    parking = rng.poisson(6, rows)
    rent = rng.normal(260, 55, rows).clip(100, 500)
    category_effect = np.select(
        [category == "카페", category == "음식점", category == "편의점"],
        [120, 210, 80], default=0,
    )
    sales = (
        180 + .022 * floating + 11 * parking - 6.5 * competitors
        - .18 * rent + category_effect + rng.normal(0, 85, rows)
    ).clip(40)
    df = pd.DataFrame({
        "지역": region,
        "업종": category,
        "상주인구": population.round(),
        "유동인구": floating.round(),
        "경쟁점포수": competitors,
        "주차장수": parking,
        "월임대료": rent.round(1),
        "월매출": sales.round(1),
    })

    # 모든 학생은 다른 위치에서 같은 유형의 품질 문제를 경험한다.
    missing_idx = rng.choice(df.index, size=max(4, rows // 20), replace=False)
    df.loc[missing_idx[: len(missing_idx) // 2], "유동인구"] = np.nan
    df.loc[missing_idx[len(missing_idx) // 2 :], "월임대료"] = np.nan
    outlier_idx = rng.choice(df.index.difference(missing_idx), size=3, replace=False)
    df.loc[outlier_idx, "월매출"] *= rng.uniform(3.5, 5.0, len(outlier_idx))
    duplicate_idx = rng.choice(df.index, size=3, replace=False)
    return pd.concat([df, df.loc[duplicate_idx]], ignore_index=True)


def load_location_sample() -> pd.DataFrame:
    """저장소의 공통 목포시 교육용 데이터를 읽는다."""
    return pd.read_csv(ROOT / "data/sample/location_features.csv")


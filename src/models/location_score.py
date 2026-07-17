"""설명 가능한 가중합 기반 입지 점수 모델."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["잠재고객", "유동인구", "경쟁완화", "접근성", "비용효율"]
DEFAULT_WEIGHTS = {
    "잠재고객": 0.25,
    "유동인구": 0.25,
    "경쟁완화": 0.20,
    "접근성": 0.15,
    "비용효율": 0.15,
}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """가중치 합을 1로 변환한다."""
    total = sum(max(float(weights.get(k, 0)), 0) for k in FEATURES)
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: max(float(weights.get(k, 0)), 0) / total for k in FEATURES}


def calculate_scores(
    data: pd.DataFrame,
    business_type: str = "카페",
    target_age: str = "20대",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """지역 특성을 0~100으로 정규화하고 추천 순위를 반환한다."""
    df = data.copy()
    customer_col = "20대인구" if target_age == "20대" else "30대인구"
    competitor_col = {"카페": "카페수", "음식점": "음식점수", "편의점": "편의점수"}.get(
        business_type, "카페수"
    )
    raw = pd.DataFrame({
        "잠재고객": df[customer_col],
        "유동인구": df["유동인구"],
        "경쟁완화": -df[competitor_col],
        "접근성": df["주차장수"],
        "비용효율": -df["평균거래가"],
    })
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(raw), columns=FEATURES, index=df.index
    )
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    df["적합도"] = sum(scaled[col] * w[col] for col in FEATURES) * 100
    df["추천이유"] = scaled.apply(
        lambda row: f"{row.nlargest(2).index[0]}·{row.nlargest(2).index[1]} 지표가 상대적으로 우수", axis=1
    )
    return df.sort_values("적합도", ascending=False).reset_index(drop=True).assign(
        순위=lambda x: x.index + 1
    )


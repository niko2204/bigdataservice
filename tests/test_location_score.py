from pathlib import Path
import pandas as pd
from src.models.location_score import calculate_scores, normalize_weights

ROOT = Path(__file__).resolve().parents[1]


def test_weights_sum_to_one():
    weights = normalize_weights({"잠재고객": 2, "유동인구": 2, "경쟁완화": 2, "접근성": 2, "비용효율": 2})
    assert abs(sum(weights.values()) - 1) < 1e-9


def test_score_range_and_rank():
    data = pd.read_csv(ROOT / "data/sample/location_features.csv")
    result = calculate_scores(data)
    assert result["적합도"].between(0, 100).all()
    assert result.iloc[0]["순위"] == 1
    assert result["적합도"].is_monotonic_decreasing


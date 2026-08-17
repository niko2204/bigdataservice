from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_features() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/sample/location_features.csv")


def load_stores() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/sample/stores.csv")


from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/sample/location_features.csv")
df.columns = ["region_name", "latitude", "longitude", "total_population", "population_20s",
              "population_30s", "floating_population", "cafe_count", "restaurant_count",
              "convenience_count", "parking_count", "average_trade_price"]
with sqlite3.connect(ROOT / "database/bigdataservice.db") as conn:
    conn.executescript((ROOT / "database/schema.sql").read_text(encoding="utf-8"))
    df.to_sql("location_features", conn, if_exists="replace", index=False)
print("database/bigdataservice.db 생성 완료")


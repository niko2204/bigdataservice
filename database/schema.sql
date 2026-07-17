CREATE TABLE IF NOT EXISTS location_features (
    region_name TEXT PRIMARY KEY,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    total_population INTEGER,
    population_20s INTEGER,
    population_30s INTEGER,
    floating_population INTEGER,
    cafe_count INTEGER,
    restaurant_count INTEGER,
    convenience_count INTEGER,
    parking_count INTEGER,
    average_trade_price REAL
);


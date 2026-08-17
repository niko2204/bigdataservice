# Course 06 학습노트: 지도와 공간 데이터 분석

## 1. 학습목표

권장 학습시간은 좌표·공간개념 80분, Folium 실습 90분, 공간 지표 설계 90분이다. 학습 후 좌표계를 확인하고 점 데이터를 지도에 표현하며 거리·반경·밀집도 분석의 가정과 한계를 설명할 수 있어야 한다.

## 2. 공간 데이터 유형

- Point: 점포, 정류장, 주차장 위치
- Line: 도로, 버스 노선
- Polygon: 행정동, 상권 경계
- Raster: 위성영상, 인구 격자

이 프로젝트의 `location_features.csv`는 행정동 대표 좌표를 가진 점 형태다. 대표점은 행정동 전체 범위를 뜻하지 않으므로 면적·경계 분석에는 행정동 Polygon이 필요하다.

## 3. 좌표계

위도·경도는 지구의 위치를 각도로 표현하며 보통 WGS84(EPSG:4326)를 사용한다. 국내 공공데이터는 TM 계열 투영좌표를 제공하기도 한다.

```python
print(df[["위도", "경도"]].describe())
assert df["위도"].between(33, 39).all()
assert df["경도"].between(124, 132).all()
```

Folium 위치는 `[위도, 경도]` 순서다. GeoJSON 좌표는 흔히 `[경도, 위도]`이므로 라이브러리 규약을 확인한다.

## 4. 거리

위도·경도를 평면 좌표처럼 빼는 것은 짧은 거리의 근사일 뿐이다. Haversine 공식은 두 지점의 구면 거리를 계산한다.

$$a=\sin^2(\Delta\phi/2)+\cos\phi_1\cos\phi_2\sin^2(\Delta\lambda/2)$$

$$d=2R\arcsin(\sqrt{a})$$

```python
from math import radians, sin, cos, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2 * 6371.0088 * asin(sqrt(a))
```

직선거리는 실제 도보·차량 이동거리와 다르다. 입지 서비스에서는 도로망, 경사, 횡단 가능성 등을 추가로 고려해야 한다.

## 5. Folium 기본 지도

```python
import folium

m = folium.Map(location=[34.804, 126.414], zoom_start=13, tiles="CartoDB positron")
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["위도"], row["경도"]],
        radius=6,
        tooltip=f"{row['행정동명']}: {row['유동인구']:,}",
        color="#2563eb", fill=True,
    ).add_to(m)
m
```

## 6. 값을 원 크기로 표현하기

사람은 원의 반지름이 아니라 면적을 크기로 지각한다. 값에 면적을 비례시키려면 반지름은 값의 제곱근에 비례해야 한다.

```python
import numpy as np

max_radius = 24
df["radius"] = max_radius * np.sqrt(df["유동인구"] / df["유동인구"].max())
```

최솟값이 보이지 않지 않도록 최소 반지름을 더하고 범례에 크기의 의미를 표시한다.

## 7. 공간 집계와 밀집도

반경 내 점포 수는 중심점과 반경 선택에 민감하다. 행정동별 점포 수는 행정동 면적과 인구가 다르면 직접 비교하기 어렵다.

- 면적당 점포 수 = 점포 수 / km²
- 인구당 점포 수 = 점포 수 / 인구 × 1,000
- 최근접 점포 거리 = min(distance)

각 지표가 어떤 의사결정을 대리하는지 설명해야 한다.

## 8. 공간 분석의 함정

- MAUP: 공간 집계 단위가 달라지면 결과가 달라진다.
- 경계 효과: 행정동 경계 밖 가까운 점포를 누락한다.
- 생태학적 오류: 지역 평균을 개인 특성으로 해석한다.
- 위치 개인정보: 개인·가구 수준 정밀 좌표 공개 위험이 있다.

## 9. 단계별 실습

1. 위도·경도 범위와 결측을 검사한다.
2. 행정동 대표점을 지도에 표시한다.
3. 값의 제곱근으로 반지름을 계산한다.
4. 업종 필터와 툴팁을 추가한다.
5. 두 점의 Haversine 거리를 손계산 구조와 함수로 확인한다.
6. 직선거리와 실제 이동거리 차이를 보고서에 기록한다.
7. 지도 캡처가 아니라 실행 가능한 HTML 또는 Streamlit 화면을 제출한다.

## 10. 완료 체크

- [ ] 좌표계와 좌표 순서를 기록했다.
- [ ] 지도 범례와 단위를 표시했다.
- [ ] 원 크기가 값을 과장하지 않는다.
- [ ] 공간 집계 단위의 한계를 설명했다.
- [ ] 위치정보 공개 수준을 검토했다.


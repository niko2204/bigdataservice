"""프로젝트 흐름을 따라가는 상세 Notebook 5개를 생성한다."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(text).strip().splitlines(True)}


SETUP = '''
from pathlib import Path
import sys, subprocess

REPO_URL = "https://github.com/niko2204/bigdataservice.git"
if "google.colab" in sys.modules:
    ROOT = Path("/content/bigdataservice")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)
else:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    ROOT = next((p.resolve() for p in candidates if (p / "data").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("bigdataservice 저장소 안에서 실행하세요.")
print("저장소:", ROOT)
'''


def save(name, cells):
    doc = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": name, "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def notebook01():
    return [
        md('''
        # 프로젝트 Notebook 01. pandas로 상권 데이터 이해하기

        목표: CSV를 읽고 관측 단위·자료형·품질을 확인한 뒤 필터, 파생변수, 집계와 기본 시각화를 수행한다.
        Course 02의 lecture_note.md를 먼저 읽고 이 Notebook을 실행한다.
        '''),
        code(SETUP),
        md('''
        ## 1. 읽기와 관측 단위

        location_features는 행정동 한 행, stores는 개별 점포 한 행이다. 관측 단위가 다른 표를 바로 병합하면 행이 늘어날 수 있다.
        '''),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        locations = pd.read_csv(ROOT / "data/sample/location_features.csv")
        stores = pd.read_csv(ROOT / "data/sample/stores.csv")
        print("지역:", locations.shape, "점포:", stores.shape)
        display(locations.head(3))
        display(stores.head(3))
        '''),
        md('''
        ## 2. 구조와 품질 프로파일

        자료형, 결측, 고유값 수와 식별자 중복을 확인한다. 열 이름만 보고 단위를 추측하지 말고 데이터 사전과 대조한다.
        '''),
        code('''
        profile = pd.DataFrame({
            "자료형": locations.dtypes.astype(str),
            "결측수": locations.isna().sum(),
            "고유값수": locations.nunique(dropna=False),
        })
        display(profile)
        print("행정동명 중복:", locations["행정동명"].duplicated().sum())
        print("점포번호 중복:", stores["상가업소번호"].duplicated().sum())
        assert locations["행정동명"].is_unique
        '''),
        md('''
        ## 3. 조건 필터와 정렬

        Series 조건에는 and 대신 &, or 대신 |를 쓰고 각 조건을 괄호로 묶는다.
        '''),
        code('''
        condition = (locations["20대인구"] >= 1500) & (locations["카페수"] <= 30)
        candidates = (
            locations.loc[condition, ["행정동명", "20대인구", "유동인구", "카페수"]]
                     .sort_values("유동인구", ascending=False)
        )
        display(candidates)
        '''),
        md('''
        ## 4. 파생변수와 0으로 나누기

        점포가 0개일 때 분모에 무조건 1을 더하면 지표 의미가 달라진다. 여기서는 계산 불가인 NaN으로 둔다.
        '''),
        code('''
        locations["카페당20대인구"] = np.where(
            locations["카페수"] > 0,
            locations["20대인구"] / locations["카페수"],
            np.nan,
        )
        display(locations[["행정동명", "카페당20대인구"]].sort_values("카페당20대인구", ascending=False).head())
        '''),
        md('''
        ## 5. 그룹 집계

        점포 식별자의 nunique를 사용하여 중복에 안전하게 집계한다.
        '''),
        code('''
        store_summary = (
            stores.groupby(["행정동명", "상권업종대분류명"], as_index=False)
                  .agg(점포수=("상가업소번호", "nunique"),
                       세부업종수=("상권업종소분류명", "nunique"))
        )
        display(store_summary)
        '''),
        md('''
        ## 6. 시각화와 세 문장 해석
        '''),
        code('''
        ordered = locations.sort_values("유동인구")
        plt.figure(figsize=(8, 5))
        plt.barh(ordered["행정동명"], ordered["유동인구"])
        plt.xlabel("유동인구 지수"); plt.ylabel("행정동")
        plt.title("목포시 교육용 행정동별 유동인구")
        plt.tight_layout(); plt.show()
        '''),
        md('''
        관찰에는 그래프에서 직접 읽은 값, 해석에는 가능한 의미, 한계에는 데이터로 말할 수 없는 것을 쓴다.

        ## 7. 독립 연습

        1. 음식점당 총인구와 편의점당 총인구를 계산한다.
        2. 세 업종의 점포당 인구 상위 5개를 한 표로 비교한다.
        3. 유동인구 상위와 점포당 인구 상위가 다른 이유를 5문장으로 설명한다.
        4. 행 수와 값 범위를 확인하는 assert를 추가한다.
        '''),
        code('''
        # TODO: 세 업종 비교표
        comparison = None
        display(comparison)
        '''),
    ]


def notebook02():
    return [
        md('''
        # 프로젝트 Notebook 02. 공공데이터 API 수집

        목표: HTTP 요청, 인증키, JSON 파싱, 페이지네이션, 오류 처리와 원본 보존을 학습한다.
        API 키가 없어도 내장 예제 응답으로 파싱 부분을 실행할 수 있다.
        '''),
        code(SETUP),
        md('''
        ## 1. 요청 구성과 키 관리

        키는 .env 또는 Colab Secrets에 저장한다. Notebook에 직접 적거나 출력하지 않는다.
        '''),
        code('''
        import os, json, time
        from datetime import datetime, timezone
        import requests
        import pandas as pd
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        API_KEY = os.getenv("DATA_GO_KR_API_KEY")
        BASE_URL = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"
        params = {
            "serviceKey": API_KEY, "cx": 126.3922, "cy": 34.8118,
            "radius": 1000, "pageNo": 1, "numOfRows": 10, "type": "json",
        }
        print("인증키 설정 여부:", bool(API_KEY))
        '''),
        md('''
        ## 2. 키 없이 JSON 파싱 연습

        중첩 구조의 키를 한 단계씩 확인한다.
        '''),
        code('''
        DEMO_RESPONSE = {
            "body": {
                "totalCount": 2,
                "items": [
                    {"bizesId": "S001", "bizesNm": "유달커피", "indsSclsNm": "카페", "lon": 126.3821, "lat": 34.7915},
                    {"bizesId": "S002", "bizesNm": "평화카페", "indsSclsNm": "카페", "lon": 126.4172, "lat": 34.8040},
                ],
            }
        }
        body = DEMO_RESPONSE["body"]
        demo_df = pd.DataFrame(body.get("items", []))
        display(demo_df)
        assert len(demo_df) == body["totalCount"]
        '''),
        md('''
        ## 3. 실제 10건 호출

        HTTP 200만 확인하지 말고 본문의 결과 코드와 건수도 확인한다.
        '''),
        code('''
        if API_KEY:
            response = requests.get(BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            print("상위 키:", payload.keys())
        else:
            print("API 키가 없어 예제 응답을 사용합니다.")
            payload = DEMO_RESPONSE
        '''),
        md('''
        ## 4. 재사용 가능한 파서

        예상 구조가 아니면 조용히 빈 표를 만들지 않고 오류를 발생시킨다.
        '''),
        code('''
        def parse_items(payload):
            body = payload.get("body")
            if not isinstance(body, dict):
                raise ValueError("응답에 body 객체가 없습니다.")
            items = body.get("items", [])
            if isinstance(items, dict):
                items = items.get("item", [])
            if not isinstance(items, list):
                raise ValueError("items가 리스트가 아닙니다.")
            return pd.DataFrame(items), int(body.get("totalCount", len(items)))

        page_df, total_count = parse_items(payload)
        print("현재:", len(page_df), "전체:", total_count)
        display(page_df.head())
        '''),
        md('''
        ## 5. 페이지네이션

        종료 조건과 최대 페이지를 함께 둔다. 페이지, 현재 건수, 누적 건수와 전체 건수를 로그로 남긴다.
        '''),
        code('''
        def collect_pages(base_url, base_params, max_pages=20, wait_seconds=.2):
            if not base_params.get("serviceKey"):
                raise RuntimeError("DATA_GO_KR_API_KEY가 필요합니다.")
            frames, collected = [], 0
            for page in range(1, max_pages + 1):
                query = {**base_params, "pageNo": page}
                response = requests.get(base_url, params=query, timeout=20)
                response.raise_for_status()
                frame, total = parse_items(response.json())
                print(f"page={page}, batch={len(frame)}, accumulated={collected}, total={total}")
                if frame.empty:
                    break
                frames.append(frame); collected += len(frame)
                if collected >= total:
                    break
                time.sleep(wait_seconds)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        '''),
        md('''
        ## 6. 수집 메타데이터
        '''),
        code('''
        metadata = {
            "source": BASE_URL,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "parameters_without_key": {k:v for k,v in params.items() if k != "serviceKey"},
            "rows_in_current_page": len(page_df),
        }
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        '''),
        md('''
        ## 7. 독립 연습

        1. 실제 응답 필드 다섯 개의 정의를 활용가이드에서 찾아 표로 작성한다.
        2. 100건 이상 수집하고 식별자 유일성, 좌표 범위와 결측을 검사한다.
        3. timeout·429·500대 오류를 제한적으로 재시도하도록 개선한다.
        4. 원본 건수와 정제 CSV 건수 차이를 자동 기록한다.
        '''),
    ]


def notebook03():
    return [
        md('''
        # 프로젝트 Notebook 03. 전처리, 통합과 EDA

        목표: 지역·점포 데이터를 품질 검사하고 관측 단위를 맞춰 통합한 뒤 질문 중심 EDA를 수행한다.
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        locations = pd.read_csv(ROOT / "data/sample/location_features.csv")
        stores = pd.read_csv(ROOT / "data/sample/stores.csv")
        '''),
        md('''
        ## 1. 원본 품질표
        '''),
        code('''
        def quality_profile(df):
            return pd.DataFrame({
                "dtype": df.dtypes.astype(str),
                "missing_n": df.isna().sum(),
                "missing_rate": df.isna().mean(),
                "unique_n": df.nunique(dropna=False),
            })
        display(quality_profile(locations))
        display(quality_profile(stores))
        '''),
        md('''
        ## 2. 점포 정제와 행정동 집계

        개별 점포를 행정동별·대분류별 점포 수로 집계하여 지역 표와 관측 단위를 맞춘다.
        '''),
        code('''
        clean_stores = stores.drop_duplicates("상가업소번호").copy()
        clean_stores["행정동명"] = clean_stores["행정동명"].str.strip()
        clean_stores[["위도", "경도"]] = clean_stores[["위도", "경도"]].apply(pd.to_numeric, errors="coerce")
        clean_stores = clean_stores.dropna(subset=["행정동명", "위도", "경도"])

        counts = clean_stores.pivot_table(
            index="행정동명", columns="상권업종대분류명",
            values="상가업소번호", aggfunc="nunique", fill_value=0
        ).reset_index()
        display(counts)
        '''),
        md('''
        ## 3. 검증 가능한 병합

        validate로 1:1을 확인하고 indicator로 미매칭 지역을 찾는다.
        '''),
        code('''
        merged = locations.merge(
            counts, on="행정동명", how="left",
            validate="one_to_one", indicator=True
        )
        print(merged["_merge"].value_counts())
        assert len(merged) == len(locations)
        display(merged.loc[merged["_merge"] != "both"])
        '''),
        md('''
        ## 4. 질문 1: 유동인구와 카페 수는 함께 증가하는가?
        '''),
        code('''
        r = merged["유동인구"].corr(merged["카페수"])
        plt.figure(figsize=(7, 4))
        plt.scatter(merged["카페수"], merged["유동인구"])
        for _, row in merged.iterrows():
            plt.annotate(row["행정동명"], (row["카페수"], row["유동인구"]), fontsize=8)
        plt.xlabel("카페 수(개)"); plt.ylabel("유동인구 지수")
        plt.title(f"카페 수와 유동인구: r={r:.2f}")
        plt.show()
        '''),
        md('''
        표본은 12개 지역뿐이고 합성 데이터이므로 상관을 실제 목포시 관계로 일반화할 수 없다.

        ## 5. 질문 2: 경쟁 대비 잠재고객
        '''),
        code('''
        merged["카페당20대인구"] = merged["20대인구"] / merged["카페수"].replace(0, np.nan)
        result = merged[["행정동명", "20대인구", "카페수", "카페당20대인구"]].sort_values(
            "카페당20대인구", ascending=False
        )
        display(result)
        '''),
        md('''
        ## 6. 독립 연습

        1. 분포·집단비교·관계 유형의 분석 질문을 하나씩 작성한다.
        2. 각 질문에 표와 그래프로 답한다.
        3. 제목, 단위, 표본 수와 데이터 범위를 표시한다.
        4. 관찰·해석·한계를 각각 두 문장 작성한다.
        5. inner join으로 바꾸었을 때 누락되는 지역을 확인한다.
        '''),
    ]


def notebook04():
    return [
        md('''
        # 프로젝트 Notebook 04. 공간 데이터와 Folium 지도

        목표: 좌표를 검증하고 값의 크기를 과장하지 않는 지도를 만든 뒤 거리와 공간 집계의 한계를 설명한다.
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import folium
        from math import radians, sin, cos, asin, sqrt
        df = pd.read_csv(ROOT / "data/sample/location_features.csv")
        '''),
        md('''
        ## 1. 좌표 품질과 좌표계

        WGS84 위도·경도를 가정한다. 목포 분석 범위를 벗어난 값과 결측을 검사한다.
        Folium은 [위도, 경도] 순서이지만 GeoJSON은 흔히 [경도, 위도] 순서다.
        '''),
        code('''
        print(df[["위도", "경도"]].describe())
        invalid = df.loc[
            ~df["위도"].between(33, 39) |
            ~df["경도"].between(124, 132) |
            df[["위도", "경도"]].isna().any(axis=1)
        ]
        display(invalid)
        assert invalid.empty
        '''),
        md('''
        ## 2. 원 면적에 값을 비례시키기

        원의 면적은 반지름 제곱에 비례하므로 반지름에는 값의 제곱근을 사용한다.
        '''),
        code('''
        max_radius = 22
        df["marker_radius"] = 4 + max_radius * np.sqrt(df["유동인구"] / df["유동인구"].max())
        display(df[["행정동명", "유동인구", "marker_radius"]].head())
        '''),
        md('''
        ## 3. Folium 지도
        '''),
        code('''
        m = folium.Map(
            location=[df["위도"].mean(), df["경도"].mean()],
            zoom_start=13, tiles="CartoDB positron"
        )
        for _, row in df.iterrows():
            folium.CircleMarker(
                [row["위도"], row["경도"]],
                radius=row["marker_radius"],
                tooltip=f"{row['행정동명']} | 유동인구 {row['유동인구']:,}",
                color="#2563eb", fill=True, fill_opacity=.55,
            ).add_to(m)
        m
        '''),
        md(r'''
        ## 4. Haversine 직선거리

        $$a=\sin^2(\Delta\phi/2)+\cos\phi_1\cos\phi_2\sin^2(\Delta\lambda/2)$$

        대표점 직선거리이며 실제 도보·차량 이동거리와 다르다.
        '''),
        code('''
        def haversine_km(lat1, lon1, lat2, lon2):
            p1, p2 = radians(lat1), radians(lat2)
            dp, dl = radians(lat2-lat1), radians(lon2-lon1)
            a = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
            return 2 * 6371.0088 * asin(sqrt(a))

        a, b = df.iloc[0], df.iloc[1]
        distance = haversine_km(a["위도"], a["경도"], b["위도"], b["경도"])
        print(a["행정동명"], b["행정동명"], round(distance, 3), "km")
        '''),
        md('''
        ## 5. 독립 연습

        1. 표현 지표를 총인구, 카페수, 음식점수로 바꾸는 함수를 작성한다.
        2. 색상은 적합도, 원 면적은 유동인구를 표현한다.
        3. 선택한 행정동과 다른 모든 행정동 거리표를 작성한다.
        4. 최근접 지역이 실제 이동시간에서도 최근접이라고 말할 수 없는 이유를 설명한다.
        5. MAUP, 경계 효과와 생태학적 오류를 각각 현재 지도와 연결해 설명한다.
        '''),
        code('''
        # TODO: 지표 이름을 받아 지도를 반환하는 함수
        def make_metric_map(data, metric):
            pass
        '''),
    ]


def notebook05():
    return [
        md(r'''
        # 프로젝트 Notebook 05. 설명 가능한 입지 추천

        목표: 기준 방향을 통일하고 정규화·가중합 점수를 계산하여 추천 근거와 민감도를 제시한다.

        $$S_i=\sum_j w_jz_{ij},\qquad \sum_jw_j=1$$
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(ROOT / "data/sample/location_features.csv")
        '''),
        md('''
        ## 1. 기준과 방향

        잠재고객·유동인구·접근성은 클수록 좋고 경쟁·비용은 작을수록 좋다고 가정한다.
        이 방향은 목적에 따른 선택이며 객관적 진실이 아니다.
        '''),
        code('''
        raw_features = pd.DataFrame({
            "잠재고객": df["20대인구"],
            "유동인구": df["유동인구"],
            "경쟁완화": -df["카페수"],
            "접근성": df["주차장수"],
            "비용효율": -df["평균거래가"],
        })
        display(raw_features.head())
        '''),
        md('''
        ## 2. Min-Max 정규화

        모든 값이 같은 열은 범위가 0이므로 중립값 0.5를 부여한다.
        '''),
        code('''
        def minmax(series):
            span = series.max() - series.min()
            if span == 0:
                return pd.Series(0.5, index=series.index)
            return (series - series.min()) / span

        normalized = raw_features.apply(minmax)
        assert normalized.min().min() >= 0
        assert normalized.max().max() <= 1
        display(normalized.round(3))
        '''),
        md('''
        ## 3. 가중합 점수와 순위
        '''),
        code('''
        weights = pd.Series({
            "잠재고객": .25, "유동인구": .25, "경쟁완화": .20,
            "접근성": .15, "비용효율": .15,
        })
        assert np.isclose(weights.sum(), 1)

        result = df[["행정동명"]].copy()
        result["적합도"] = normalized.mul(weights, axis=1).sum(axis=1) * 100
        result["순위"] = result["적합도"].rank(method="min", ascending=False).astype(int)
        result = result.sort_values("순위")
        display(result)
        '''),
        md('''
        ## 4. 지표별 점수 기여

        총점뿐 아니라 지표별 기여도를 보여야 사용자가 추천을 검증할 수 있다.
        '''),
        code('''
        contribution = normalized.mul(weights, axis=1) * 100
        contribution.insert(0, "행정동명", df["행정동명"])
        top_index = result.index[0]
        top_contribution = contribution.loc[top_index].drop("행정동명").sort_values(ascending=False)
        print("1위:", df.loc[top_index, "행정동명"])
        display(top_contribution.rename("점수기여").to_frame())
        '''),
        md('''
        ## 5. 민감도 분석

        선호가 달라져도 상위 지역이 유지되는지 세 시나리오를 비교한다.
        '''),
        code('''
        scenarios = {
            "균형": [.25, .25, .20, .15, .15],
            "유동중심": [.15, .45, .15, .10, .15],
            "저비용": [.15, .15, .15, .10, .45],
        }
        ranking = pd.DataFrame({"행정동명": df["행정동명"]})
        for name, values in scenarios.items():
            w = pd.Series(values, index=normalized.columns)
            score = normalized.mul(w, axis=1).sum(axis=1)
            ranking[name] = score.rank(method="min", ascending=False).astype(int)
        ranking["순위범위"] = ranking[list(scenarios)].max(axis=1) - ranking[list(scenarios)].min(axis=1)
        display(ranking.sort_values("균형"))
        '''),
        md('''
        순위범위가 크면 하나의 절대적 추천 대신 조건별 후보를 제시한다.

        ## 6. 독립 연습

        1. 업종을 음식점·편의점으로 바꾸어 경쟁 기준을 변경한다.
        2. 30대 고객 시나리오를 추가한다.
        3. Min-Max 대신 z점수 또는 순위점수를 적용해 상위 5개를 비교한다.
        4. 각 가중치를 ±10% 변화시키는 민감도 분석을 자동화한다.
        5. 추천 1위의 강점·약점과 실제 적용 전 필요한 데이터를 작성한다.
        '''),
        code('''
        # TODO: 업종·고객·가중치를 인자로 받는 함수
        def recommend(data, business_type, target_age, weights):
            pass
        '''),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_pandas_basics.ipynb": notebook01(),
        "02_public_api.ipynb": notebook02(),
        "03_preprocessing_eda.ipynb": notebook03(),
        "04_geospatial.ipynb": notebook04(),
        "05_recommendation.ipynb": notebook05(),
    }
    for name, cells in notebooks.items():
        save(name, cells)
    print(f"{len(notebooks)}개 프로젝트 Notebook 생성 완료")


if __name__ == "__main__":
    main()

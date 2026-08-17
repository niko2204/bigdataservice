"""상세 데이터 분석 실습 Notebook 8개를 재현 가능하게 생성한다."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "data_analysis"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    }


SETUP = r'''
from pathlib import Path
import sys, subprocess

REPO_URL = "https://github.com/niko2204/bigdataservice.git"
if "google.colab" in sys.modules:
    ROOT = Path("/content/bigdataservice")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)
else:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    ROOT = next((p.resolve() for p in candidates if (p / "src").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("bigdataservice 저장소 루트에서 Notebook을 실행하세요.")

sys.path.insert(0, str(ROOT))
STUDENT_ID = "20260001"  # 반드시 본인 학번으로 변경
print("저장소:", ROOT)
print("실습 학번:", STUDENT_ID)
'''


def save(name: str, cells: list[dict]) -> None:
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": name, "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / name).write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")


def lab00() -> list[dict]:
    return [
        md(r'''
        # Lab 00. 데이터 분석을 위한 수학과 NumPy

        이 Notebook은 평균, 편차, 분산, 벡터 거리와 표준화가 실제 코드에서 어떻게 사용되는지 학습한다.
        먼저 완성 예제를 실행하고, 그 다음 연습 문제의 TODO를 직접 완성한다.

        학습 후 다음을 할 수 있어야 한다.

        1. 합과 평균을 반복문과 NumPy로 각각 계산한다.
        2. 편차 합이 0이 되는 이유를 설명한다.
        3. 표본분산에서 n-1을 사용하는 의미를 말한다.
        4. 유클리드 거리를 계산하고 단위가 다른 변수의 문제를 설명한다.

        평균과 표본분산:

        $$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad
        s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2$$
        '''),
        code(SETUP),
        md('''
        ## 1. 완성 예제: 평균과 편차

        다섯 값의 합을 개수로 나누어 평균을 계산한다. 각 값에서 평균을 뺀 편차의 합은 부동소수점 오차를 제외하면 0이다.
        '''),
        code('''
        import numpy as np

        x = np.array([8, 10, 12, 14, 16], dtype=float)
        manual_sum = 0
        for value in x:
            manual_sum += value
        manual_mean = manual_sum / len(x)

        deviations = x - manual_mean
        print("반복문 평균:", manual_mean)
        print("NumPy 평균:", x.mean())
        print("편차:", deviations)
        print("편차 합:", deviations.sum())
        assert np.isclose(manual_mean, x.mean())
        assert np.isclose(deviations.sum(), 0)
        '''),
        md('''
        ## 2. 완성 예제: 모집단분산과 표본분산

        NumPy의 ddof는 분모에서 빼는 자유도다. ddof=0은 n으로 나누고, ddof=1은 n-1로 나눈다.
        '''),
        code('''
        squared_sum = ((x - x.mean()) ** 2).sum()
        population_variance = squared_sum / len(x)
        sample_variance = squared_sum / (len(x) - 1)

        print("편차제곱합:", squared_sum)
        print("모집단분산:", population_variance, np.var(x, ddof=0))
        print("표본분산:", sample_variance, np.var(x, ddof=1))
        assert np.isclose(sample_variance, np.var(x, ddof=1))
        '''),
        md(r'''
        ## 3. 완성 예제: 벡터 거리와 척도

        두 벡터의 유클리드 거리는 다음과 같다.

        $$d(a,b)=\sqrt{\sum_j(a_j-b_j)^2}$$

        인구처럼 수천 단위인 변수는 주차장처럼 한 자리 단위인 변수보다 거리 계산을 거의 전부 지배한다.
        '''),
        code('''
        A = np.array([12000, 18000, 8], dtype=float)
        B = np.array([9000, 15000, 5], dtype=float)
        difference = A - B
        distance = np.sqrt((difference ** 2).sum())

        combined = np.vstack([A, B])
        mean = combined.mean(axis=0)
        std = combined.std(axis=0, ddof=0)
        scaled = (combined - mean) / std
        scaled_distance = np.linalg.norm(scaled[0] - scaled[1])

        print("변수별 차이:", difference)
        print("원자료 거리:", distance)
        print("표준화 후 거리:", scaled_distance)
        '''),
        md('''
        ## 4. 따라하기

        values를 본인 학번 마지막 네 자리의 각 숫자로 바꾼다. 0만 나오면 각 숫자에 1을 더한다.
        반복문으로 평균과 표본분산을 계산한 뒤 NumPy 결과와 비교한다.
        '''),
        code('''
        values = np.array([2, 0, 2, 6, 1, 7, 0, 1], dtype=float)  # 본인 값으로 변경

        # TODO: np.mean, np.var를 사용하지 않고 계산
        my_mean = None
        my_variance = None

        # 완성 후 주석을 제거하여 검증
        # assert np.isclose(my_mean, values.mean())
        # assert np.isclose(my_variance, values.var(ddof=1))
        print("직접 계산:", my_mean, my_variance)
        '''),
        md('''
        ## 5. 독립 연습과 해석

        1. 세 지역을 [유동인구, 경쟁점포수, 월임대료] 벡터로 표현한다.
        2. 표준화 전후 모든 지역 쌍의 거리를 계산한다.
        3. 가장 가까운 지역 쌍이 바뀌었는지 확인한다.
        4. 바뀌었다면 어떤 변수가 원자료 거리를 지배했는지 단위와 함께 설명한다.
        '''),
        code('''
        regions = np.array([
            [18000, 12, 250],
            [15000,  9, 340],
            [21000, 24, 230],
        ], dtype=float)

        # TODO: 표준화 전후 3x3 거리행렬을 작성
        raw_distance_matrix = None
        scaled_distance_matrix = None
        print(raw_distance_matrix)
        print(scaled_distance_matrix)
        '''),
        md('''
        ## 6. 자가점검

        - [ ] 평균과 표본분산을 라이브러리 없이 계산했다.
        - [ ] ddof=0과 ddof=1의 차이를 말할 수 있다.
        - [ ] 거리에서 제곱과 제곱근의 역할을 설명할 수 있다.
        - [ ] 표준화 전후 거리가 다른 이유를 변수 단위로 설명했다.
        '''),
    ]


def lab01() -> list[dict]:
    return [
        md(r'''
        # Lab 01. 데이터 품질과 전처리

        전처리의 목표는 결측을 0개로 만드는 것이 아니라 분석 목적에 맞는 품질 기준을 세우고 처리 영향을 기록하는 것이다.
        이 Notebook에서는 학생별로 다른 결측, 중복과 이상값을 가진 데이터로 연습한다.

        IQR 이상값 후보 경계:

        $$IQR=Q_3-Q_1,\qquad [Q_1-1.5IQR,\;Q_3+1.5IQR]$$
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from src.education.personalized_data import make_student_dataset

        raw = make_student_dataset(STUDENT_ID)
        print("원본 크기:", raw.shape)
        display(raw.head())
        '''),
        md('''
        ## 1. 완성 예제: 품질 프로파일

        자료형, 결측 수·비율, 고유값 수를 한 표로 만든다. 완전 중복과 특정 식별자 중복은 구분해야 한다.
        '''),
        code('''
        quality = pd.DataFrame({
            "자료형": raw.dtypes.astype(str),
            "결측수": raw.isna().sum(),
            "결측률": raw.isna().mean().round(3),
            "고유값수": raw.nunique(dropna=False),
        })
        print("완전 중복 행:", raw.duplicated().sum())
        display(quality)
        '''),
        md('''
        ## 2. 완성 예제: 결측 대체 방법 비교

        전체 중앙값과 업종별 중앙값을 비교한다. 평균 대체도 가능하지만 이상값에 민감하고 분산을 줄일 수 있다.
        '''),
        code('''
        compare = raw.copy()
        compare["임대료_전체중앙값"] = compare["월임대료"].fillna(compare["월임대료"].median())
        compare["임대료_업종중앙값"] = compare["월임대료"].fillna(
            compare.groupby("업종")["월임대료"].transform("median")
        )
        rows = raw["월임대료"].isna()
        display(compare.loc[rows, ["업종", "월임대료", "임대료_전체중앙값", "임대료_업종중앙값"]])
        '''),
        md('''
        ## 3. 완성 예제: IQR 이상값은 후보

        경계 밖 행을 바로 삭제하지 않고 업종, 지역과 다른 변수를 함께 확인한다.
        '''),
        code('''
        sales = raw["월매출"].dropna()
        q1, q3 = sales.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = ~raw["월매출"].between(lower, upper)
        print(f"Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}, 경계=({lower:.1f}, {upper:.1f})")
        display(raw.loc[outlier_mask].sort_values("월매출", ascending=False))
        '''),
        md('''
        ## 4. 따라하기: 정제 파이프라인

        아래 정책은 하나의 예일 뿐이다. 유동인구와 임대료는 업종별 중앙값으로 대체하고 완전 중복을 제거한다.
        이상값은 분석용 플래그만 만들고 원자료 값은 유지한다.
        '''),
        code('''
        clean = raw.drop_duplicates().copy()
        for column in ["유동인구", "월임대료"]:
            clean[column] = clean[column].fillna(
                clean.groupby("업종")[column].transform("median")
            )
        clean["매출_이상후보"] = ~clean["월매출"].between(lower, upper)

        assert clean.duplicated().sum() == 0
        assert clean[["유동인구", "월임대료"]].isna().sum().sum() == 0
        print("정제 크기:", clean.shape)
        display(clean["매출_이상후보"].value_counts())
        '''),
        md('''
        ## 5. 처리 전후 영향 비교

        결측 대체와 중복 제거가 평균, 중앙값과 표준편차에 미친 영향을 확인한다.
        '''),
        code('''
        before = raw["월임대료"].agg(["count", "mean", "median", "std"])
        after = clean["월임대료"].agg(["count", "mean", "median", "std"])
        comparison = pd.concat([before.rename("처리전"), after.rename("처리후")], axis=1)
        display(comparison)
        '''),
        md('''
        ## 6. 독립 연습

        1. 유동인구 결측을 전체 평균, 전체 중앙값, 업종별 중앙값으로 각각 대체한다.
        2. 세 방법의 평균·표준편차와 유동인구–월매출 상관계수를 비교한다.
        3. IQR 계수를 1.5와 3.0으로 바꾸어 이상 후보 수를 비교한다.
        4. 최종 처리 정책을 선택하고 데이터 생성 과정과 분석 목적을 근거로 5문장 이상 설명한다.
        '''),
        code('''
        # TODO: 세 대체 방법 비교표 작성
        imputation_comparison = None
        display(imputation_comparison)

        # TODO: IQR 1.5와 3.0의 이상 후보 행 번호와 개수 비교
        outlier_comparison = None
        display(outlier_comparison)
        '''),
        md('''
        ## 7. 자가점검

        - [ ] 원본 DataFrame을 변경하지 않았다.
        - [ ] 결측·중복·이상값을 서로 다른 문제로 처리했다.
        - [ ] 이상값을 삭제하기 전에 원인과 영향을 확인했다.
        - [ ] 처리 전후 행 수와 통계량을 기록했다.
        - [ ] 선택하지 않은 대안의 결과도 비교했다.
        '''),
    ]


def lab02() -> list[dict]:
    return [
        md(r'''
        # Lab 02. 기술통계, 표준편차와 z점수

        평균은 중심을, 분산과 표준편차는 퍼짐을 나타낸다. z점수는 관측값이 평균에서 표준편차 몇 배 떨어져 있는지 표현한다.
        z점수가 크다는 사실만으로 좋은 지역 또는 오류라고 판단할 수 없다.
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from src.education.personalized_data import make_student_dataset

        df = make_student_dataset(STUDENT_ID).drop_duplicates()
        df["유동인구"] = df.groupby("업종")["유동인구"].transform(lambda s: s.fillna(s.median()))
        df["월임대료"] = df.groupby("업종")["월임대료"].transform(lambda s: s.fillna(s.median()))
        '''),
        md('''
        ## 1. 작은 데이터로 공식을 검증

        먼저 다섯 값으로 편차, 편차제곱과 표본분산을 한 단계씩 확인한다.
        '''),
        code('''
        x = np.array([2, 4, 4, 5, 10], dtype=float)
        x_bar = x.sum() / len(x)
        table = pd.DataFrame({"x": x, "편차": x - x_bar, "편차제곱": (x - x_bar) ** 2})
        sample_variance = table["편차제곱"].sum() / (len(x) - 1)
        sample_std = np.sqrt(sample_variance)

        display(table)
        print("평균:", x_bar, "표본분산:", sample_variance, "표본표준편차:", sample_std)
        assert np.isclose(sample_variance, np.var(x, ddof=1))
        '''),
        md('''
        ## 2. 학생별 데이터 기술통계

        평균과 중앙값의 차이가 크면 비대칭 분포나 이상값을 의심할 수 있다.
        '''),
        code('''
        numeric = ["상주인구", "유동인구", "경쟁점포수", "주차장수", "월임대료", "월매출"]
        summary = df[numeric].agg(["count", "mean", "median", "std", "min", "max"]).T
        summary["평균-중앙값"] = summary["mean"] - summary["median"]
        display(summary.round(2))
        '''),
        md('''
        ## 3. z점수 직접 계산

        pandas 표본표준편차 ddof=1을 사용한다. 절댓값 3 초과는 확인 후보일 뿐 자동 삭제 규칙이 아니다.
        '''),
        code('''
        mean_sales = df["월매출"].mean()
        std_sales = df["월매출"].std(ddof=1)
        df["매출_z"] = (df["월매출"] - mean_sales) / std_sales
        extreme = df.loc[df["매출_z"].abs() > 3].sort_values("매출_z", key=abs, ascending=False)
        print("z점수 평균:", df["매출_z"].mean())
        print("z점수 표준편차:", df["매출_z"].std(ddof=1))
        display(extreme)
        '''),
        md('''
        ## 4. 업종별 기술통계

        전체 평균은 업종 구성비의 영향을 받는다. 집단별 개수도 함께 봐야 한다.
        '''),
        code('''
        group_summary = (
            df.groupby("업종")["월매출"]
              .agg(개수="count", 평균="mean", 중앙값="median", 표준편차="std")
              .sort_values("평균", ascending=False)
        )
        group_summary["변동계수"] = group_summary["표준편차"] / group_summary["평균"]
        display(group_summary.round(2))
        '''),
        md('''
        ## 5. 독립 연습

        1. 월임대료의 평균, 표본분산, 표본표준편차를 NumPy의 mean, var, std 없이 계산한다.
        2. 유동인구 z점수와 경쟁점포수 z점수를 계산한다.
        3. 업종 안에서 계산한 z점수와 전체에서 계산한 z점수가 가장 크게 다른 행 5개를 찾는다.
        4. 어떤 기준의 z점수가 입지 판단에 더 적절한지 설명한다.
        '''),
        code('''
        # TODO 1
        rent = df["월임대료"].to_numpy()
        manual_mean = None
        manual_variance = None
        manual_std = None

        # TODO 2~3
        df["유동인구_z_전체"] = None
        df["유동인구_z_업종"] = None
        largest_difference = None
        display(largest_difference)
        '''),
        md('''
        ## 6. 자가점검

        - [ ] 분산의 단위와 표준편차의 단위를 구분한다.
        - [ ] ddof=0과 ddof=1을 코드로 비교했다.
        - [ ] 평균·중앙값 차이를 분포와 연결했다.
        - [ ] 전체 z점수와 집단 내 z점수의 질문이 다름을 설명한다.
        '''),
    ]


def lab03() -> list[dict]:
    return [
        md(r'''
        # Lab 03. 표본분포, 표준오차와 부트스트랩

        표본통계량은 표본에 따라 달라진다. 표본평균의 표준오차는 큰 표본에서 대략 다음과 같다.

        $$SE(\bar{x})=\frac{s}{\sqrt{n}}$$

        부트스트랩은 관측 표본에서 복원추출을 반복해 통계량의 표본분포를 근사한다.
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from src.education.personalized_data import make_student_dataset, student_seed

        df = make_student_dataset(STUDENT_ID).drop_duplicates()
        sales = df["월매출"].dropna().to_numpy()
        rng = np.random.default_rng(student_seed(STUDENT_ID))
        print("관측 수:", len(sales), "평균:", sales.mean())
        '''),
        md('''
        ## 1. 완성 예제: 표본 크기와 표준오차

        현재 데이터를 모집단처럼 두고 서로 다른 크기의 표본을 1,000번 추출한다.
        '''),
        code('''
        records = []
        for n in [10, 20, 50, 100]:
            sample_means = [
                rng.choice(sales, size=n, replace=True).mean()
                for _ in range(1000)
            ]
            records.append({
                "n": n,
                "평균의평균": np.mean(sample_means),
                "시뮬레이션_SE": np.std(sample_means, ddof=1),
                "공식_SE": np.std(sales, ddof=1) / np.sqrt(n),
            })
        se_table = pd.DataFrame(records)
        display(se_table.round(3))
        '''),
        md('''
        표본 크기가 4배가 될 때 표준오차가 약 절반이 되는지 수치로 확인한다. 시뮬레이션과 공식은 반복 횟수와 분포 때문에 완전히 같지는 않다.
        '''),
        md('''
        ## 2. 완성 예제: 평균의 부트스트랩 신뢰구간
        '''),
        code('''
        bootstrap_means = np.array([
            rng.choice(sales, size=len(sales), replace=True).mean()
            for _ in range(3000)
        ])
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        print(f"월매출 평균={sales.mean():.1f}, 95% bootstrap CI=({ci_low:.1f}, {ci_high:.1f})")

        plt.figure(figsize=(8, 4))
        plt.hist(bootstrap_means, bins=35, edgecolor="white")
        plt.axvline(ci_low, color="red", linestyle="--")
        plt.axvline(ci_high, color="red", linestyle="--")
        plt.xlabel("부트스트랩 표본평균")
        plt.ylabel("빈도")
        plt.title("월매출 평균의 부트스트랩 분포")
        plt.show()
        '''),
        md('''
        ## 3. 신뢰구간 해석

        “모평균이 이 특정 구간에 있을 확률이 95%”라고 해석하지 않는다. 같은 절차로 표본추출과 구간 계산을 반복하면 장기적으로 약 95%의 구간이 모평균을 포함한다는 뜻이다.
        '''),
        md('''
        ## 4. 따라하기: 업종별 평균 신뢰구간

        각 업종에서 별도로 부트스트랩을 수행한다. 업종별 표본 수가 다르면 구간 폭도 달라질 수 있다.
        '''),
        code('''
        rows = []
        for category, group in df.groupby("업종"):
            values = group["월매출"].dropna().to_numpy()
            means = [
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(2000)
            ]
            low, high = np.percentile(means, [2.5, 97.5])
            rows.append({"업종": category, "n": len(values), "평균": values.mean(), "하한": low, "상한": high})
        display(pd.DataFrame(rows).round(1))
        '''),
        md('''
        ## 5. 독립 연습

        1. 평균 대신 중앙값의 95% 부트스트랩 신뢰구간을 계산한다.
        2. 반복 횟수 200, 1,000, 5,000에서 구간 끝값이 얼마나 변하는지 비교한다.
        3. 이상값 포함·제외 시 평균과 중앙값 구간을 비교한다.
        4. 부트스트랩으로 해결할 수 없는 표본 편향의 예를 상권 데이터에서 제시한다.
        '''),
        code('''
        # TODO: 반복 횟수별 중앙값 신뢰구간 표
        median_ci_table = None
        display(median_ci_table)
        '''),
        md('''
        ## 6. 자가점검

        - [ ] 복원추출과 비복원추출의 차이를 설명한다.
        - [ ] n과 표준오차의 제곱근 관계를 확인했다.
        - [ ] 신뢰구간을 확률 문장으로 잘못 해석하지 않는다.
        - [ ] 부트스트랩이 대표성 편향을 고치지 못함을 설명한다.
        '''),
    ]


def lab04() -> list[dict]:
    return [
        md(r'''
        # Lab 04. 가설검정, 효과크기와 상관

        이 Notebook은 두 집단 평균 차이와 두 수치 변수 관계를 분석한다.
        p값만 보고 결론을 내리지 않고 원 단위 차이, 효과크기, 가정과 한계를 함께 보고한다.

        Pearson 상관계수:

        $$r=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}
        {\sqrt{\sum(x_i-\bar{x})^2\sum(y_i-\bar{y})^2}}$$
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy import stats
        from src.education.personalized_data import make_student_dataset

        df = make_student_dataset(STUDENT_ID).drop_duplicates()
        for column in ["유동인구", "월임대료"]:
            df[column] = df.groupby("업종")[column].transform(lambda s: s.fillna(s.median()))
        display(df.groupby("업종")["월매출"].agg(["count", "mean", "median", "std"]).round(1))
        '''),
        md('''
        ## 1. 분석 질문과 가설

        질문: 개인 데이터에서 카페와 음식점의 평균 월매출이 다른가?

        - H0: 카페와 음식점의 모집단 평균 월매출은 같다.
        - H1: 두 모집단 평균 월매출은 다르다.
        - 유의수준: alpha=0.05

        두 집단 분산이 같다고 가정하지 않는 Welch t검정을 사용한다.
        '''),
        code('''
        cafe = df.loc[df["업종"] == "카페", "월매출"].dropna()
        restaurant = df.loc[df["업종"] == "음식점", "월매출"].dropna()
        t_stat, p_value = stats.ttest_ind(cafe, restaurant, equal_var=False)
        mean_difference = cafe.mean() - restaurant.mean()

        print("카페 n, 평균:", len(cafe), cafe.mean())
        print("음식점 n, 평균:", len(restaurant), restaurant.mean())
        print("평균차(카페-음식점):", mean_difference)
        print("Welch t, p:", t_stat, p_value)
        '''),
        md('''
        p<0.05이면 H0를 기각하지만 “H0가 틀릴 확률이 95%”라고 말하지 않는다.
        p>=0.05도 두 집단이 같다는 증명이 아니라 현재 표본에서 차이를 확인할 증거가 충분하지 않다는 뜻이다.
        '''),
        md(r'''
        ## 2. 효과크기 Cohen d

        $$d=\frac{\bar{x}_1-\bar{x}_2}{s_{pooled}}$$

        효과크기는 차이를 표준편차 단위로 표현한다. 분야 맥락과 원 단위 차이를 함께 해석한다.
        '''),
        code('''
        def cohens_d(a, b):
            a, b = np.asarray(a), np.asarray(b)
            pooled_var = (
                (len(a)-1) * a.var(ddof=1) + (len(b)-1) * b.var(ddof=1)
            ) / (len(a) + len(b) - 2)
            return (a.mean() - b.mean()) / np.sqrt(pooled_var)

        d = cohens_d(cafe, restaurant)
        print("Cohen d:", d)
        '''),
        md('''
        ## 3. 완성 예제: 상관계수를 공식과 라이브러리로 비교

        산점도를 먼저 확인한다. 이상값, 비선형 패턴과 업종별 집단이 섞인 구조를 살핀다.
        '''),
        code('''
        x = df["유동인구"].to_numpy()
        y = df["월매출"].to_numpy()
        numerator = ((x - x.mean()) * (y - y.mean())).sum()
        denominator = np.sqrt(((x-x.mean())**2).sum() * ((y-y.mean())**2).sum())
        r_manual = numerator / denominator
        r_pandas = df["유동인구"].corr(df["월매출"])

        print("직접 계산 r:", r_manual, "pandas r:", r_pandas)
        plt.figure(figsize=(7, 4))
        for category, group in df.groupby("업종"):
            plt.scatter(group["유동인구"], group["월매출"], alpha=.65, label=category)
        plt.xlabel("유동인구")
        plt.ylabel("월매출")
        plt.legend()
        plt.title("업종별 유동인구와 월매출")
        plt.show()
        '''),
        md('''
        ## 4. 전체 상관과 집단 내 상관

        전체 관계가 집단별 관계와 다를 수 있다. 집단 구성이 만든 관계인지 확인한다.
        '''),
        code('''
        correlations = (
            df.groupby("업종")
              .apply(lambda g: g["유동인구"].corr(g["월매출"]), include_groups=False)
              .rename("업종내_r")
        )
        print("전체 r:", r_pandas)
        display(correlations)
        '''),
        md('''
        ## 5. 독립 연습

        1. 주차장수가 중앙값 이상인 집단과 미만인 집단의 월매출을 비교한다.
        2. 집단 정의, H0·H1, 검정 선택 이유, 표본 수, 평균차, p값, Cohen d를 표로 제시한다.
        3. 월임대료–월매출의 Pearson r과 Spearman rho를 비교한다.
        4. 관계에 영향을 줄 수 있는 제3의 변수 두 개를 제안한다.
        '''),
        code('''
        # TODO: 주차장 두 집단 비교
        parking_cut = df["주차장수"].median()
        high_parking = None
        low_parking = None
        test_result = None
        effect_size = None

        # TODO: Pearson과 Spearman 비교
        correlation_comparison = None
        display(correlation_comparison)
        '''),
        md('''
        ## 6. 자가점검

        - [ ] H0와 H1을 분석 전에 작성했다.
        - [ ] p값을 귀무가설이 참일 확률로 해석하지 않는다.
        - [ ] 효과크기와 원 단위 평균차를 함께 보고했다.
        - [ ] 상관관계를 인과관계로 표현하지 않았다.
        - [ ] 전체 관계와 집단 내 관계를 비교했다.
        '''),
    ]


def lab05() -> list[dict]:
    return [
        md(r'''
        # Lab 05. 선형회귀와 예측 오차

        선형회귀는 다음 형태의 예측식을 학습한다.

        $$\hat y=\beta_0+\beta_1x_1+\cdots+\beta_px_p$$

        최소제곱법은 잔차제곱합을 최소화한다.

        $$SSE=\sum_i(y_i-\hat y_i)^2$$

        이 Notebook에서는 단순회귀 기울기를 직접 계산한 뒤 다중회귀와 테스트 데이터 평가로 확장한다.
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from src.education.personalized_data import make_student_dataset, student_seed

        df = make_student_dataset(STUDENT_ID).drop_duplicates()
        for column in ["유동인구", "월임대료"]:
            df[column] = df[column].fillna(df[column].median())
        '''),
        md(r'''
        ## 1. 완성 예제: 단순회귀 계수 직접 계산

        단순회귀 기울기와 절편은 다음과 같다.

        $$\beta_1=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2},
        \qquad \beta_0=\bar y-\beta_1\bar x$$
        '''),
        code('''
        x = df["유동인구"].to_numpy()
        y = df["월매출"].to_numpy()
        beta1 = ((x-x.mean()) * (y-y.mean())).sum() / ((x-x.mean())**2).sum()
        beta0 = y.mean() - beta1 * x.mean()
        manual_pred = beta0 + beta1 * x

        simple = LinearRegression().fit(x.reshape(-1, 1), y)
        print("직접 계산:", beta0, beta1)
        print("sklearn:", simple.intercept_, simple.coef_[0])
        assert np.isclose(beta1, simple.coef_[0])
        '''),
        md('''
        기울기는 유동인구가 1단위 증가할 때 예측 월매출이 평균적으로 얼마나 달라지는지를 뜻한다.
        관찰 연구이므로 유동인구 증가가 매출 증가의 원인이라고 단정할 수 없다.
        '''),
        md('''
        ## 2. 학습·테스트 분할

        테스트 데이터는 모델 선택과 학습에 사용하지 않는다. random_state를 학번 시드로 고정하면 재현할 수 있다.
        '''),
        code('''
        features = ["유동인구", "경쟁점포수", "주차장수", "월임대료"]
        X = df[features]
        y = df["월매출"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=student_seed(STUDENT_ID)
        )
        print("학습:", X_train.shape, "테스트:", X_test.shape)
        '''),
        md('''
        ## 3. 다중회귀와 평가

        MAE는 평균 절대오차, RMSE는 큰 오차에 더 민감한 제곱 기반 오차다. R²는 평균 예측과 비교한 설명 비율이다.
        '''),
        code('''
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** .5
        r2 = r2_score(y_test, pred)
        coefficients = pd.Series(model.coef_, index=features, name="회귀계수")
        display(coefficients.to_frame())
        print(f"MAE={mae:.1f}, RMSE={rmse:.1f}, R2={r2:.3f}")
        '''),
        md('''
        ## 4. 기준모델과 비교

        복잡한 모델이 의미 있으려면 최소한 학습 데이터 평균만 예측하는 기준모델보다 좋아야 한다.
        '''),
        code('''
        baseline_pred = np.repeat(y_train.mean(), len(y_test))
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_rmse = mean_squared_error(y_test, baseline_pred) ** .5
        print("기준 MAE/RMSE:", baseline_mae, baseline_rmse)
        print("회귀 MAE/RMSE:", mae, rmse)
        '''),
        md('''
        ## 5. 잔차 분석

        잔차는 실제값-예측값이다. 예측값에 따른 곡선 패턴이나 깔때기 모양은 선형성·등분산 가정을 의심하게 한다.
        '''),
        code('''
        residual = y_test.to_numpy() - pred
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(y_test, pred, alpha=.7)
        limits = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        axes[0].plot(limits, limits, "--", color="red")
        axes[0].set(xlabel="실제 월매출", ylabel="예측 월매출", title="실제값과 예측값")

        axes[1].scatter(pred, residual, alpha=.7)
        axes[1].axhline(0, linestyle="--", color="red")
        axes[1].set(xlabel="예측 월매출", ylabel="잔차", title="잔차 진단")
        plt.tight_layout()
        plt.show()
        '''),
        md('''
        ## 6. 독립 연습

        1. 업종을 one-hot encoding하여 모델에 추가하고 테스트 성능을 비교한다.
        2. 이상 후보를 포함한 모델과 제외한 모델의 MAE·RMSE를 비교한다.
        3. 학습·테스트 분할 시드를 세 개 바꾸어 성능 변동을 확인한다.
        4. 계수의 단위와 부호를 설명하고 상식과 다른 계수의 가능한 원인을 제안한다.
        '''),
        code('''
        # TODO: get_dummies와 ColumnTransformer 중 하나를 이용해 업종 포함 모델 작성
        extended_result = None
        display(extended_result)

        # TODO: 세 random_state의 평가표
        stability_table = None
        display(stability_table)
        '''),
        md('''
        ## 7. 자가점검

        - [ ] 단순회귀 기울기를 공식으로 계산했다.
        - [ ] 테스트 데이터는 학습에 사용하지 않았다.
        - [ ] 기준모델과 MAE·RMSE를 비교했다.
        - [ ] 잔차 그래프에서 패턴과 이상값을 확인했다.
        - [ ] 예측 관계를 인과관계로 표현하지 않았다.
        '''),
    ]


def lab06() -> list[dict]:
    return [
        md(r'''
        # Lab 06. K-means 군집과 PCA

        K-means는 군집 내부 제곱거리 합을 최소화한다.

        $$J=\sum_{k=1}^{K}\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2$$

        거리 기반 모델이므로 표준화 전후 결과를 반드시 비교한다. PCA는 고차원 데이터를 시각화하지만 정보 손실이 있다.
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        from src.education.personalized_data import make_student_dataset, student_seed

        df = make_student_dataset(STUDENT_ID).drop_duplicates()
        columns = ["유동인구", "경쟁점포수", "주차장수", "월임대료", "월매출"]
        X = df[columns].copy().fillna(df[columns].median())
        display(X.describe().T[["mean", "std", "min", "max"]].round(1))
        '''),
        md('''
        ## 1. 표준화

        StandardScaler는 학습 데이터의 평균을 빼고 표준편차로 나눈다. 변환 후 각 변수 평균은 약 0, 표준편차는 약 1이다.
        '''),
        code('''
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaled_summary = pd.DataFrame(X_scaled, columns=columns).agg(["mean", "std"]).T
        display(scaled_summary.round(3))
        '''),
        md('''
        ## 2. k 후보 비교

        inertia는 k가 증가하면 항상 감소하므로 단독으로 최적 k를 정할 수 없다.
        silhouette는 -1에서 1 사이이며 같은 군집에는 가깝고 다른 군집에는 멀수록 높다.
        '''),
        code('''
        records = []
        seed = student_seed(STUDENT_ID)
        for k in range(2, 7):
            model = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels = model.fit_predict(X_scaled)
            records.append({
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            })
        scores = pd.DataFrame(records)
        display(scores.round(3))

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(scores["k"], scores["inertia"], marker="o", label="inertia")
        ax1.set_xlabel("k"); ax1.set_ylabel("inertia")
        ax2 = ax1.twinx()
        ax2.plot(scores["k"], scores["silhouette"], marker="s", color="orange")
        ax2.set_ylabel("silhouette")
        plt.title("군집 수 후보 비교")
        plt.show()
        '''),
        md('''
        ## 3. 군집 프로필

        아래 BEST_K는 예시다. 본인 점수표와 분석 목적을 근거로 바꾼다. 군집 번호에는 순서 의미가 없다.
        '''),
        code('''
        BEST_K = int(scores.loc[scores["silhouette"].idxmax(), "k"])
        kmeans = KMeans(n_clusters=BEST_K, random_state=seed, n_init=20)
        df["군집"] = kmeans.fit_predict(X_scaled)
        profile = df.groupby("군집")[columns].agg(["count", "mean"])
        print("선택 k:", BEST_K)
        display(profile.round(1))
        '''),
        md('''
        ## 4. PCA 2차원 시각화와 적재량

        설명분산비율은 각 주성분이 전체 표준화 분산의 얼마를 보존하는지 나타낸다.
        적재량은 원래 변수가 주성분 축에 얼마나 기여하는지 보여준다.
        '''),
        code('''
        pca = PCA(n_components=2)
        points = pca.fit_transform(X_scaled)
        loadings = pd.DataFrame(pca.components_.T, index=columns, columns=["PC1", "PC2"])
        print("설명분산비율:", pca.explained_variance_ratio_)
        display(loadings.round(3))

        plt.figure(figsize=(7, 5))
        plt.scatter(points[:, 0], points[:, 1], c=df["군집"], cmap="tab10", alpha=.7)
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("PCA 투영과 K-means 군집")
        plt.show()
        '''),
        md('''
        ## 5. 표준화하지 않은 결과와 비교

        원자료 K-means는 값의 규모가 큰 유동인구가 거리를 지배할 가능성이 높다.
        '''),
        code('''
        raw_model = KMeans(n_clusters=BEST_K, random_state=seed, n_init=20)
        raw_labels = raw_model.fit_predict(X)
        agreement = pd.crosstab(df["군집"], raw_labels, rownames=["표준화 군집"], colnames=["원자료 군집"])
        display(agreement)
        '''),
        md('''
        ## 6. 독립 연습

        1. k를 선택한 근거를 inertia, silhouette와 해석 가능성으로 작성한다.
        2. 군집별 평균을 전체 평균 대비 z점수로 표현한다.
        3. 각 군집에 “고유동·고비용”처럼 데이터 기반 이름을 붙인다.
        4. 변수 하나를 제외하고 군집이 얼마나 바뀌는지 비교한다.
        5. PCA 두 축의 누적 설명분산이 충분한지 판단하고 2차원 그림의 한계를 설명한다.
        '''),
        code('''
        # TODO: 전체 평균 대비 군집 프로필 z점수
        cluster_z_profile = None
        display(cluster_z_profile)

        # TODO: 변수 제거 민감도 표
        sensitivity = None
        display(sensitivity)
        '''),
        md('''
        ## 7. 자가점검

        - [ ] 표준화 전후 결과를 비교했다.
        - [ ] k를 한 지표만으로 선택하지 않았다.
        - [ ] 군집 번호를 서열로 해석하지 않았다.
        - [ ] 군집 이름의 근거 변수를 제시했다.
        - [ ] PCA 설명분산비율과 적재량을 확인했다.
        '''),
    ]


def lab07() -> list[dict]:
    return [
        md('''
        # Lab 07. 개인별 데이터 분석 종합 실기

        이 실기는 팀 프로젝트와 별도로 개인의 데이터 처리·통계·모델링·설명 능력을 평가한다.
        권장 시간은 독립 분석 120분, 교수자 앞 코드 설명·수정 8분이다.

        제출 규칙:

        1. STUDENT_ID와 이름을 본인 정보로 바꾼다.
        2. 모든 셀을 위에서 아래로 실행한다.
        3. 최종 숫자뿐 아니라 중간 표와 검증 결과를 남긴다.
        4. 모든 그래프 아래에 관찰, 해석, 한계를 구분해 작성한다.
        5. 외부 도구를 사용했다면 사용 위치와 검증 방법을 기록한다.
        '''),
        code(SETUP),
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy import stats
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from src.education.personalized_data import make_student_dataset, student_seed

        NAME = "이름입력"
        raw = make_student_dataset(STUDENT_ID)
        print(NAME, STUDENT_ID, raw.shape)
        display(raw.sample(5, random_state=student_seed(STUDENT_ID)))
        '''),
        md('''
        ## 문제 1. 데이터 이해와 품질 진단(15점)

        한 행과 각 열의 의미를 설명한다. 자료형, 결측, 중복, 범위와 IQR 이상 후보를 하나의 품질표로 제시한다.

        확인 기준: 원본을 변경하지 않고 문제 행 번호와 건수를 재현할 수 있어야 한다.
        '''),
        code('''
        # TODO 답안 1
        quality_report = None
        outlier_rows = None
        display(quality_report)
        display(outlier_rows)
        '''),
        md('''
        ## 문제 2. 전처리 대안 비교(15점)

        결측 처리 방법 두 개와 이상값 처리 방법 두 개를 비교한다.
        각 조합의 행 수, 월매출 평균·중앙값·표준편차, 유동인구–월매출 상관계수를 표로 만든다.
        최종 방법을 선택하고 데이터 생성 과정과 분석 목적을 근거로 설명한다.
        '''),
        code('''
        # TODO 답안 2
        preprocessing_comparison = None
        clean = None
        display(preprocessing_comparison)
        assert clean is not None
        '''),
        md('''
        ## 문제 3. 수학과 기술통계(15점)

        정제 데이터 월매출 첫 5개 값의 평균과 표본분산을 라이브러리 없이 계산하고 pandas 결과와 비교한다.
        업종별 개수, 평균, 중앙값, 표준편차와 변동계수 표를 작성한다.
        '''),
        code('''
        # TODO 답안 3
        first5 = None
        manual_mean = None
        manual_variance = None
        group_summary = None
        display(group_summary)
        '''),
        md('''
        ## 문제 4. 통계적 추론(20점)

        다음 중 하나를 선택한다.

        - 두 업종 평균 비교: 가설, Welch t검정, 원 단위 차이, Cohen d
        - 한 통계량 추정: 부트스트랩 95% 신뢰구간
        - 두 변수 관계: 산점도, Pearson·Spearman, 제3 변수

        선택 이유, 가정, 결과와 잘못 해석하면 안 되는 내용을 작성한다.
        '''),
        code('''
        # 분석 질문:
        # 선택 방법과 이유:
        # TODO 답안 4
        inference_result = None
        display(inference_result)
        '''),
        md('''
        ## 문제 5. 모델링과 검증(20점)

        회귀 또는 군집을 선택한다.

        회귀: 기준모델, 학습·테스트 분리, MAE·RMSE·R², 잔차 분석이 필요하다.
        군집: 표준화, k=2~6 비교, inertia·silhouette, 군집 프로필과 민감도 분석이 필요하다.
        '''),
        code('''
        # 선택 모델:
        # TODO 답안 5
        model_evaluation = None
        display(model_evaluation)
        '''),
        md('''
        ## 문제 6. 시각화와 의사결정(15점)

        서로 다른 목적의 그래프 두 개를 만든다. 제목, 축 단위, 범례와 데이터 범위를 표시한다.

        각 그래프 아래에 작성:

        - 관찰: 데이터에서 직접 읽을 수 있는 사실
        - 해석: 가능한 의미
        - 한계: 데이터로 말할 수 없는 것
        - 의사결정: 가능한 결정과 불가능한 결정
        '''),
        code('''
        # TODO 답안 6: 그래프 두 개
        '''),
        md('''
        ## 현장 확인 준비

        교수자는 다음 중 일부를 요청한다.

        - IQR 계수를 1.5에서 3.0으로 바꾸고 결과 설명
        - 평균 대체를 업종별 중앙값 대체로 변경
        - 비교 집단 또는 분석 변수를 즉석 변경
        - 회귀 특성 하나 제거 또는 군집 k 변경
        - 오류가 있는 한 셀을 수정하고 재실행

        완성 코드를 외웠는지가 아니라 데이터 흐름을 이해하고 새로운 조건에 맞게 수정하는지를 평가한다.
        '''),
        md('''
        ## 최종 제출 체크

        - [ ] Restart & Run All이 오류 없이 끝난다.
        - [ ] 학번과 이름이 정확하다.
        - [ ] 원본, 정제, 분석 데이터를 구분했다.
        - [ ] 수치의 단위와 표본 수를 표시했다.
        - [ ] 처리 대안과 모델 기준선을 비교했다.
        - [ ] 관찰·해석·한계를 구분했다.
        - [ ] 사용한 외부 도구와 검증 방법을 기록했다.
        - [ ] 구두 확인에서 임의 셀을 설명하고 수정할 수 있다.
        '''),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save("00_math_numpy.ipynb", lab00())
    save("01_data_quality.ipynb", lab01())
    save("02_descriptive_statistics.ipynb", lab02())
    save("03_sampling_confidence_interval.ipynb", lab03())
    # Lab 04~07은 아래 별도 빌더에서 정의한다.
    for name, builder in [
        ("04_hypothesis_correlation.ipynb", lab04),
        ("05_regression.ipynb", lab05),
        ("06_clustering_pca.ipynb", lab06),
        ("07_practical_assessment.ipynb", lab07),
    ]:
        save(name, builder())
    print(f"{OUT}에 Notebook 8개 생성 완료")


if __name__ == "__main__":
    main()

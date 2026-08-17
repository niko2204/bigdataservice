# Big Data Service

국립목포대학교 컴퓨터학부 4학년 프로젝트 중심 교과목 **빅데이터서비스** 공개 저장소입니다.

## 프로젝트

**지역 생활인구·상권 빅데이터 기반 창업 입지 추천 서비스**를 개발합니다. 목포시 교육용 샘플 데이터를 사용해 데이터 수집, 전처리, 탐색적 분석, 공간 분석, 추천 모델, 데이터베이스, Streamlit 서비스까지 완성합니다.

## 학습 구성

| Course | 내용 | Jupyter 강의노트 | 핵심 산출물 |
|---:|---|---|---|
| 01 | 프로젝트 이해와 환경 구축 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course01_project_setup/lecture_note.ipynb) | 문제 정의서 |
| 02 | Python 데이터 분석 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course02_python_data_analysis/lecture_note.ipynb) | 분석 노트북 |
| 03 | 공공데이터 API | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course03_public_data_api/lecture_note.ipynb) | 수집 코드 |
| 04 | 데이터 전처리·통합 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course04_data_preprocessing/lecture_note.ipynb) | 통합 데이터셋 |
| 05 | 탐색적 분석·시각화 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course05_eda_visualization/lecture_note.ipynb) | EDA 보고서 |
| 06 | 지도·공간 분석 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course06_geospatial_analysis/lecture_note.ipynb) | 상권 지도 |
| 07 | 입지 평가·추천 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course07_recommendation_model/lecture_note.ipynb) | 추천 모델 |
| 08 | 데이터베이스 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course08_database/lecture_note.ipynb) | SQLite DB |
| 09 | Streamlit 서비스 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course09_streamlit_service/lecture_note.ipynb) | 웹 서비스 |
| 10 | 통합·발표 | [Colab에서 열기](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/archive/original-bigdata-service/courses/course10_final_project/lecture_note.ipynb) | 최종 결과물 |

Course는 주차가 아닙니다. 필요한 시점에 학습하고, 나머지 수업은 팀 프로젝트·상담·코드 리뷰로 운영합니다.

## 바로 실행하기

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/Home.py
```

API 키 없이 `data/sample/`의 교육용 데이터를 이용해 모든 화면을 실행할 수 있습니다.

## 폴더 안내

- `courses/`: 10개 Course 강의·실습·과제·퀴즈
- `notebooks/`: Google Colab/Jupyter 단계별 실습
- `notebooks/data_analysis/`: 이론·수학·개인별 데이터 분석 실습 8종
- `src/`: 수집, 전처리, 분석, 추천 코드
- `app/`: Streamlit 다중 페이지 서비스
- `data/`: 샘플·원본·중간·완료 데이터
- `database/`: SQLite 스키마와 적재 코드
- `assignments/`, `evaluation/`, `templates/`: 프로젝트 운영 자료
- `tests/`: 핵심 로직 자동 테스트

## 담당 교수

- 이영호 교수
- 국립목포대학교 컴퓨터학부
- youngho@ce.mokpo.ac.kr

데이터의 수치와 추천 결과는 교육용이며 실제 창업 의사결정의 근거로 단독 사용하지 않습니다.

## 데이터 분석 실기

프로젝트만 제출하는 방식의 한계를 보완하기 위해 학번 기반 개인별 데이터, 손계산, 통계 검증, 코드 수정 시연과 구두 확인을 포함합니다. 자세한 운영 방법은 `docs/data_analysis_curriculum.md`와 `evaluation/data_analysis_practical_rubric.md`를 참고하세요.

## 학생 학습 방법

위 표의 **Colab에서 열기**를 누르면 해당 Course의 Jupyter Notebook이 바로 실행됩니다. 같은 내용의 `lecture_note.md`는 GitHub 열람과 Notebook 재생성을 위한 원본입니다. 전체 학습 경로와 오류 해결 방법은 [학생 자율학습 가이드](docs/student_learning_guide.md)를 참고하세요.

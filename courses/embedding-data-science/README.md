# 임베딩 기반 데이터 과학 — 한국어 10주 강의자료

이 디렉터리는 [https://github.com/HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)의 공개 예제를 참고하여
국립목포대학교 컴퓨터학부 4학년 수업용으로 재구성한 한국어 강의 패키지입니다.
원본 12장을 순서대로 번역하지 않고, 임베딩을 중심으로 수학·데이터 분석·검색 평가·RAG·
멀티모달 서비스를 하나의 10주 과정으로 다시 설계했습니다.

## 구성

- `SYLLABUS.md`: 10주 강의계획 및 평가
- `lectures/`: 주차별 한국어 강의노트
- `slides/`: 용어·정의·수식·논문 도판·최신 동향을 포함한 Marp 강의 덱 10개
- `notebooks/`: 학생이 실행하고 기록하는 Jupyter Notebook 10개
- `assignments/`: 명확한 명세와 루브릭을 유지하면서 초보자용 단계·예시 코드를 제공하는
  개인 과제 2개와 최종 프로젝트
- `exams/`: 이론 중심 중간·기말고사와 교수자용 모범답안·부분점수 기준
- `INSTRUCTOR_GUIDE.md`: 교수자 운영·시연·구술평가 가이드
- `ASSESSMENT_RUBRIC.md`: 공통 채점 루브릭
- `data/policy_documents.csv`: 작은 한국어 예제 데이터
- `SOURCE_AND_LICENSE.md`: 원본, 기준 커밋, 변경 및 라이선스 기록

## 주차별 바로가기

PR을 `main`에 병합한 뒤 아래 Colab 링크를 누르면 별도 설치 없이 Notebook을 열 수 있습니다.

|주차|주제|Marp 슬라이드|강의노트|Jupyter Notebook|Colab|
|---:|---|---|---|---|---|
|1|벡터 표현과 임베딩|[슬라이드](slides/week01_embedding_foundations.md)|[노트](lectures/week01.md)|[Notebook](notebooks/week01.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week01.ipynb)|
|2|토큰화와 TF-IDF|[슬라이드](slides/week02_tokenization_sparse.md)|[노트](lectures/week02.md)|[Notebook](notebooks/week02.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week02.ipynb)|
|3|단어 임베딩|[슬라이드](slides/week03_word_embeddings.md)|[노트](lectures/week03.md)|[Notebook](notebooks/week03.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week03.ipynb)|
|4|문맥 임베딩과 Transformer|[슬라이드](slides/week04_contextual_transformer.md)|[노트](lectures/week04.md)|[Notebook](notebooks/week04.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week04.ipynb)|
|5|문장·문서 임베딩|[슬라이드](slides/week05_sentence_embeddings.md)|[노트](lectures/week05.md)|[Notebook](notebooks/week05.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week05.ipynb)|
|6|분류·군집·시각화|[슬라이드](slides/week06_embedding_analytics.md)|[노트](lectures/week06.md)|[Notebook](notebooks/week06.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week06.ipynb)|
|7|임베딩 검색 평가|[슬라이드](slides/week07_evaluation.md)|[노트](lectures/week07.md)|[Notebook](notebooks/week07.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week07.ipynb)|
|8|벡터 검색과 RAG|[슬라이드](slides/week08_vector_search_rag.md)|[노트](lectures/week08.md)|[Notebook](notebooks/week08.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week08.ipynb)|
|9|멀티모달 임베딩|[슬라이드](slides/week09_multimodal.md)|[노트](lectures/week09.md)|[Notebook](notebooks/week09.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week09.ipynb)|
|10|종합 프로젝트|[슬라이드](slides/week10_capstone_trends.md)|[노트](lectures/week10.md)|[Notebook](notebooks/week10.ipynb)|[실행](https://colab.research.google.com/github/niko2204/bigdataservice/blob/main/courses/embedding-data-science/notebooks/week10.ipynb)|

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r courses/embedding-data-science/requirements.txt
jupyter lab courses/embedding-data-science/notebooks
```

각 Notebook은 기본 실습이 API 키 없이 실행되도록 작성했습니다. Transformer와 CLIP 모델
실습은 인터넷과 모델 다운로드가 필요한 선택 셀입니다. Sentence Transformers 셀도 주차별
연습에서는 선택할 수 있지만, 과제 1·2의 dense 검색에서는 필수입니다. 모델 다운로드가
어려우면 교수자가 제공한 저장 임베딩을 사용합니다. 모델 셀은 `RUN_*` 값을 `True`로 바꾸어
실행합니다.

## 과제와 시험

|구분|자료|핵심 평가|
|---|---|---|
|공통 과제 안내|[제출 구조·협업·AI·재현성 규칙](assignments/README.md)|명세 준수와 독립 검증|
|과제 1|[희소/밀집 검색의 통제 비교](assignments/assignment01.md)|qrels, 검색 지표, 오류 분석|
|과제 2|[벡터 검색과 근거 기반 응답](assignments/assignment02.md)|chunk 실험, 검색/생성 오류, 거부|
|최종 프로젝트|[임베딩 데이터 서비스](assignments/final_project.md)|기준선, 개선 실험, 서비스 방어|
|중간고사|[학생용 문제지](exams/midterm_exam.md)|1–5주차 정의·수식·비교|
|기말고사|[학생용 문제지](exams/final_exam.md)|6–10주차 평가·RAG·멀티모달·재현성|

교수자용 모범답안은 `exams/instructor/`에 있으므로 실제 시험 전 비공개 LMS 또는 교수자
전용 저장소로 옮기거나 수치와 사례를 변형해야 합니다.

과제 1은 문서 50개·질의 10개, 과제 2는 원문 10개·질문 12개를 최소 범위로 한다. 과제에서
nDCG 구현, 다중 청크 조합, 복수 ablation과 성능 신뢰구간은 선택 확장 활동이며 필수 채점
범위가 아니다. 단, nDCG의 정의와 작은 손계산은 7주차 수업과 기말고사 이론 범위에 포함된다.

## 교육 원칙

1. TF-IDF 또는 작은 행렬 계산을 항상 기준선으로 남긴다.
2. 모델 결과를 그래프로만 판단하지 않고 검색·분류 지표로 평가한다.
3. 성공 사례뿐 아니라 실패 사례와 편향을 분석한다.
4. 생성형 AI 사용을 허용하되 프롬프트와 검증 기록을 제출한다.
5. 최종 제출 시 코드 무작위 변경과 개별 구술평가를 실시한다.

## 라이선스

원본은 Apache License 2.0입니다. 이 파생 자료에도 원본 라이선스와 저작권 고지를
유지하며, 자세한 내용은 `SOURCE_AND_LICENSE.md`와 `LICENSE`를 확인하세요.

# 주차별 상세 강의노트 읽기 안내

## 권장 학습 순서

1. 강의노트의 학습목표와 핵심 질문을 먼저 읽습니다.
2. 연결된 슬라이드에서 전체 구조와 논문 도판을 확인합니다.
3. 강의노트의 수식 예제를 종이에 직접 계산합니다.
4. 연결된 Jupyter Notebook을 위에서부터 실행해 손계산과 출력이 일치하는지 확인합니다.
5. ‘흔한 오해’와 ‘복습 문제’에 자신의 말로 답합니다.
6. 이해가 부족한 항목은 참고문헌의 원 논문에서 그림·문제 정의·실험 조건을 찾아봅니다.

> 생성형 AI로 답을 작성했다면 결과를 그대로 받아들이지 말고, 수식·코드·원 논문 중 하나
> 이상으로 검증합니다. 검증하지 않은 설명은 학습 결과로 인정하지 않습니다.

## 주차별 자료

|주차|상세 강의노트|Marp 슬라이드|Jupyter Notebook|
|---:|---|---|---|
|1|[벡터 표현과 임베딩의 의미](week01.md)|[슬라이드](../slides/week01_embedding_foundations.md)|[Notebook](../notebooks/week01.ipynb)|
|2|[토큰화와 희소 문서 벡터](week02.md)|[슬라이드](../slides/week02_tokenization_sparse.md)|[Notebook](../notebooks/week02.ipynb)|
|3|[분포 가설과 단어 임베딩](week03.md)|[슬라이드](../slides/week03_word_embeddings.md)|[Notebook](../notebooks/week03.ipynb)|
|4|[문맥 임베딩과 트랜스포머](week04.md)|[슬라이드](../slides/week04_contextual_transformer.md)|[Notebook](../notebooks/week04.ipynb)|
|5|[문장·문서 임베딩과 의미 유사도](week05.md)|[슬라이드](../slides/week05_sentence_embeddings.md)|[Notebook](../notebooks/week05.ipynb)|
|6|[임베딩 기반 분류·군집·시각화](week06.md)|[슬라이드](../slides/week06_embedding_analytics.md)|[Notebook](../notebooks/week06.ipynb)|
|7|[임베딩과 검색 시스템 평가](week07.md)|[슬라이드](../slides/week07_evaluation.md)|[Notebook](../notebooks/week07.ipynb)|
|8|[벡터 검색과 RAG](week08.md)|[슬라이드](../slides/week08_vector_search_rag.md)|[Notebook](../notebooks/week08.ipynb)|
|9|[멀티모달 임베딩](week09.md)|[슬라이드](../slides/week09_multimodal.md)|[Notebook](../notebooks/week09.ipynb)|
|10|[임베딩 서비스 종합 프로젝트](week10.md)|[슬라이드](../slides/week10_capstone_trends.md)|[Notebook](../notebooks/week10.ipynb)|

## 강의노트의 공통 구조

- **학습목표와 핵심 질문:** 무엇을 설명할 수 있어야 하는지 제시합니다.
- **개념 설명:** 슬라이드의 압축된 문장을 배경과 예시를 포함해 풀이합니다.
- **수학적 이해:** 기호의 의미, 식의 가정, 작은 손계산을 연결합니다.
- **논문 도판 읽기:** 그림의 구성요소, 화살표, 비교 조건과 한계를 설명합니다.
- **Notebook 연결:** 어느 셀에서 무엇을 확인하고 어떤 값을 바꿀지 안내합니다.
- **실패 분석:** 평균 점수만으로 보이지 않는 오류와 데이터 누수를 다룹니다.
- **복습 문제와 참고문헌:** 이론 검증 문제와 원문 읽기 경로를 제공합니다.

## 논문 도판과 인용

강의노트의 논문 도판은 `../slides/assets/papers/`의 교육용 사본을 재사용합니다. 각 그림 아래에
저자·연도·논문명·그림 번호와 원문 링크를 표시하며, 그림을 단순 장식으로 사용하지 않고
학생이 무엇을 읽어야 하는지 본문에서 설명합니다. 도판의 저작권과 재배포 조건은 각 논문과
출판사의 라이선스를 따릅니다. 전체 출처표는 [강의용 논문 도판 출처](../slides/assets/papers/README.md)에서
확인할 수 있습니다.

## 용어 표기

한국어 수학·NLP·검색 용어는 [공통 용어표](../TERMINOLOGY.md)를 따릅니다. 코드 식별자,
모델명과 논문 제목은 원문 표기를 유지하고, 처음 등장하는 전문용어는 가능한 경우 한국어와
영어를 함께 씁니다. 예를 들어 `norm`은 처음에 **벡터의 크기(L2 노름)**로, `word
co-occurrence matrix`는 **단어 동시 출현 행렬**로 설명합니다.

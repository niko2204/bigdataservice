# 임베딩 데이터 과학 용어 표기 기준

이 문서는 슬라이드·강의노트·Notebook·과제·시험에서 사용하는 한국어 용어를 통일한다.
학생에게 보이는 설명은 한국어를 우선하고, 코드·파일명·모델명·평가 지표의 기호는 원문을
유지한다.

## 표기 원칙

1. 첫 등장에서는 `한국어 용어(영문 용어 또는 약어)`로 쓴다.
2. 이후 설명에서는 한국어 용어를 사용한다.
3. 코드 식별자, CSV 열 이름, 라이브러리 인수와 모델명은 변경하지 않는다.
4. 직역이 더 낯선 용어는 통용되는 외래어를 사용하되 쉬운 뜻을 함께 설명한다.
5. 일상어와 충돌하는 전문어는 뜻을 먼저 밝힌다. 예: `벡터의 크기(L2 노름, norm)`.

## 수학과 벡터

|권장 표기|영문·기호|설명|
|---|---|---|
|벡터의 크기(L2 노름)|L2 norm|처음에는 ‘크기’를 함께 쓰고 이후 `L2 노름`으로 쓴다.|
|내적|inner product, dot product|두 벡터의 대응 성분 곱을 더한 값|
|코사인 유사도|cosine similarity|벡터 방향의 유사성을 나타내는 값|
|유클리드 거리|Euclidean distance|두 점 사이의 직선거리|
|정규화|normalization|벡터 크기나 값의 범위를 일정한 규칙으로 맞추는 과정|
|표준화|standardization|평균과 표준편차 등을 이용해 수치 척도를 맞추는 과정|
|행렬 분해|matrix factorization|행렬을 여러 행렬의 곱으로 나타내는 과정|
|특이값 분해|singular value decomposition, SVD|행렬을 특이벡터와 특이값으로 분해하는 방법|
|직교|orthogonal|내적이 0인 관계|

## 텍스트 표현과 모델

|권장 표기|영문·약어|설명|
|---|---|---|
|임베딩|embedding|대상의 관계를 벡터 공간에 나타낸 표현 또는 변환 함수|
|희소 표현|sparse representation|대부분의 좌표가 0인 표현|
|밀집 표현|dense representation|대부분의 좌표가 값을 갖는 표현|
|말뭉치|corpus|분석이나 학습에 사용하는 텍스트 집합|
|단어 동시 출현 행렬|word co-occurrence matrix|정해진 문맥 범위에서 두 단어가 함께 나타난 횟수의 행렬|
|문맥 창|context window|중심 단어 주변에서 함께 살펴보는 범위|
|토큰화기|tokenizer|문자열을 토큰과 토큰 ID로 변환하는 도구|
|부분 단어|subword|단어보다 작은 토큰화 단위|
|미등록어|out-of-vocabulary, OOV|어휘집에 없는 입력 단위|
|은닉 상태|hidden state|신경망 내부에서 계산된 중간 표현|
|어텐션|attention|입력 위치별 관련성을 가중하여 정보를 모으는 연산|
|자기 어텐션|self-attention|같은 입력 안에서 질의·키·값을 구성하는 어텐션|
|마스크|mask|계산에 포함하거나 제외할 위치를 표시하는 값|
|채움 토큰|padding token|입력 길이를 맞추기 위해 추가하는 토큰|
|벡터 집계(풀링)|pooling|여러 토큰 벡터를 하나의 문장·문서 벡터로 합치는 연산|
|미세 조정|fine-tuning|사전학습 모델을 과업 데이터로 추가 학습하는 과정|

## 데이터와 실험

|권장 표기|영문|설명|
|---|---|---|
|특징|feature|모델 입력으로 사용하는 측정값이나 표현|
|정답 범주 또는 레이블|label|분류·평가에서 사용하는 정답 표시|
|범주|class|분류가 구분하는 집단|
|기준선|baseline|새 방법과 비교하기 위한 단순한 방법|
|학습 데이터|train data|모델 학습에 사용하는 데이터|
|개발 데이터|development data, dev|설정과 모델을 선택하는 데이터|
|최종 평가 데이터|test data|설정을 고정한 뒤 마지막 성능만 측정하는 데이터|
|데이터 분할|data split|데이터를 학습·개발·최종 평가 용도로 나누는 과정|
|난수 시드|random seed|무작위 연산을 재현하기 위한 초기값|
|임곗값|threshold|판정이 바뀌는 기준값|
|요소 제거 실험|ablation study|구성요소를 하나씩 제거해 효과를 확인하는 실험|
|데이터 누수|data leakage|평가에 사용될 정보가 학습이나 설정 선택에 들어가는 문제|
|평가 데이터 오염|benchmark contamination|평가 항목이 사전학습·미세 조정 데이터에 포함된 문제|
|분포 변화|data drift|시간에 따라 입력 데이터 분포가 달라지는 현상|
|개념 변화|concept drift|같은 입력과 정답 사이의 관계가 달라지는 현상|
|임베딩 공간 변화|embedding drift|모델 교체 등으로 벡터 공간이 달라지는 현상|
|상충 관계|trade-off|한 기준을 개선하면 다른 기준이 나빠질 수 있는 관계|

## 검색과 RAG

|권장 표기|영문·약어|설명|
|---|---|---|
|질의|query|사용자의 질문이나 검색어|
|검색기|retriever|질의와 관련된 문서나 항목을 찾는 모듈|
|색인|index|빠른 검색을 위해 미리 구성한 자료구조|
|정확 최근접 이웃 탐색|exact nearest-neighbor search|모든 후보를 비교해 최근접 이웃을 찾는 방법|
|근사 최근접 이웃 탐색|approximate nearest-neighbor search, ANN|일부 후보만 탐색해 속도를 높이는 방법|
|희소·밀집 혼합 검색|sparse–dense hybrid search|희소 검색과 밀집 검색을 결합하는 방법|
|재순위화|reranking|1차 검색 후보의 순서를 더 정밀한 모델로 다시 정하는 과정|
|문서 조각(청크)|chunk|검색과 생성을 위해 문서를 나눈 단위|
|검색 증강 생성|retrieval-augmented generation, RAG|검색한 근거를 언어 모델 입력에 추가하는 생성 방식|
|근거 일치성|groundedness|답변의 주장이 제시된 근거로 뒷받침되는 정도|
|응답 보류|abstention|근거가 부족할 때 답하지 않는 시스템 동작|
|근거 없는 생성(환각)|hallucination|입력 근거로 뒷받침되지 않는 내용을 생성하는 현상|
|지연시간|latency|요청부터 응답까지 걸린 시간|
|전체 시스템|end-to-end system|입력부터 최종 출력까지 연결된 모든 단계|

## 평가 지표

|권장 표기|영문·약어|설명|
|---|---|---|
|정확도|accuracy|전체 예측 중 맞은 예측의 비율|
|정밀도|precision|양성으로 예측한 항목 중 실제 양성의 비율|
|재현율|recall|실제 양성 또는 관련 항목 중 찾아낸 비율|
|상위 k개 정밀도|Precision@k, P@k|상위 k개 검색 결과 중 관련 문서의 비율|
|상위 k개 재현율|Recall@k, R@k|전체 관련 문서 중 상위 k개에서 찾은 비율|
|역순위|reciprocal rank, RR|첫 관련 문서 순위의 역수|
|평균 역순위|mean reciprocal rank, MRR|여러 질의의 역순위를 평균한 값|
|할인 누적 이득|discounted cumulative gain, DCG|낮은 순위의 관련성 이득을 감쇠해 더한 값|
|정규화 할인 누적 이득|normalized DCG, nDCG|DCG를 이상적인 순위의 DCG로 나눈 값|
|실루엣 계수|silhouette score|군집 내부 응집도와 군집 간 분리도를 함께 나타내는 값|
|군집 중심|centroid|군집을 대표하는 중심 벡터|

## 그대로 유지하는 이름과 기호

다음은 모델·방법·데이터셋의 고유명 또는 널리 쓰이는 기호이므로 원문을 유지하되 첫
등장에 설명한다.

- 모델·방법: Transformer, BERT, Word2Vec, GloVe, fastText, SBERT, CLIP, ColBERT,
  BERTopic, FAISS, HNSW
- 수학·평가 기호: TF–IDF, PMI, PPMI, SVD, PCA, MRR, DCG, nDCG, ANN, RRF
- 데이터와 과업: MTEB, MMTEB, STS, NLI, qrels
- 코드 안의 변수·열 이름: `query`, `label`, `split`, `rank`, `score`, `seed`, `threshold`

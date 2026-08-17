---
marp: true
theme: 임베딩-course
paginate: true
size: 16:9
math: katex
header: "임베딩 기반 데이터 과학 · 05주차"
footer: "국립목포대학교 컴퓨터학부 · Hands-On LLM 연계"
title: "05주차 · 문장·문서 임베딩"
---

<!-- _class: lead -->

###### WEEK 05 · SENTENCE EMBEDDINGS

# 문장 전체를 검색 가능한 한 벡터로

**풀링 → 이중 인코더 → 대조 학습 → 혼동 비관련 예시**

---

# 문장 임베딩의 계약


> **정의**
> **문장 임베딩(sentence embedding)**: 가변 길이 텍스트 $x$를 고정 길이 벡터 $z=f_\theta(x)\in\mathbb R^d$로 바꾸어, 정의한 유사도 함수로 비교할 수 있게 한 표현.


성공 조건:

- 같은 의도·관련성을 가진 쌍은 가깝게
- 무관하거나 상충하는 쌍은 멀게
- 새 문장을 서로 독립적으로 인코딩 가능
- 대규모 말뭉치의 벡터를 미리 계산·색인 가능

---

# 학습목표

1. 풀링 전략과 패딩 마스크의 영향을 설명한다.
2. 교차 인코더, 이중 인코더, 후기 상호작용을 비용–정확도 관점에서 비교한다.
3. 쌍·삼중항·InfoNCE·MNRL 손실함수의 관련·비관련 예시 구조를 설명한다.
4. SBERT 논문의 계산 효율 주장을 재구성한다.
5. Hands-On LLM 10장의 임베딩 미세 조정 절차를 평가 누수 없이 설계한다.

---

# Token → Sentence: 풀링

$H=[\mathbf h_1,\dots,\mathbf h_T]\in\mathbb R^{T\times d}$에서

| 방식 | 계산 | 특징 |
|---|---|---|
| CLS | $\mathbf h_{CLS}$ | 해당 토큰이 문장표현으로 학습됐는지 중요 |
| mean | $\sum_i m_i\mathbf h_i/\sum_i m_i$ | 모든 유효 토큰 평균, 강한 기본값 |
| max | 좌표별 $\max_i h_{ij}$ | 두드러진 특징 보존, 불안정 가능 |
| 가중 | $\sum_i a_i\mathbf h_i$ | 학습된 가중치/IDF/어텐션 사용 |


> **주의**
> 패딩과 특수 토큰을 무심코 포함하면 길이별 편향이 생긴다. 풀링 구현은 모델 카드의 권장법을 따른다.


---

# 교차 인코더와 이중 인코더


- **교차 인코더(cross-encoder)** — `[q; d]`를 함께 트랜스포머에 넣어 점수화. **장점:** 모든 토큰 쌍이 상호작용해 정밀함. **약점:** 모든 질의–문서 쌍을 다시 계산함.
- **이중 인코더(bi-encoder)** — $f(q)$와 $g(d)$를 독립적으로 인코딩한 뒤 내적이나 코사인 유사도로 비교. **장점:** 문서 벡터를 미리 계산해 ANN 검색 가능. **약점:** 한 벡터에 정보를 압축함.


---

# SBERT의 문제 정의: 조합 폭발

문장 10,000개에서 모든 쌍:

$$\frac{n(n-1)}{2}=49{,}995{,}000$$

SBERT 논문의 보고:

| 방식 | 작업 | 당시 보고 시간 |
|---|---|---:|
| BERT 교차 인코더 | 약 5천만 쌍 추론 | 약 65시간 |
| SBERT 이중 인코더 | 10,000개 인코딩 | 약 5초 |
| 코사인 유사도 행렬 | 벡터 비교 | 약 0.01초 |


*출처: Reimers & Gurevych (2019), [“Sentence-BERT”](https://arxiv.org/abs/1908.10084), 논문의 당시 V100 조건. 현대 하드웨어 절대시간으로 일반화하지 말고 복잡도 차이를 읽는다.*


---

# 논문 핵심 구조: Siamese SBERT

![h:430](assets/papers/sbert-figure2.webp)

*그림: 두 문장을 공유 가중치 BERT와 풀링으로 각각 벡터화한 뒤 코사인 유사도로 비교한다.*

> **교재 연결**
> Chapter 10의 `SentenceTransformer`, 손실함수, 평가기는 이 이중 인코더 구조를 **학습 가능한 임베딩 모델**로 만드는 세 구성요소다.

*출처: Reimers & Gurevych (2019), “Sentence-BERT,” Figure 2 · [원 논문](https://arxiv.org/abs/1908.10084)*


---

# 학습 데이터의 세 구조


- **쌍(pair)** — $(x_1,x_2,y)$: STS 점수 또는 NLI 라벨
- **삼중항(triplet)** — $(a,p,n)$: 기준–관련–비관련 예시
- **배치(batch)** — $(q_i,d_i^+)_{i=1}^B$: 다른 문서를 배치 내 비관련 예시로 사용


> **정의**
> **관련(positive)**과 **비관련(negative)**은 자연적 속성이 아니라 과업의 관련성 정의로 만든 훈련 신호다.


---

# 코사인 회귀과 쌍 손실함수

STS 레이블 $y\in[0,1]$에 맞추는 예:

$$\mathcal L_{MSE}=(\cos(f(x_1),f(x_2))-y)^2$$

분류형 쌍:

$$o=\operatorname{softmax}(W[u;v;|u-v|])$$

- NLI의 함의·중립·모순 관계로 표현을 학습할 수 있다.
- 학습 레이블의 의미와 배포 과업의 “관련성”이 다르면 전이 격차가 생긴다.

---

# 삼중항 손실

$$\mathcal L=\max(d(a,p)-d(a,n)+m,0)$$

- $a$: 기준 예시, $p$: 관련 예시, $n$: 비관련 예시
- $m$: 관련 예시가 비관련 예시보다 최소한 더 가까워야 하는 여유값
- 너무 쉬운 비관련 예시는 손실함수가 0이라 학습 정보가 적다.
- 너무 어렵지만 사실 관련 있는 잘못된 비관련 판정는 잘못된 방향으로 민다.


> **실습**
> **손계산:** $d(a,p)=0.4$, $d(a,n)=0.7$, $m=0.5$일 때 손실함수는? negative를 더 멀리 보내야 하는가?


---

# Multiple Negatives Ranking Loss

배치에서 $(a_i,p_i)$만 정답이고 다른 $p_j$를 비관련 예시로:

$$\mathcal L_i=-\log\frac{e^{s(a_i,p_i)/\tau}}{\sum_{j=1}^{B}e^{s(a_i,p_j)/\tau}}$$

- 한 배치로 $B^2$ 점수를 만들어 데이터 효율이 높다.
- 큰 배치는 더 많은 negative를 제공하지만 메모리·잘못된 비관련 판정 위험도 증가한다.
- 동일 문서/중복 의미를 배치에서 제거하거나 multi-positive 손실함수를 고려한다.

---

# 혼동 비관련 예시 선별


> **정의**
> **혼동 비관련 예시**: 현재 모델이 높은 점수를 주지만 라벨 기준으로는 관련 없는 후보.


대표 절차:

1. 기준선 이중 인코더로 상위 k개 후보를 검색한다.
2. 정답을 제외한 상위 후보를 비관련 후보로 만든다.
3. 교차 인코더 또는 사람 라벨로 잘못된 비관련 판정을 걸러낸다.
4. 쉬운 비관련 예시와 혼동 비관련 예시를 섞어 재학습한다.


> **주의**
> 정답 목록이 불완전한 검색 데이터에서 “top인데 미라벨”을 바로 negative로 쓰면 실제 정답을 밀어내는 오류가 생긴다.


---

# 지시문과 비대칭 검색

현대 모델은 입력 역할을 명시하는 접두문 또는 지시문(instruction)을 사용한다.

```text
질의: 장학금 신청 마감일은?
passage: 2026학년도 2학기 장학금 신청은 ...
```

- 질의는 정보 요구, 문서는 답의 근거라 분포가 비대칭이다.
- 모델 카드가 요구하는 접두문을 빼면 성능이 크게 변할 수 있다.
- 대칭형 STS와 비대칭 검색을 같은 프롬프트로 평가하지 않는다.

---

<!-- _class: section -->

# Hands-On LLM 연결

## 10장의 미세 조정을 데이터–손실함수–평가 삼각형으로 읽는다

---

# Chapter 10 파이프라인


**데이터 쌍/triplet** → **기반 모델 + 풀링** → **손실 목표** → **평가기** → **학습·저장**


> **교재 연결**
> **원본 핵심:** 코사인 손실, MNRL, 지도학습 미세 조정, AugSBERT, TSDAE를 “코드 레시피”가 아니라 어떤 데이터 가정과 신호를 쓰는지 비교한다.


필수 기록:

- 기반 체크포인트와 모델 버전
- 풀링/정규화/프롬프트
- 분할 기준(문서·주제·시간 중복 방지)
- 난수 시드, 배치, 학습률, 에포크
- best checkpoint 선택 지표

---

# Sentence Transformers 코드 읽기

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("model-name")
Z = model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
scores = Z @ Z.T  # 정규화했을 때 cosine
```

질문:

1. `normalize_embeddings=False`면 이 내적은 무엇을 포함하는가?
2. 긴 문서는 어디서 잘리는가?
3. `model.max_seq_length`와 토큰화기의 길이 제한이 일치하는가?
4. dtype을 float16/int8로 바꾸면 순위가 얼마나 변하는가?

---

# 평가: STS와 검색은 다르다

| 과업 | 입력 | 대표 지표 |
|---|---|---|
| STS | 문장 쌍 + 연속 유사도 | Spearman $\rho$ |
| 분류 | 벡터 + 레이블 | 정확도, 매크로 F1 |
| 군집화 | 벡터 집합 | V-measure, ARI |
| 검색 | 질의 + 말뭉치 + qrels | nDCG@k, 재현율@k, MRR |


> **주의**
> STS가 높은 모델이 장문 한국어 검색에서도 최고라는 보장은 없다. 학습목표·입력 길이·언어·도메인이 다르다.


---

# 최신 동향 ① 다기능·다국어 임베딩

| 연구 | 보고된 방향 | 강의에서 확인할 것 |
|---|---|---|
| BGE-M3 (2024) | 밀집/희소/다중 벡터, 다국어·장문 | 모드별 비용과 결합 이득 |
| Gemini Embedding (2025) | 대규모 multilingual 모델 | 공개 평가 조건·접근 방식 |
| Qwen3 Embedding (2025) | 0.6B/4B/8B 임베딩·재순위화 모델 | 크기–성능–지연시간 |
| jina-embeddings-v4 (2025) | multimodal, single/multi-vector, 과업 LoRA | 모달리티·adapter별 성능 |


*출처: [BGE-M3](https://arxiv.org/abs/2402.03216) · [Gemini Embedding](https://arxiv.org/abs/2503.07891) · [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) · [Jina v4](https://arxiv.org/abs/2506.18902)*


---

# 최신 동향 ② Matryoshka와 비용 조절

한 모델의 출력 차원을 서비스 예산에 맞춰 줄이는 경우:

| 차원 | 벡터 100만 개 float32 | 기대 효과 |
|---:|---:|---|
| 1024 | 약 3.81 GiB | 최대 표현 용량 |
| 256 | 약 0.95 GiB | 저장공간·대역폭 1/4 |
| 64 | 약 0.24 GiB | 빠른 1차 후보 검색 |


> **최신 동향**
> 중요한 것은 차원 감소율이 아니라 자신의 말뭉치에서 재현율@k와 지연시간이 어떻게 변하는지 측정하는 것이다.


*출처: Kusupati et al. (2022), [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)*


---

# 수업 활동: 손실함수를 데이터로 선택하라

세 상황에 어떤 손실함수/데이터가 적합한가?

1. 문장 쌍에 0–5 유사도 점수가 있다.
2. 각 질문에 정답 문서 하나가 있고 말뭉치가 크다.
3. 라벨 없는 도메인 문서만 있다.

팀별로 다음을 명시한다.

- positive/negative의 정의
- 잘못된 비관련 판정 방지책
- 개발 데이터 지표
- 기준선과 요소 제거 실험


> **실습**
> 정답은 하나가 아니다. 선택이 데이터 가정과 지표에 논리적으로 연결되는지가 평가 대상이다.


---

# 형성평가

1. cross-인코더가 정확하지만 1차 검색에 비싼 이유는?
2. mean 풀링에서 어텐션 마스크가 필요한 이유는?
3. in-배치 negative의 장점과 위험은?
4. 혼동 비관련 예시를 사람이/교사 모델이 다시 확인해야 하는 이유는?
5. STS 성능만으로 검색 모델을 선택하면 안 되는 이유는?


> **교재 연결**
> **Notebook:** `../notebooks/week05.ipynb` · 풀링과 모델별 유사도/검색 오류를 비교한다.


---

# 핵심 정리

- 문장 임베딩은 가변 길이 텍스트를 과업에 맞는 한 벡터로 압축한다.
- bi-인코더는 사전 계산으로 확장성을 얻고 세밀한 토큰 interaction을 희생한다.
- 손실함수의 이름보다 positive/negative가 무엇인지가 중요하다.
- 혼동 비관련 예시는 강한 학습 신호이지만 잘못된 비관련 판정 검증이 필수다.
- 최신 모델 선택은 순위표 하나가 아니라 언어·도메인·길이·비용으로 결정한다.

> **핵심**
> 다음 주: 임베딩을 **분류·군집·토픽 분석의 데이터**로 쓴다.

---

<!-- _class: compact -->

# 참고문헌과 원본 연결

- Reimers & Gurevych (2019), “Sentence-BERT.” [arXiv](https://arxiv.org/abs/1908.10084)
- Gao et al. (2021), “SimCSE.” [EMNLP](https://aclanthology.org/2021.emnlp-main.552/)
- Wang et al. (2022), “Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).” [arXiv](https://arxiv.org/abs/2212.03533)
- Kusupati et al. (2022), “Matryoshka Representation Learning.” [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)
- Zhang et al. (2025), “Qwen3 Embedding.” [arXiv](https://arxiv.org/abs/2506.05176)
- Hands-On Large Language Models, Chapter 10. [upstream](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter10)

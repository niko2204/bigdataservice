---
marp: true
theme: 임베딩-course
paginate: true
size: 16:9
math: katex
header: "임베딩 기반 데이터 과학 · 09주차"
footer: "국립목포대학교 컴퓨터학부 · Hands-On LLM 연계"
title: "09주차 · 멀티모달 임베딩"
---

<!-- _class: lead -->

###### WEEK 09 · MULTIMODAL EMBEDDINGS

# 텍스트와 이미지를 같은 의미 공간에 정렬하기

**dual 인코더 → 대조학습 정렬 → 시각 문서 검색 → 옴니모달**

---

# 다른 센서, 공유할 수 있는 의미


- **텍스트** — “푸른 바다 위를 항해하는 흰 배”
- **이미지** — 픽셀 배열 속 바다·배·색·구도


> **정의**
> **멀티모달 임베딩**: 서로 다른 입력 양식(modality)을 비교 가능한 공동 공간 또는 정렬된 표현으로 매핑한 학습 표현.


목표: 대응 이미지–텍스트는 가깝게, 비대응 쌍은 멀게.

---

# 학습목표

1. 입력 양식, 정렬, 융합, 공유 임베딩 공간을 설명한다.
2. CLIP의 dual 인코더와 symmetric contrastive 손실함수를 설명한다.
3. 무예시 분류와 교차 양식 검색의 프롬프트 의존성을 분석한다.
4. OCR 기반 문서 검색과 시각 문서 임베딩을 비교한다.
5. Hands-On LLM 9장의 멀티모달 입력을 검색/생성 관점으로 분리한다.

---

# 멀티모달 설계의 세 방식


- **Dual 인코더** — 각 modality를 독립 인코딩 후 공동 공간에서 비교. 대규모 검색에 유리.
- **융합 인코더** — 토큰/패치를 함께 어텐션으로 처리. 세밀하지만 모든 쌍 계산이 비쌈.
- **후기 상호작용** — 각자 미리 계산한 다중 벡터를 질의 시점에 세밀하게 대응.


텍스트 검색의 bi-/cross-/ColBERT 구분과 구조적으로 대응한다.

---

# 이미지를 토큰처럼: 패치 표현

Vision Transformer의 기본 직관:

1. 이미지를 $P\times P$ patch로 나눈다.
2. 각 patch를 펼쳐 선형 투영한다.
3. 위치 정보를 더한다.
4. 트랜스포머가 패치 관계를 문맥화한다.

$$N=\frac{HW}{P^2},\qquad X_{patch}\in\mathbb R^{N\times d}$$


> **주의**
> 작은 글자·표·세밀한 도형은 이미지 해상도와 패치 크기 때문에 손실될 수 있다.


---

# CLIP: 대조적 언어–이미지 사전학습

배치 $N$개의 이미지–텍스트 쌍에서 $N\times N$ 유사도 행렬을 만든다.

$$S_{ij}=\frac{\hat I_i^\top\hat T_j}{\tau}$$

대각선 $(i=i)$을 정답으로 이미지→텍스트와 텍스트→이미지 cross-entropy를 평균한다.

$$\mathcal L=\frac{1}{2}(\mathcal L_{I\rightarrow T}+\mathcal L_{T\rightarrow I})$$


> **논문 읽기**
> 원 연구는 웹에서 수집한 4억 이미지–텍스트 쌍로 학습하고 30개가 넘는 vision 데이터셋에서 무예시 transfer를 연구했다.


*출처: Radford et al. (2021), [“Learning Transferable Visual Models From Natural Language Supervision”](https://arxiv.org/abs/2103.00020)*


---

# 논문 핵심 도판: CLIP의 세 단계


![w:960](assets/papers/clip-figure1.webp)


*그림: (1) 이미지–텍스트 대조 사전학습 → (2) 레이블 문장으로 분류기 생성 → (3) 무예시 예측.*


> **교재 연결**
> Chapter 9의 CLIP 코드는 이미지 임베딩과 텍스트 임베딩을 직접 비교하고, 프롬프트 문구가 무예시 분류기의 결정경계를 어떻게 바꾸는지 실험한다.


*출처: Radford et al. (2021), Figure 1 · [원문 HTML](https://ar5iv.labs.arxiv.org/html/2103.00020)*


---

# Figure 1 읽기


- **1. Pre-학습** — 대각선 쌍 점수는 높게, 나머지는 낮게.
- **2. 텍스트 분류기** — `A photo of a {label}`을 인코딩해 클래스 가중치로.
- **3. Predict** — 새 이미지 임베딩과 범주 텍스트 임베딩의 cosine 비교.


> **정의**
> 분류 head를 레이블 데이터로 다시 학습하지 않고 자연어 레이블 설명으로 동적으로 구성하는 것이 무예시 transfer의 핵심이다.


---

# Prompt가 곧 클래스 정의다

단순 레이블 `배`는 fruit/boat/stomach 중 모호하다.

```text
배
바다 위를 항해하는 배의 사진
먹을 수 있는 한국 배 과일의 사진
```

- 범주 name만 쓰지 말고 문맥 template을 사용
- 여러 template 임베딩을 평균해 ensemble
- 한국어/영어 prompt의 pretraining 분포 차이 평가
- 범주가 폐쇄집합인지, “해당 없음” 처리가 있는지 확인

---

# 교차 양식 검색

| 방향 | 질의 | 말뭉치 | 예 |
|---|---|---|---|
| 텍스트→이미지 | 문장 | 이미지 embeddings | “빨간 안전모를 쓴 작업자” |
| 이미지→텍스트 | 이미지 | captions/docs | 사진 설명·상품 찾기 |
| 이미지→이미지 | 이미지 | images | 유사 디자인/제품 |
| 텍스트→page | 질문 | PDF page images | 표·도표가 있는 규정 페이지 |

평가: 재현율@k, 순위 중앙값, nDCG, 언어/객체/텍스트 포함 여부별 부분집합.

---

# OCR-first 문서 검색


**PDF page** → **OCR/layout** → **텍스트 청크** → **텍스트 임베딩** → **retrieve**


장점:

- 텍스트를 직접 인용·검색·필터링 가능
- 기존 텍스트 RAG 인프라 재사용

약점:

- 표 셀 관계, 도식, 글자 위치가 선형 텍스트에서 손실
- OCR 오류가 후속 모든 단계에 전파

---

# 시각 문서 검색

페이지 이미지를 vision-language 인코더로 직접 표현:


**페이지 이미지** → **패치 다중 벡터** → **질의 토큰 다중 벡터** → **후기 상호작용**


- 원래 layout·표·그림·font 단서를 보존
- page당 다중 벡터라 저장·검색 비용 증가
- 찾은 페이지에서 정확한 답을 추출하는 별도 단계 필요

---

# 논문 도판: ColPali의 토큰–패치 유사도


![h:420](assets/papers/colpali-similarity-map.webp)


*그림: 텍스트 질의 토큰이 문서 페이지의 어떤 시각 영역과 높은 후기 상호작용 점수를 만드는지 보여주는 유사도 지도.*


*출처: Faysse et al. (2024), “ColPali,” similarity map · [원 논문](https://arxiv.org/abs/2407.01449)*


---

# ColPali가 바꾼 질문

기존 질문: “OCR을 얼마나 잘해서 텍스트로 바꿀까?”

새 질문: “페이지 시각 구조를 그대로 보존해 검색할 수 있을까?”

- PaliGemma 계열 VLM으로 페이지를 패치 수준 다중 벡터로 표현
- ColBERT식 후기 상호작용으로 질의 토큰과 페이지 패치를 대응
- 시각적으로 풍부한 문서 검색을 위한 ViDoRe benchmark 제시


> **주의**
> 유사도 map은 모델 내부 점수 시각화다. 정답 근거의 인과적 증명이나 OCR 정확도를 대신하지 않는다.


---

<!-- _class: section -->

# Hands-On LLM 연결

## Chapter 9의 VLM을 표현·검색·생성으로 분해한다

---

# 멀티모달 LLM 입력 흐름


**이미지** → **Vision 인코더** → **시각 토큰** → **텍스트 토큰** → **LLM 생성**


> **교재 연결**
> **구분:** VLM이 이미지를 설명하는 생성 능력과, 말뭉치 전체를 빠르게 검색하는 임베딩 능력은 같은 평가 문제가 아니다.


확인할 tensor/조건:

- 이미지 resolution/resize/crop
- 이미지 토큰 또는 패치 수
- 텍스트–vision projector
- 최대 문맥와 여러 이미지 처리
- chat template와 특수 토큰

---

# 생성 데모를 평가로 바꾸기

```python
prompt = "이 이미지의 표에서 2026년 예산을 근거와 함께 답하라."
output = vlm.generate(image, prompt)
```

필요한 평가:

- OCR 정확도: 숫자·단위·한글
- 근거 일치성: 답이 실제 표 셀/영역에 있는가?
- 근거 없는 생성(환각): 보이지 않는 내용을 만들었는가?
- robustness: resize, crop, 회전, 압축에 견디는가?
- 지연시간/메모리 사용량: 텍스트-only 대비 비용

---

# 최신 동향 ① 범용 멀티모달 임베딩

| 연구 | 방향 | 발표 상태 |
|---|---|---|
| jina-embeddings-v4 | 텍스트+이미지, single/multi-vector, 과업 LoRA | 2025 preprint |
| Qwen3-VL-Embedding | 텍스트/이미지/mixed input, 긴 문맥·Matryoshka | 2026 preprint |
| Gemini Embedding 2 | 텍스트/이미지/audio/video 공동 공간 | 2026 preprint |
| jina-v5-omni | 텍스트/이미지/audio/video, GELATO | 2026 preprint |


*출처: [Jina v4](https://arxiv.org/abs/2506.18902) · [Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) · [Gemini Embedding 2](https://arxiv.org/html/2605.27295v1) · [Jina v5 omni](https://arxiv.org/html/2605.08384v2)*


---

# 최신 동향 ② “한 벡터”에서 선택 가능한 표현으로


- **Single vector** — 빠른 1차 검색
- **Multi-vector** — 토큰–패치 정밀 매칭
- **Matryoshka** — 차원별 비용 조절


> **최신 동향**
> 모델 하나가 여러 모달리티·과업·granularity를 지원하는 방향이지만, 색인 복잡도·adapter 선택·평가 범위가 함께 커진다.


---

# 멀티모달 편향과 안전

- 웹 이미지–텍스트 쌍의 문화·언어·성별·직업 연관 편향
- 얼굴/사람 검색의 프라이버시와 오인식 피해
- OCR에 포함된 개인정보·문서 권한 유출
- prompt에 따라 범주 definition이 이동
- 적대적 이미지/텍스트가 공동 공간을 교란


> **주의**
> **배포 전:** 얼굴/민감 속성 용도 제한, 집단별 평가, consent와 retention, access control, human review를 요구사항으로 둔다.


---

# 수업 활동: OCR과 시각 임베딩 비교

같은 PDF 페이지 20장과 질문 10개로 비교:

| 방식 | 표현 | 측정 |
|---|---|---|
| A | OCR 텍스트 + 텍스트 임베딩 | 재현율@5, OCR 오류 |
| B | 페이지 이미지 + 시각 임베딩 | 재현율@5, 지연시간/저장공간 |
| C | A+B 혼합 | 재현율@5, 오류 보완 |

페이지 유형별 부분집합: 본문 / 표 / 차트 / 스캔 / 2단 편집.


> **실습**
> **결론:** 평균뿐 아니라 어떤 layout에서 어느 방식이 이기는지 오류 지도를 작성한다.


---

# 형성평가

1. CLIP의 $N\times N$ 행렬에서 positive는 어디인가?
2. 무예시 분류에서 텍스트 prompt가 하는 역할은?
3. 이중 인코더와 융합 인코더의 검색 비용 차이는?
4. OCR-first와 page-이미지 검색의 정보 손실이 각각 무엇인가?
5. ColPali가 single-vector CLIP 검색과 다른 점은?


> **교재 연결**
> **Notebook:** `../notebooks/week09.ipynb` · 이미지/텍스트 유사도와 프롬프트 변화를 기록한다.


---

# 핵심 정리

- 멀티모달 임베딩은 서로 다른 입력을 비교 가능한 공간에 정렬한다.
- CLIP은 배치 대조학습과 자연어 클래스 설명으로 무예시 transfer를 가능하게 했다.
- prompt는 멀티모달 분류의 동적 클래스 정의다.
- 문서 검색은 OCR 텍스트와 시각적 레이아웃 중 무엇을 보존할지 선택한다.
- 최신 방향은 single/multi-vector, Matryoshka, 옴니모달을 한 모델에 통합한다.

> **핵심**
> 다음 주: 모든 선택을 **하나의 재현 가능한 프로젝트**로 통합한다.

---

<!-- _class: compact -->

# 참고문헌과 원본 연결

- Radford et al. (2021), “CLIP.” [arXiv](https://arxiv.org/abs/2103.00020)
- Dosovitskiy et al. (2021), “An Image is Worth 16x16 Words.” [ICLR](https://openreview.net/forum?id=YicbFdNTTy)
- Faysse et al. (2024), “ColPali.” [arXiv](https://arxiv.org/abs/2407.01449)
- Lähn et al. (2025), “jina-embeddings-v4.” [arXiv](https://arxiv.org/abs/2506.18902)
- Qwen Team (2026), “Qwen3-VL-Embedding.” [arXiv](https://arxiv.org/abs/2601.04720)
- Hands-On Large Language Models, Chapter 9. [upstream](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter09)

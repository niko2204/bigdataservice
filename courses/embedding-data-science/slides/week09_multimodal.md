---
marp: true
theme: embedding-course
paginate: true
size: 16:9
math: katex
header: "임베딩 기반 데이터 과학 · 09주차"
footer: "국립목포대학교 컴퓨터학부 · Hands-On LLM 연계"
title: "09주차 · 멀티모달 임베딩"
---

<!-- _class: lead -->

<p class="kicker">WEEK 09 · MULTIMODAL EMBEDDINGS</p>

# 텍스트와 이미지를<br>같은 의미 공간에 정렬하기

**dual encoder → contrastive alignment → visual document retrieval → omni-modal**

---

# 다른 센서, 공유할 수 있는 의미

<div class="cols">
<div class="card"><h3>텍스트</h3><p>“푸른 바다 위를 항해하는 흰 배”</p></div>
<div class="card"><h3>이미지</h3><p>픽셀 배열 속 바다·배·색·구도</p></div>
</div>

<div class="definition"><b>멀티모달 임베딩</b>: 서로 다른 양식(modality)의 입력을 비교 가능한 공동 공간 또는 정렬된 표현으로 매핑하는 학습된 표현.</div>

목표: 대응 image–text는 가깝게, 비대응 쌍은 멀게.

---

# 학습목표

1. modality, alignment, fusion, shared embedding space를 설명한다.
2. CLIP의 dual encoder와 symmetric contrastive loss를 설명한다.
3. zero-shot 분류와 cross-modal retrieval의 prompt 의존성을 분석한다.
4. OCR 기반 문서 검색과 visual document embedding을 비교한다.
5. Hands-On LLM 9장의 멀티모달 입력을 검색/생성 관점으로 분리한다.

---

# 멀티모달 설계의 세 방식

<div class="three">
<div class="card"><h3>Dual encoder</h3><p>각 modality를 독립 encode 후 공동 공간에서 비교. 대규모 검색에 유리.</p></div>
<div class="card"><h3>Fusion encoder</h3><p>토큰/패치를 함께 attention. 세밀하지만 모든 쌍 계산이 비쌈.</p></div>
<div class="card"><h3>Late interaction</h3><p>각자 사전 계산한 다중 벡터를 query 시점에 세밀하게 매칭.</p></div>
</div>

텍스트 검색의 bi-/cross-/ColBERT 구분과 구조적으로 대응한다.

---

# 이미지를 토큰처럼: patch representation

Vision Transformer의 기본 직관:

1. 이미지를 $P\times P$ patch로 나눈다.
2. 각 patch를 펼쳐 선형 투영한다.
3. position 정보를 더한다.
4. Transformer가 patch 관계를 문맥화한다.

$$N=\frac{HW}{P^2},\qquad X_{patch}\in\mathbb R^{N\times d}$$

<div class="warning">작은 글자·표·세밀한 도형은 image resolution과 patch size 때문에 손실될 수 있다.</div>

---

# CLIP: 대조적 언어–이미지 사전학습

배치 $N$개의 image–text pair에서 $N\times N$ 유사도 행렬을 만든다.

$$S_{ij}=\frac{\hat I_i^\top\hat T_j}{\tau}$$

대각선 $(i=i)$을 정답으로 image→text와 text→image cross-entropy를 평균한다.

$$\mathcal L=\frac{1}{2}(\mathcal L_{I\rightarrow T}+\mathcal L_{T\rightarrow I})$$

<div class="paper-note">원 연구는 웹에서 수집한 4억 image–text pair로 학습하고 30개가 넘는 vision dataset에서 zero-shot transfer를 연구했다.</div>

<div class="source">Radford et al. (2021), <a href="https://arxiv.org/abs/2103.00020">“Learning Transferable Visual Models From Natural Language Supervision”</a></div>

---

# 논문 핵심 도판: CLIP의 세 단계

<div class="figure figure-sm">

![CLIP Figure 1](https://ar5iv.labs.arxiv.org/html/2103.00020/assets/main-diagrams.png)

</div>

<div class="caption">(1) image–text 대조 사전학습 → (2) label 문장으로 분류기 생성 → (3) zero-shot 예측.</div>

<div class="source">Radford et al. (2021), Figure 1 · <a href="https://ar5iv.labs.arxiv.org/html/2103.00020">원문 HTML</a></div>

---

# Figure 1 읽기

<div class="three">
<div class="card"><h3>1. Pre-train</h3><p>대각선 pair 점수는 높게, 나머지는 낮게.</p></div>
<div class="card"><h3>2. Text classifier</h3><p>`A photo of a {label}`을 encode해 클래스 weight로.</p></div>
<div class="card"><h3>3. Predict</h3><p>새 image embedding과 class text embedding의 cosine 비교.</p></div>
</div>

<div class="definition">분류 head를 label data로 다시 학습하지 않고 자연어 label 설명으로 동적으로 구성하는 것이 zero-shot transfer의 핵심이다.</div>

---

# Prompt가 곧 클래스 정의다

단순 label `배`는 fruit/boat/stomach 중 모호하다.

```text
배
바다 위를 항해하는 배의 사진
먹을 수 있는 한국 배 과일의 사진
```

- class name만 쓰지 말고 문맥 template을 사용
- 여러 template embedding을 평균해 ensemble
- 한국어/영어 prompt의 pretraining 분포 차이 평가
- class가 폐쇄집합인지, “해당 없음” 처리가 있는지 확인

---

# Cross-modal retrieval

| 방향 | query | corpus | 예 |
|---|---|---|---|
| text→image | 문장 | image embeddings | “빨간 안전모를 쓴 작업자” |
| image→text | 이미지 | captions/docs | 사진 설명·상품 찾기 |
| image→image | 이미지 | images | 유사 디자인/제품 |
| text→page | 질문 | PDF page images | 표·도표가 있는 규정 페이지 |

평가: Recall@k, median rank, nDCG, 언어/객체/텍스트 포함 여부별 slice.

---

# OCR-first 문서 검색

<div class="pipeline">
<div>PDF page</div><span>→</span><div>OCR/layout</div><span>→</span><div>text chunks</div><span>→</span><div>text embedding</div><span>→</span><div>retrieve</div>
</div>

장점:

- 텍스트를 직접 인용·검색·필터링 가능
- 기존 text RAG 인프라 재사용

약점:

- 표 셀 관계, 도식, 글자 위치가 선형 텍스트에서 손실
- OCR 오류가 후속 모든 단계에 전파

---

# Visual document retrieval

페이지 이미지를 vision-language encoder로 직접 표현:

<div class="pipeline">
<div>page image</div><span>→</span><div>patch<br>multi-vectors</div><span>↔</span><div>query token<br>multi-vectors</div><span>→</span><div>late interaction</div>
</div>

- 원래 layout·표·그림·font 단서를 보존
- page당 다중 벡터라 저장·검색 비용 증가
- 찾은 페이지에서 정확한 답을 추출하는 별도 단계 필요

---

# 논문 도판: ColPali의 token–patch 유사도

<div class="figure">

![ColPali similarity map for energy query](https://arxiv.org/html/2407.01449v2/images/similarity_maps/similarity_map_energy.png)

</div>

<div class="caption">텍스트 query token이 문서 페이지의 어떤 시각 영역과 높은 late-interaction 점수를 만드는지 보여주는 similarity map.</div>

<div class="source">Faysse et al. (2024), “ColPali,” similarity map · <a href="https://arxiv.org/abs/2407.01449">원 논문</a></div>

---

# ColPali가 바꾼 질문

기존 질문: “OCR을 얼마나 잘해서 텍스트로 바꿀까?”

새 질문: “페이지 시각 구조를 그대로 보존해 검색할 수 있을까?”

- PaliGemma 계열 VLM으로 페이지를 patch-level multi-vector로 표현
- ColBERT식 late interaction으로 query token과 page patch 매칭
- 시각적으로 풍부한 문서 검색을 위한 ViDoRe benchmark 제시

<div class="warning">similarity map은 모델 내부 점수 시각화다. 정답 근거의 인과적 증명이나 OCR 정확도를 대신하지 않는다.</div>

---

<!-- _class: section -->

# Hands-On LLM 연결

## Chapter 9의 VLM을<br>표현·검색·생성으로 분해한다

---

# 멀티모달 LLM 입력 흐름

<div class="pipeline">
<div>이미지</div><span>→</span><div>Vision encoder</div><span>→</span><div>visual tokens</div><span>+</span><div>text tokens</div><span>→</span><div>LLM generation</div>
</div>

<div class="hllm"><b>구분:</b> VLM이 이미지를 설명하는 생성 능력과, corpus 전체를 빠르게 검색하는 embedding 능력은 같은 평가 문제가 아니다.</div>

확인할 tensor/조건:

- image resolution/resize/crop
- image token 또는 patch 수
- text–vision projector
- 최대 context와 여러 이미지 처리
- chat template와 special token

---

# 생성 데모를 평가로 바꾸기

```python
prompt = "이 이미지의 표에서 2026년 예산을 근거와 함께 답하라."
output = vlm.generate(image, prompt)
```

필요한 평가:

- OCR 정확도: 숫자·단위·한글
- grounding: 답이 실제 표 셀/영역에 있는가?
- hallucination: 보이지 않는 내용을 만들었는가?
- robustness: resize, crop, 회전, 압축에 견디는가?
- latency/memory: text-only 대비 비용

---

# 최신 동향 ① 범용 멀티모달 임베딩

| 연구 | 방향 | 발표 상태 |
|---|---|---|
| jina-embeddings-v4 | text+image, single/multi-vector, task LoRA | 2025 preprint |
| Qwen3-VL-Embedding | text/image/mixed input, 긴 문맥·Matryoshka | 2026 preprint |
| Gemini Embedding 2 | text/image/audio/video 공동 공간 | 2026 preprint |
| jina-v5-omni | text/image/audio/video, GELATO | 2026 preprint |

<div class="source"><a href="https://arxiv.org/abs/2506.18902">Jina v4</a> · <a href="https://arxiv.org/abs/2601.04720">Qwen3-VL-Embedding</a> · <a href="https://arxiv.org/html/2605.27295v1">Gemini Embedding 2</a> · <a href="https://arxiv.org/html/2605.08384v2">Jina v5 omni</a></div>

---

# 최신 동향 ② “한 벡터”에서 선택 가능한 표현으로

<div class="three">
<div class="card"><h3>Single vector</h3><p>빠른 1차 검색</p></div>
<div class="card"><h3>Multi-vector</h3><p>토큰–패치 정밀 매칭</p></div>
<div class="card"><h3>Matryoshka</h3><p>차원별 비용 조절</p></div>
</div>

<div class="trend">모델 하나가 여러 모달리티·과업·granularity를 지원하는 방향이지만, index 복잡도·adapter 선택·평가 범위가 함께 커진다.</div>

---

# 멀티모달 편향과 안전

- 웹 image–text pair의 문화·언어·성별·직업 연관 편향
- 얼굴/사람 검색의 프라이버시와 오인식 피해
- OCR에 포함된 개인정보·문서 권한 유출
- prompt에 따라 class definition이 이동
- adversarial image/text가 공동 공간을 교란

<div class="warning"><b>배포 전:</b> 얼굴/민감 속성 용도 제한, 집단별 평가, consent와 retention, access control, human review를 요구사항으로 둔다.</div>

---

# 수업 활동: OCR vs visual embedding

같은 PDF 페이지 20장과 질문 10개로 비교:

| 방식 | 표현 | 측정 |
|---|---|---|
| A | OCR text + text embedding | Recall@5, OCR error |
| B | page image + visual embedding | Recall@5, latency/storage |
| C | A+B hybrid | Recall@5, 오류 보완 |

페이지 유형별 slice: 본문 / 표 / 차트 / 스캔 / 2단 편집.

<div class="lab"><b>결론:</b> 평균뿐 아니라 어떤 layout에서 어느 방식이 이기는지 오류 지도를 작성한다.</div>

---

# 형성평가

1. CLIP의 $N\times N$ 행렬에서 positive는 어디인가?
2. zero-shot 분류에서 text prompt가 하는 역할은?
3. dual encoder와 fusion encoder의 검색 비용 차이는?
4. OCR-first와 page-image 검색의 정보 손실이 각각 무엇인가?
5. ColPali가 single-vector CLIP 검색과 다른 점은?

<div class="hllm"><b>Notebook:</b> `../notebooks/week09.ipynb` · image/text similarity와 prompt 변화를 기록한다.</div>

---

# 핵심 정리

- 멀티모달 임베딩은 서로 다른 입력을 비교 가능한 공간에 정렬한다.
- CLIP은 batch 대조학습과 자연어 클래스 설명으로 zero-shot transfer를 가능하게 했다.
- prompt는 멀티모달 분류의 동적 클래스 정의다.
- 문서 검색은 OCR text와 visual layout 중 무엇을 보존할지 선택한다.
- 최신 방향은 single/multi-vector, Matryoshka, omni-modal을 한 모델에 통합한다.

<p class="takeaway">다음 주: 모든 선택을<br><mark>하나의 재현 가능한 프로젝트</mark>로 통합한다.</p>

---

<!-- _class: compact -->

# 참고문헌과 원본 연결

- Radford et al. (2021), “CLIP.” [arXiv](https://arxiv.org/abs/2103.00020)
- Dosovitskiy et al. (2021), “An Image is Worth 16x16 Words.” [ICLR](https://openreview.net/forum?id=YicbFdNTTy)
- Faysse et al. (2024), “ColPali.” [arXiv](https://arxiv.org/abs/2407.01449)
- Lähn et al. (2025), “jina-embeddings-v4.” [arXiv](https://arxiv.org/abs/2506.18902)
- Qwen Team (2026), “Qwen3-VL-Embedding.” [arXiv](https://arxiv.org/abs/2601.04720)
- Hands-On Large Language Models, Chapter 9. [upstream](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter09)

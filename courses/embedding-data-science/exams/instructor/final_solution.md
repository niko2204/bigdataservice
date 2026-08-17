# 교수자용 · 기말고사 모범답안과 채점 루브릭

> **배포 주의:** 공개 저장소에서 학생이 볼 수 있다. 실제 시험 전 비공개 위치로 옮기거나
> 수치·사례를 변형한다.

총점 100점. 의미가 같은 논리적 답을 인정하며, 단순 용어 나열은 완전한 점수를 주지 않는다.

## 문제 1 — 15점

### 1.1 — 6점

각 2점: 선택 1점 + 이유 1점.

1. 분류: 이미 정의된 부서 라벨로 새 관측치를 예측. 군집 번호는 기존 업무 라벨과 자동 대응하지 않음.
2. 군집: 라벨 없이 구조·주제 후보를 탐색. 분류는 사전 라벨이 필요.
3. 차원축소: 시각화를 위한 저차원 mapping. 군집/분류는 2차원 좌표 제공이 목적이 아님.

### 1.2 — 4점

타당한 이유 각 2점, 두 개:

- UMAP은 비선형이며 하이퍼파라미터·seed에 따라 그림이 변함.
- 국소 구조와 전역 거리/밀도를 동시에 완전히 보존하지 않음.
- 2D에서 생긴 분리 또는 겹침이 원공간 decision boundary와 동일하지 않음.
- 시각적 선택·라벨 색칠이 판단을 편향할 수 있음.

### 1.3 — 5점

- silhouette는 같은 군집의 응집도와 다른 군집과의 분리를 거리로 측정하는 내부 지표 (1.5)
- 사용자 의미, 라벨, 과업 성공과 직접 동일하지 않음 (1.5)
- 외부/과업 평가는 gold label, downstream 성능, 사람 유용성 등 외부 기준 사용 (1)
- 사람 평가 예: top terms/대표 문서를 가리고 topic coherence·intruder detection·명명 가능성을
  복수 평가자가 판정 (1)

## 문제 2 — 20점

### 2.1 — 12점

- binary relevant는 세 개: 위치 1, 3, 5.
- $P@5=3/5=0.6$ (2)
- $R@5=3/4=0.75$ (2)
- 첫 relevant가 rank 1이므로 reciprocal rank $=1$ (2)

$$
DCG=\frac{3}{1}+0+\frac{1}{2}+0+\frac{3}{2.585}
\approx3+0.5+1.160=4.660
$$

DCG 3점.

검색되지 않은 관련 문서의 relevance 1도 포함하면 ideal relevance는 `[2,2,1,1,0]`:

$$
IDCG=3+\frac{3}{1.585}+\frac{1}{2}+\frac{1}{2.322}
\approx3+1.893+0.5+0.431=5.824
$$

$$nDCG\approx4.660/5.824\approx0.800$$

IDCG 2점, nDCG 1점. 반올림 허용.

### 2.2 — 4점

- FAQ: MRR, Hit@1 또는 P@1 우선. 첫 답을 빨리 찾는 경험 (2)
- 법률 조사: Recall@k 우선. 관련 문서를 놓치는 비용이 큼. precision/검토비용도 함께 고려 (2)

다른 지표를 선택해도 사용자 행동·오류비용과 논리적으로 연결하면 인정.

### 2.3 — 4점

- test에 대한 model/threshold overfitting이며 보고 점수가 낙관적으로 편향 (2)
- dev에서 선택하고 잠근 뒤 별도 test에서 한 번 평가하거나 새 test를 구축 (2)

## 문제 3 — 20점

### 3.1 — 6점

- exact: 모든 후보를 비교해 정확한 최근접, 큰 N·d에서 시간/연산 큼 (2)
- ANN: 일부 탐색/인덱스 구조로 latency를 줄이나 이웃을 놓칠 수 있고 인덱스 메모리·구축비 발생 (2)
- ANN Recall@k는 exact top-k 중 얼마를 찾았는지 측정해 근사 손실을 분리 (2)

### 3.2 — 4점

- recall 증가와 latency 증가의 trade-off이며 서비스 SLA·QPS·오류비용에 따라 선택 (2)
- 필요한 정보 두 가지, 각 1점: 허용 p95, 동시 요청/QPS, missed result 비용, 하드웨어·메모리,
  end-to-end 예산 등.

### 3.3 — 6점

각 1.5점: 단계 0.75 + 이유 0.75.

1. 분할: 완전한 근거가 한 chunk에 보존되지 않음.
2. 생성: 정답 근거를 검색했지만 날짜를 잘못 생성.
3. 검색: corpus의 정답 문서가 top-k에 없음.
4. 인용: 내용은 맞지만 citation이 지지하지 않음.

### 3.4 — 4점

타당한 설명 각 2점, 두 개:

- 무관한 context가 늘어 attention/선택을 방해.
- prompt token·latency·비용 증가.
- 중요한 근거가 긴 context 중간/후반에서 약화되는 position effect.
- 오래된/상충 근거가 함께 들어가 생성기가 잘못 조합.

## 문제 4 — 15점

### 4.1 — 5점

- 올바른 행동: $16+8=24$, 정확도 $24/30=0.8$ (3)
- unsupported answer rate: $2/30\approx0.0667$ (2)

### 4.2 — 4점

- threshold 증가 → 더 많이 거부, false refusal 증가 경향 (1.5)
- unsupported answer 감소 경향 (1.5)
- 의료/법률에서는 두 오류의 비용, 사람 검토, coverage, 위험 허용 수준으로 threshold 선택 (1)

### 4.3 — 3점

- correctness는 답 내용의 참/과업 적합성, groundedness는 제공 근거가 답을 지지하는지 측정 (1.5)
- 우연히 맞은 unsupported answer는 갱신·감사·신뢰 측면에서 위험하므로 별도 평가 (1.5)

### 4.4 — 3점

문제 두 개 각 1점: judge 편향/자기선호, prompt/model 민감성, 비결정성, 긴 근거 누락,
그럴듯함 선호. 검증 1점: 표본을 사람이 독립 평가하고 일치도·불일치 유형을 보고하거나
복수 judge와 사람 판정을 비교.

## 문제 5 — 15점

### 5.1 — 6점

- image encoder와 text encoder를 같은 공간에 정렬 (1.5)
- batch의 matching pair 유사도는 높이고 nonmatching은 낮추는 양방향 대조학습 (1.5)
- 클래스 이름을 prompt로 text embedding (1)
- 이미지 embedding과 클래스 text embedding similarity 계산 (1)
- 최고 similarity를 label로 선택, 별도 task classifier 학습 없음 (1)

### 5.2 — 4점

타당한 항목 1점씩, 최대 4점:

- 같은 이미지·정답·metric으로 한국어/영어/번역 prompt 비교
- 짧은 클래스명/문장형 prompt/template ensemble
- 고유명사·지역어·한영 혼합·추상/구체 질의 slice
- 번역 모델 자체의 영향을 별도 조건으로 분리
- 모델 선택 dev와 최종 test 분리
- zero-shot 기준선과 latency 포함

### 5.3 — 5점

- single vector는 전체 문서를 압축해 저장·검색이 단순/저렴 (1.5)
- multi-vector는 query token과 image patch별 최대/late interaction으로 작은 지역 증거 보존 (1.5)
- 표의 작은 단어처럼 국소 patch와 질의 token이 직접 대응해 유리 (1)
- token/patch 수만큼 저장·점수 계산·인덱스 비용 증가 (1)

## 문제 6 — 15점

### 6.1 — 5점

- 긴 벡터의 여러 prefix 길이에도 학습 손실을 줘 각 앞부분이 유용하도록 최적화 (2)
- 일반 embedding을 사후 절단하는 것과 달리 prefix 품질을 학습 단계에서 보장하려 함 (1)
- 짧은 prefix로 넓게 후보 생성, 긴 prefix로 재평가 (1)
- 정확도–메모리–latency trade-off를 동적으로 선택 (1)

### 6.2 — 5점

타당한 항목 1점씩, 최대 5점: 데이터/규모/라이선스, split/qrels, 모델 ID/revision,
pooling/normalization/similarity, query prefix, metric 정의/k, seed, library version, hardware,
latency 측정법, confidence interval, dev 선택 과정, 질의별 결과.

### 6.3 — 5점

- 새 명칭·시점 변화는 입력 분포와 corpus coverage 변화일 수 있어 코드 결함과 구별 필요 (1)
- drift 지표 두 개, 각 1점: OOV/token fertility, embedding distribution distance, 새 entity 비율,
  no-result rate, slice별 Recall/MRR, query length/category shift 등.
- 대응 실험 두 개, 각 1점: 최신 문서 재인덱싱, qrels 갱신 평가, hybrid/사전 확장,
  domain fine-tuning, drift 전후 slice 비교, shadow test.

## 권장 피드백 코드

|코드|의미|
|---|---|
|F1|2D 시각화와 원공간 성능을 동일시|
|F2|Precision/Recall의 분모를 혼동|
|F3|ANN 근사 손실과 모델 retrieval 오류를 구분하지 않음|
|F4|RAG 검색·생성·인용 오류를 한 단계로 취급|
|F5|정답성과 groundedness를 동일시|
|F6|정확도만 보고 latency·비용·위험을 누락|
|F7|test를 선택에 재사용|

## 변형 출제 지침

- 문제 2의 relevance 배열을 `[0,2,1,0,2]`처럼 바꾸면 MRR도 함께 달라진다.
- 문제 4의 answerable/unanswerable 수를 바꾸되 네 confusion 항목이 모두 존재하게 한다.
- RAG 사례의 최초 오류 단계를 섞어 A/B형을 만든다.
- 멀티모달 도메인을 관광에서 안전장비·의료·위성영상으로 바꾼다.
- Matryoshka 대신 multi-vector 또는 multilingual alignment의 비용–품질 문제를 출제한다.

# Marp 강의 슬라이드

`week01_*.md`부터 `week10_*.md`까지는 16:9 한국어 강의용 Marp 덱입니다. 각 덱은
용어와 정의, 핵심 수식, Hands-On Large Language Models 원본 Notebook 연결, 논문 도판,
최신 연구 동향, 수업 활동과 참고문헌을 포함합니다.

## 미리보기와 PDF 출력

VS Code의 **Marp for VS Code** 확장을 설치하고 Markdown 미리보기를 열거나, 저장소
루트에서 Marp CLI를 실행합니다.

```bash
# 한 주차 미리보기 서버
npx @marp-team/marp-cli@latest \
  --theme-set courses/embedding-data-science/slides/theme/embedding-course.css \
  --html --server courses/embedding-data-science/slides

# 전체 PDF 출력
mkdir -p courses/embedding-data-science/slides/dist
for f in courses/embedding-data-science/slides/week*.md; do
  npx @marp-team/marp-cli@latest \
    --theme-set courses/embedding-data-science/slides/theme/embedding-course.css \
    --html --pdf "$f" -o "courses/embedding-data-science/slides/dist/$(basename "${f%.md}").pdf"
done
```

본문 레이아웃은 HTML의 `<div class="...">` 없이 Markdown 제목·목록·표·인용문과
Marp 이미지 크기 문법으로 작성했습니다. 대표 논문 도판은 오프라인에서도 렌더링되도록
`assets/papers/`에 보관하며, 각 슬라이드와 [도판 출처표](assets/papers/README.md)에 원문
링크와 Figure 번호를 표시합니다. 그림은 강의·비평을 위한 인용이므로 배포 시 원 논문의
라이선스 조건을 함께 확인하세요.

## 권장 운영

- 한 덱 전체는 100–120분 분량입니다. 3시간 수업에서는 중간의 `Hands-On LLM 연결`과
  `수업 활동` 뒤에 Notebook 실습을 배치합니다.
- 슬라이드의 수치나 모델 순위는 영구적 사실이 아닙니다. 학기 시작 전 참고문헌의
  최신 버전과 MTEB/MMTEB 리더보드를 다시 확인하세요.
- PDF 배포본에는 발표자 노트가 보이지 않습니다. Markdown 원본의 HTML 주석을
  교수자용 설명으로 사용하세요.

## 파일 구성

| 파일 | 핵심 주제 |
|---|---|
| `week01_embedding_foundations.md` | 표현, 벡터 공간, 거리와 유사도 |
| `week02_tokenization_sparse.md` | 토큰화, TF–IDF, BM25, 희소·혼합 검색 |
| `week03_word_embeddings.md` | Word2Vec, GloVe, fastText |
| `week04_contextual_transformer.md` | 문맥 임베딩, self-attention, Transformer |
| `week05_sentence_embeddings.md` | SBERT, 대조학습, 파인튜닝 |
| `week06_embedding_analytics.md` | 분류, 군집, 토픽 모델링, 시각화 |
| `week07_evaluation.md` | STS·검색 지표, MTEB/MMTEB, 시스템 평가 |
| `week08_vector_search_rag.md` | ANN, 재순위화, RAG, ColBERT |
| `week09_multimodal.md` | CLIP, 문서 이미지 검색, 옴니모달 임베딩 |
| `week10_capstone_trends.md` | 종합 설계, 재현성, 2026 기술 동향 |

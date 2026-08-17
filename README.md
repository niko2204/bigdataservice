# Big Data Service · 강의자료 허브

국립목포대학교 컴퓨터학부 4학년 수업을 위한 공개 강의자료 저장소입니다. 현재 과정인
**임베딩 기반 데이터 과학**과 기존 **지역 상권 빅데이터 서비스 프로젝트**를 분리해
보관합니다.

## 현재 강의: 임베딩 기반 데이터 과학

[10주 강의 패키지 열기](courses/embedding-data-science/README.md)

- Marp 한국어 슬라이드 10개: 용어·정의·수식·핵심 논문 도판·2024–2026 기술 동향
- 주차별 강의노트와 Jupyter Notebook 10개
- 개인 과제 2개, 최종 프로젝트, 평가 루브릭, 교수자 가이드
- Hands-On Large Language Models의 관련 원본 장 2·3·4·5·8·9·10 선택본

```bash
pip install -r courses/embedding-data-science/requirements.txt
jupyter lab courses/embedding-data-science/notebooks
```

Marp 사용법은 [슬라이드 안내](courses/embedding-data-science/slides/README.md)를
참고하세요.

## 기존 자료

기존 지역 생활인구·상권 빅데이터 기반 창업 입지 추천 서비스 과정은 자료와 실행 구조를
그대로 보존해 [archive/original-bigdata-service](archive/original-bigdata-service/README.md)로
이동했습니다.

```bash
cd archive/original-bigdata-service
pip install -r requirements.txt
streamlit run app/Home.py
```

## 담당 교수

- 이영호 교수
- 국립목포대학교 컴퓨터학부
- youngho@ce.mokpo.ac.kr

## 라이선스와 출처

저장소 루트의 라이선스와 각 과정 폴더의 라이선스·NOTICE를 확인하세요. Hands-On Large
Language Models 선택본과 이를 바탕으로 재구성한 강의자료에는 원본 Apache License 2.0
고지를 유지합니다. 논문 도판은 각 슬라이드에 원문 링크와 캡션을 표시했습니다.

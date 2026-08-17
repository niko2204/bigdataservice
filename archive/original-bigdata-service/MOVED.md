# 기존 과정 이동 안내

이 폴더는 `bigdataservice` 저장소 루트에 있던 기존 지역 상권 빅데이터 서비스 과정의
전체 파일을 보존한 위치입니다. 상대 경로 기반 코드는 이 폴더를 작업 디렉터리로 두고
실행하세요.

```bash
cd archive/original-bigdata-service
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/Home.py
```

기존 Colab 링크는 이동된 경로를 가리키도록 README에서 갱신했습니다.

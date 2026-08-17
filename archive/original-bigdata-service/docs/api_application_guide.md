# 공공데이터 API 신청과 보안

1. 공공데이터포털 회원가입 후 원하는 API의 `활용신청`을 선택한다.
2. 개발 계정 인증키를 발급받는다.
3. `.env.example`을 `.env`로 복사하고 키를 입력한다.
4. `.env`가 `.gitignore`에 포함되었는지 확인한다.
5. 샘플 10건으로 응답 구조와 오류 코드를 먼저 검사한다.

```python
from src.data_collection.api_client import fetch_stores_by_radius

result = fetch_stores_by_radius(126.3922, 34.8118, radius=1000)
print(result)
```

인증키, 개인정보, 원본 대용량 파일은 GitHub에 올리지 않는다.


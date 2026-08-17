# Course 03 학습노트: 공공데이터 API 수집

## 1. 학습목표

권장 학습시간은 HTTP·API 이론 60분, 샘플 호출 60분, 수집기 구현 120분이다. 학습 후 요청·응답 구조, 인증, 페이지네이션, 재시도, 수집 기록을 포함한 재현 가능한 수집기를 작성할 수 있어야 한다.

## 2. REST 요청의 구조

API 요청은 보통 다음 요소로 구성된다.

- Endpoint: 기능을 나타내는 URL
- Method: 조회는 주로 GET
- Query parameter: 지역, 기간, 페이지, 건수
- Header 또는 key: 인증 정보
- Response: JSON 또는 XML 본문과 상태 코드

```python
import requests

params = {
    "cx": 126.3922,
    "cy": 34.8118,
    "radius": 1000,
    "pageNo": 1,
    "numOfRows": 10,
    "type": "json",
}
response = requests.get(BASE_URL, params=params, timeout=20)
print(response.status_code)
print(response.url.replace(API_KEY, "***"))
```

HTTP 200은 통신 성공을 뜻할 뿐 데이터 성공을 보장하지 않는다. 공공 API는 200 응답 안에 오류 코드나 빈 항목을 넣기도 하므로 본문의 결과 코드와 건수를 확인한다.

## 3. 인증키 관리

`.env`에 키를 보관하고 저장소에는 `.env.example`만 공유한다.

```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DATA_GO_KR_API_KEY")
if not API_KEY:
    raise RuntimeError("DATA_GO_KR_API_KEY가 설정되지 않았습니다.")
```

키를 출력하거나 Notebook 출력 셀에 남기지 않는다. 실수로 커밋했다면 파일만 지우는 것으로 충분하지 않으며 키를 폐기·재발급해야 한다.

## 4. JSON 구조 탐색

응답 전체를 바로 DataFrame으로 변환하지 말고 단계별로 구조를 확인한다.

```python
payload = response.json()
print(payload.keys())
body = payload["body"]
items = body.get("items", [])
print("전체 건수:", body.get("totalCount"))
print("현재 건수:", len(items))
```

실제 API마다 `response/body/items/item`처럼 중첩 구조가 다르다. 활용가이드의 필드 정의와 실제 응답을 함께 확인한다.

## 5. 페이지네이션

API가 한 번에 100건만 제공하고 전체가 530건이면 6페이지를 요청해야 한다.

```python
import math

all_items = []
page = 1
while True:
    params["pageNo"] = page
    payload = requests.get(BASE_URL, params=params, timeout=20).json()
    body = payload["body"]
    batch = body.get("items", [])
    all_items.extend(batch)

    total = int(body.get("totalCount", 0))
    if len(all_items) >= total or not batch:
        break
    page += 1
```

종료 조건을 잘못 쓰면 무한 반복이나 누락이 발생한다. 페이지 번호, 누적 건수, 전체 건수를 로그로 남긴다.

## 6. 안정적인 수집기

다음 실패를 예상해야 한다.

| 실패 | 처리 |
|---|---|
| timeout | 제한된 횟수로 재시도 |
| 429 | 호출 간격을 늘리고 제한 확인 |
| 500대 오류 | 잠시 후 재시도, 계속되면 중단 |
| JSON 파싱 오류 | 응답 일부와 Content-Type 기록 |
| 빈 데이터 | 지역·기간 파라미터와 전체 건수 확인 |

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry))
```

## 7. 원본 보존과 메타데이터

원본은 가공하지 않고 수집 시각을 포함하여 저장한다. 별도 메타데이터에는 출처 URL, API 버전, 파라미터, 기준일, 수집일, 건수, 라이선스를 기록한다.

```python
from datetime import datetime, timezone
from pathlib import Path
import json

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
path = Path("data/raw") / f"stores_{timestamp}.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

## 8. 단계별 실습

1. 샘플 10건을 요청하고 상태 코드·본문 결과 코드를 확인한다.
2. 첫 항목의 필드와 활용가이드 정의를 대조한다.
3. 페이지네이션을 추가하여 100건 이상 수집한다.
4. 중간 실패를 가정하고 timeout·재시도를 추가한다.
5. 원본 JSON과 정제 CSV를 별도 저장한다.
6. 두 파일의 행 수와 식별자 유일성을 검증한다.

## 9. 자가점검 질문

- 같은 파라미터로 다시 실행하면 같은 범위의 데이터가 수집되는가?
- 빈 페이지가 나왔을 때 정상 종료하는가?
- 키가 코드, URL 출력, Notebook 출력에 남지 않는가?
- 원본과 정제 파일을 구분할 수 있는가?
- 데이터 기준일과 수집일을 구분해 기록했는가?


"""공공데이터포털 API 공통 클라이언트 예제."""
from __future__ import annotations

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def fetch_json(base_url: str, params: dict, timeout: int = 20) -> dict:
    api_key = os.getenv("DATA_GO_KR_API_KEY")
    if not api_key:
        raise RuntimeError(".env에 DATA_GO_KR_API_KEY를 설정하세요.")
    query = {**params, "serviceKey": api_key, "type": "json"}
    response = requests.get(base_url, params=query, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_stores_by_radius(cx: float, cy: float, radius: int = 1000, num_rows: int = 100) -> dict:
    """소상공인시장진흥공단 반경 내 상가 조회 예제."""
    url = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"
    return fetch_json(url, {
        "cx": cx, "cy": cy, "radius": radius,
        "pageNo": 1, "numOfRows": num_rows,
    })


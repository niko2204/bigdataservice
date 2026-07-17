# Course 08 학습노트: 분석 서비스를 위한 데이터베이스

## 1. 학습목표

권장 학습시간은 관계형 모델 60분, SQL 90분, SQLite 연동 90분이다. 학습 후 관측 단위에 맞는 스키마를 설계하고 제약조건과 SQL로 데이터 품질을 보장하며 Python 서비스와 연동할 수 있어야 한다.

## 2. 파일과 데이터베이스

CSV는 교환과 단순 분석에 편리하지만 자료형·관계·동시성 제약이 약하다. 관계형 데이터베이스는 스키마, 제약조건, 트랜잭션과 반복 질의를 제공한다.

## 3. 테이블과 키

한 테이블의 한 행 의미를 명확히 한다.

- `regions`: 행정동 한 행
- `observations`: 행정동·기준월 한 행
- `stores`: 점포 한 행
- `scores`: 분석실행·업종·행정동 한 행

기본키는 행을 유일하게 식별하고, 외래키는 테이블 간 참조 무결성을 보장한다.

```sql
CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    region_name TEXT NOT NULL UNIQUE,
    latitude REAL CHECK(latitude BETWEEN 33 AND 39),
    longitude REAL CHECK(longitude BETWEEN 124 AND 132)
);

CREATE TABLE observations (
    region_id INTEGER NOT NULL,
    reference_month TEXT NOT NULL,
    population INTEGER CHECK(population >= 0),
    floating_population INTEGER CHECK(floating_population >= 0),
    PRIMARY KEY(region_id, reference_month),
    FOREIGN KEY(region_id) REFERENCES regions(region_id)
);
```

## 4. 정규화

반복되는 행정동 이름을 모든 테이블에 저장하면 오탈자와 갱신 불일치가 생긴다. 개체를 분리하고 키로 연결한다. 다만 분석 성능을 위해 집계 테이블을 별도로 두는 비정규화가 필요할 수 있으며 목적과 갱신 정책을 기록한다.

## 5. SQL 기본 질의

```sql
SELECT region_name, floating_population, cafe_count,
       ROUND(CAST(floating_population AS REAL) / NULLIF(cafe_count, 0), 1) AS people_per_cafe
FROM location_features
ORDER BY people_per_cafe DESC
LIMIT 5;
```

`NULLIF(cafe_count, 0)`은 0으로 나누는 오류를 방지한다.

그룹 집계 예시:

```sql
SELECT business_type,
       COUNT(*) AS n,
       ROUND(AVG(monthly_sales), 1) AS avg_sales
FROM stores
GROUP BY business_type
HAVING COUNT(*) >= 5;
```

## 6. Python 연동과 파라미터

```python
import sqlite3
import pandas as pd

with sqlite3.connect("database/bigdataservice.db") as conn:
    query = "SELECT * FROM location_features WHERE floating_population >= ?"
    result = pd.read_sql_query(query, conn, params=[15000])
```

문자열 조합으로 사용자 입력을 SQL에 넣지 말고 파라미터 바인딩을 사용한다.

## 7. 트랜잭션과 재실행

여러 적재 단계가 일부만 성공하면 데이터가 불일치할 수 있다. 트랜잭션 안에서 전체가 성공할 때만 커밋한다. 반복 실행할 때 중복이 생기지 않도록 `PRIMARY KEY`, upsert 또는 적재 전 교체 정책을 사용한다.

## 8. 인덱스

자주 필터·조인하는 열에 인덱스를 만들면 조회가 빨라지지만 저장 공간과 쓰기 비용이 증가한다.

```sql
CREATE INDEX idx_observations_month ON observations(reference_month);
```

작은 교육용 데이터에서는 차이가 작으므로 `EXPLAIN QUERY PLAN`으로 동작을 확인한다.

## 9. 단계별 실습

1. 프로젝트 테이블의 한 행 의미를 문장으로 쓴다.
2. 기본키·외래키가 포함된 ERD를 작성한다.
3. `schema.sql`로 DB를 생성한다.
4. CSV를 적재하고 원본 행 수와 DB 행 수를 비교한다.
5. 필터, 집계, JOIN을 사용하는 질의 3개를 작성한다.
6. Python에서 파라미터 질의를 실행한다.
7. 같은 적재를 두 번 실행해 중복 여부를 확인한다.

## 10. 완료 체크

- [ ] 모든 테이블의 관측 단위가 명확하다.
- [ ] 키와 제약조건이 데이터 규칙을 반영한다.
- [ ] 0 나눗셈과 NULL을 처리한다.
- [ ] 사용자 입력은 파라미터로 전달한다.
- [ ] 적재가 반복 실행 가능하다.


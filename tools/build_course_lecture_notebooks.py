"""10개 Course의 lecture_note.md를 학습용 Jupyter Notebook으로 변환한다."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSES = ROOT / "courses"

SETUP_CODE = """
from pathlib import Path
import os, sys, subprocess

REPO_URL = "https://github.com/niko2204/bigdataservice.git"
if "google.colab" in sys.modules:
    ROOT = Path("/content/bigdataservice")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)
else:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    ROOT = next((p.resolve() for p in candidates if (p / "courses").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("bigdataservice 저장소 안에서 Notebook을 실행하세요.")

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("학습 저장소:", ROOT)
"""


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code_cell(text: str, language: str = "python") -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["lecture-example", language]},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def split_markdown(source: str) -> list[dict]:
    """Markdown 문서의 Python fence만 코드 셀로 분리한다."""
    lines = source.splitlines(keepends=True)
    cells: list[dict] = []
    prose: list[str] = []
    code_lines: list[str] = []
    in_fence = False
    language = ""

    def flush_prose() -> None:
        text = "".join(prose).strip()
        if text:
            cells.append(markdown_cell(text))
        prose.clear()

    for line in lines:
        fence = re.match(r"^```([A-Za-z0-9_+-]*)\s*$", line.strip())
        if fence and not in_fence:
            flush_prose()
            in_fence = True
            language = fence.group(1).lower()
            code_lines = []
            continue
        if line.strip() == "```" and in_fence:
            block = "".join(code_lines).strip()
            if language in {"python", "py"}:
                cells.append(code_cell(block, "python"))
            else:
                rendered = f"```{language}\n{block}\n```"
                cells.append(markdown_cell(rendered))
            in_fence = False
            language = ""
            code_lines = []
            continue
        if in_fence:
            code_lines.append(line)
        else:
            prose.append(line)

    if in_fence:
        prose.extend([f"```{language}\n", *code_lines])
    flush_prose()
    return cells


def build_one(course_dir: Path) -> Path:
    source_path = course_dir / "lecture_note.md"
    source = source_path.read_text(encoding="utf-8")
    cells = [
        markdown_cell(
            """
> **Jupyter 학습 안내**
>
> 이 Notebook은 Course의 상세 학습노트입니다. 본문과 수식을 먼저 읽고 Python 예제 셀을 한 단계씩 실행하세요.
> 코드 셀은 개념을 보여주는 작은 예제로, 필요한 데이터와 변수는 바로 앞 설명을 확인해야 합니다.
> 처음부터 끝까지 실행하는 통합 실습은 저장소의 notebooks와 notebooks/data_analysis를 사용합니다.
> 예제 결과를 예상한 뒤 실행하고, 값·조건·열 이름을 바꾸어 결과 차이를 기록하세요.
            """
        ),
        code_cell(SETUP_CODE, "python"),
        *split_markdown(source),
        markdown_cell(
            """
## 학습 마무리

1. 이 Course의 핵심 개념 세 가지를 본인의 말로 정리한다.
2. 수식 하나를 작은 숫자로 손계산하고 Python 결과와 비교한다.
3. 예제 코드의 입력이나 조건을 하나 바꾸어 결과 차이를 설명한다.
4. quiz.md에 먼저 답한 뒤 quiz_answer.md와 비교한다.
5. assignment.md와 연결된 실습 Notebook을 Restart & Run All로 확인한다.
            """
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {
                "name": f"{course_dir.name}_lecture_note.ipynb",
                "provenance": [],
            },
            "source_markdown": str(source_path.relative_to(ROOT)),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = course_dir / "lecture_note.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def main() -> None:
    course_dirs = sorted(p for p in COURSES.glob("course*") if (p / "lecture_note.md").exists())
    if len(course_dirs) != 10:
        raise RuntimeError(f"Course 학습노트가 10개여야 하지만 {len(course_dirs)}개를 찾았습니다.")
    for course_dir in course_dirs:
        target = build_one(course_dir)
        print(target.relative_to(ROOT))
    print("Course Lecture Notebook 10개 생성 완료")


if __name__ == "__main__":
    main()

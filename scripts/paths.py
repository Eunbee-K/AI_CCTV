"""데이터·결과 폴더 위치를 한 곳에서 정한다.

이 저장소는 이동식 드라이브에 있고, **연결한 PC마다 드라이브 문자가 바뀐다**
(원래 PC에서는 E:). 스크립트마다 `Path(r"E:/AI_CCTV_DATASET")`처럼 박아두면
다른 PC에서 전부 깨지므로, 이 파일이 놓인 위치에서 드라이브를 되짚어 쓴다.

환경변수로 덮어쓸 수 있다 — 워크스테이션 로컬 사본(`/home/workstation/ai_cctv/`)
처럼 드라이브 개념이 없는 곳에서 쓴다.

    AI_CCTV_DATASET_DIR   기본: <드라이브>/AI_CCTV_DATASET
    AI_CCTV_RESULTS_DIR   기본: <드라이브>/AI_CCTV_RESULTS

쓰는 법:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/
    from paths import DATASET, RESULTS, FIELD_VIDEO
"""

from __future__ import annotations

import os
from pathlib import Path

# scripts/paths.py -> scripts -> 저장소 루트 -> 드라이브
REPO = Path(__file__).resolve().parents[1]
DRIVE = Path(REPO.anchor)


def _dir(env: str, name: str) -> Path:
    v = os.getenv(env, "").strip()
    return Path(v) if v else DRIVE / name


DATASET = _dir("AI_CCTV_DATASET_DIR", "AI_CCTV_DATASET")
RESULTS = _dir("AI_CCTV_RESULTS_DIR", "AI_CCTV_RESULTS")

# Stage-1 필터 관련 데이터는 여기 모아둔다(2026-08-14 정리).
# clsdata_*, valset_*, testset_*, yajang_frames, 평가 캐시 등. 데이터셋 루트에
# 흩어져 있으면 검출·세그멘테이션 자료와 섞여 뭐가 뭔지 알 수 없다.
FILTER_DATA = DATASET / "filter"

# 현장 검증용 영상 + 한글 조사표. 원래 바탕화면에 있었는데 외장하드만 들고
# 다니려고 여기로 옮겼다(2026-08-12).
FIELD_VIDEO = DATASET / "video" / "상면 CCTV"
SANGMYEON_0128 = FIELD_VIDEO / "20260128.가평군 상면 노후하수관로 정비공사"
SANGMYEON_0131 = FIELD_VIDEO / "20260131.가평군 상면 노후하수관로 정비공사"

# ver_2.2 앱과 그 학습데이터(LLM 판독 예시 사진)
LEGACY_CCTV = DRIVE / "CCTV"
LLM_EXAMPLES = LEGACY_CCTV / "ver.2.0_학습데이터"


def check(*paths: Path) -> list[Path]:
    """없는 경로만 돌려준다. 스크립트 첫머리에서 확인용으로 쓴다."""
    return [p for p in paths if not p.exists()]

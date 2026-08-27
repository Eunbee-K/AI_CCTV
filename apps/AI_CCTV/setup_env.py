"""처음 실행할 때 필요한 패키지를 자동으로 깔아주는 준비 스크립트.

제출본을 받은 PC에는 파이썬만 있고 fastapi도 ultralytics도 없다. 예전
`실행하기.bat`은 곧바로 `serve_demo.py`를 불러서, 새 PC에서는 `ModuleNotFoundError`
한 줄만 뜨고 끝났다. 이 파일이 그 사이를 메운다.

**전용 가상환경(`.venv`)에 깐다.** 받는 사람 PC의 파이썬을 건드리지 않는 것이
목적이다 — 심사용으로 잠깐 돌려보는 프로그램이 남의 개발환경 패키지 버전을
바꿔놓으면 안 된다. 폴더째 지우면 흔적 없이 사라진다.

설치를 세 단계로 나눈다. 뒤로 갈수록 무겁고, 실패해도 잃는 것이 적다.

  1. 기본(core)   — 화면이 뜨는 데 필요. 실패하면 실행을 멈춘다.
  2. 판독기(detector) — 분석에 필요. torch는 **CPU 판**으로 받는다(아래 참고).
  3. 글자판독(ocr)   — 거리·관경 자동 판독. 실패해도 그냥 진행한다.

torch를 그냥 `pip install`하면 CUDA가 딸린 2.5GB짜리를 받는다. 이 앱은 CPU로만
돌므로 순수 낭비이고, 회선이 느린 곳에서는 사실상 설치가 끝나지 않는다.
`--index-url .../whl/cpu`를 붙여 200MB대 CPU 판을 받는다.

`.venv/.setup_done`에 성공 표시를 남겨서 두 번째 실행부터는 건너뛴다.

사용:
  python setup_env.py            # 준비만 (설치 후 종료)
  python setup_env.py --recheck  # 표시를 무시하고 다시 점검
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
MARKER = VENV / ".setup_done"

# 이 표시가 바뀌면 기존 .venv가 있어도 설치를 다시 점검한다.
# 필요한 패키지 목록을 고칠 때 숫자를 올린다.
SETUP_VERSION = "2"

# ── 설치 묶음 ────────────────────────────────────────────────────────────
#
# **위·아래 한계를 모두 둔다.** 하한만 두고 돌려보니 opencv 5.0, numpy 2.5처럼
# 이 앱으로 한 번도 검증한 적 없는 판이 깔렸다. 제출본은 심사위원이 어느 날
# 열어도 그대로 떠야 하므로, 검증한 범위(다음 메이저 판 직전)까지로 묶는다.
# 하한은 기능이 필요한 최소치, 상한은 호환이 깨질 수 있는 다음 메이저 판이다.
CORE = [
    "fastapi>=0.110,<1",
    "uvicorn[standard]>=0.29,<1",
    "websockets>=12.0",
    "requests>=2.30,<3",
    "itsdangerous>=2.0,<3",
    "python-multipart>=0.0.9",
    "pymupdf>=1.24,<2",
    "openpyxl>=3.1,<4",
    "pillow>=11.0,<13",
    "numpy>=1.26,<3",
    "opencv-python>=4.10,<5",
    "onnxruntime>=1.18,<2",
]

# torch는 CPU 전용 저장소에서 먼저 받는다. 이걸 이미 깔아두면 다음 줄의
# ultralytics가 torch를 다시 받지 않는다(요구조건이 이미 충족되므로).
TORCH_CPU = ["torch", "torchvision", "--index-url",
             "https://download.pytorch.org/whl/cpu"]
DETECTOR = ["ultralytics>=8.3,<9"]

# paddlepaddle은 파이썬 새 판(3.14 등)에 아직 휠이 없을 수 있다. 실패해도
# 앱은 돈다 — 거리·관경을 손으로 넣게 될 뿐이다.
OCR = ["paddlepaddle>=3.0,<4", "paddleocr>=3.0,<4"]


def log(msg: str = "") -> None:
    """콘솔에 한 줄. 인코딩 때문에 죽지 않는다.

    배치파일이 띄우는 콘솔은 CP949다. 여기 출력은 영어로 둔다
    (serve_demo.py 기동 로그와 같은 이유 - 한글이 깨져 보이느니 영어가 낫다).

    그런데 영어만 써도 안심할 수 없다. em-dash 같은 문장부호 하나가 CP949로
    변환되지 않아 UnicodeEncodeError로 죽는다. 실제로 그 때문에 설치를 다 마친
    뒤 마지막 안내 줄에서 죽는 것을 확인했다. 설치 도구가 인쇄 때문에 죽는 것은
    말이 안 되므로, 못 쓰는 글자는 '?'로 바꿔서라도 진행한다.
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(args: list[str]) -> bool:
    """pip을 돌리고 성공 여부만 돌려준다. 출력은 그대로 흘려보낸다."""
    try:
        return subprocess.call(args) == 0
    except Exception as exc:  # pragma: no cover - 환경 문제
        log(f"        {exc}")
        return False


def ensure_venv() -> Path:
    """가상환경을 만들고 그 안의 python 경로를 준다."""
    py = venv_python()
    if py.exists():
        return py

    log("  Creating a private environment (.venv) ...")
    # --copies: 일부 윈도우 환경에서 심볼릭 링크 권한이 없어 venv 생성이 깨진다.
    ok = _run([sys.executable, "-m", "venv", "--copies", str(VENV)])
    if not ok or not py.exists():
        # venv 모듈이 없는 축소 설치판(마이크로소프트 스토어판 등)일 수 있다.
        log("  [ERROR] Could not create a virtual environment.")
        log("          Install Python from python.org (not the Store version).")
        raise SystemExit(1)
    return py


def pip_base(py: Path) -> list[str]:
    return [str(py), "-m", "pip", "install", "--disable-pip-version-check"]


def has_module(py: Path, mod: str) -> bool:
    return subprocess.call(
        [str(py), "-c", f"import {mod}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def install(py: Path, step: str, label: str, args: list[str],
            probe: str, required: bool) -> bool:
    """한 묶음을 깐다. 이미 있으면 건너뛴다."""
    if has_module(py, probe):
        log(f"  [{step}] {label} - already installed")
        return True

    log(f"  [{step}] Installing {label} ... (first run only)")
    ok = _run(pip_base(py) + args)
    # pip이 0을 줘도 실제로 import가 안 되는 경우가 있어(휠은 받았는데 의존성이
    # 깨진 경우) 반드시 import로 다시 확인한다.
    if ok and has_module(py, probe):
        log(f"  [{step}] {label} - done")
        return True

    if required:
        log(f"  [{step}] {label} - FAILED")
        return False
    log(f"  [{step}] {label} - skipped (optional, the app still runs)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recheck", action="store_true",
                    help="설치 완료 표시를 무시하고 다시 점검한다")
    args = ap.parse_args()

    if MARKER.exists() and not args.recheck:
        if MARKER.read_text(encoding="utf-8").strip() == SETUP_VERSION:
            return 0  # 준비 끝 — 조용히 통과

    log()
    log("  Preparing the environment. The first run downloads about 1 GB")
    log("  and takes 5-15 minutes depending on your connection.")
    log("  Later runs start immediately.")
    log()

    py = ensure_venv()

    # pip 자체가 낡으면 최신 휠의 메타데이터를 못 읽는 일이 있다.
    _run(pip_base(py) + ["--upgrade", "pip", "setuptools", "wheel"])

    if not install(py, "1/3", "core packages", CORE, "fastapi", True):
        log()
        log("  [ERROR] Could not install the core packages.")
        log("          Check your internet connection (a proxy or firewall may")
        log("          be blocking pypi.org) and run this file again.")
        return 1

    # 판독기: CPU torch 먼저, 그다음 ultralytics.
    if not has_module(py, "ultralytics"):
        log("  [2/3] Installing the detector (CPU build, ~300 MB) ...")
        if not _run(pip_base(py) + TORCH_CPU):
            # CPU 저장소를 막는 사내망일 수 있다. 일반 저장소로 한 번 더 시도한다
            # (용량은 크지만 되는 것이 낫다).
            log("        CPU build unavailable, trying the default index ...")
            _run(pip_base(py) + ["torch", "torchvision"])
    if not install(py, "2/3", "the detector", DETECTOR, "ultralytics", True):
        log()
        log("  [ERROR] Could not install the detector (ultralytics/torch).")
        log("          The app cannot analyse video without it.")
        return 1

    install(py, "3/3", "text reading (OCR)", OCR, "paddleocr", False)

    MARKER.write_text(SETUP_VERSION, encoding="utf-8")
    log()
    log("  Setup complete.")
    log()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log()
        log("  Interrupted. Run the file again to resume "
            "(finished downloads are cached).")
        sys.exit(1)

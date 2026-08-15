"""다중 클래스 분류기 — 프레임 하나에 **결함 이름**을 붙인다.

YOLO와 나란히 선다. YOLO는 "어디에 무엇이"를 풀고 이 모델은 "무엇이"만 푼다.
후자가 쉬운 문제라 이름 정확도가 더 높다 — 테스트셋(23종 455장)에서 이 모델
69.7%, YOLO 검출 45%였다. 대신 **박스가 없다.** 위치는 여전히 YOLO 몫이다.

둘을 바꾸지 않고 나란히 두는 이유는 서로 다른 것을 놓치기 때문이다.
    HL(천공)  분류기 65% / YOLO-cls 15%
    JS(이탈)  분류기 0%  — 이음부 계열은 사진만으로 갈라지지 않는다
한쪽이 못 잡는 것을 다른 쪽이 잡으면 검수자에게 후보가 하나 더 생긴다.

`defect_filter.py`와 같은 이유로 onnxruntime을 쓴다(앱에 torch를 안 깐다).
전처리도 같은 256 -> N 두 단계다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import (CLASSIFIER_BATCH_SIZE, CLASSIFIER_MIN_CONF,
                     CLASSIFIER_MODEL_PATH)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_SIZE_STORE = 256   # 학습 데이터가 저장된 크기 (build_multiclass 경유)

# 정상 계열 — 이름을 붙이지 않는다. train_multiclass.py의 NORMAL_CLASSES와 같아야 한다.
NORMAL_CLASSES = {"normal", "IN", "PJ", "OUT_MH", "OUT_CAR", "OUT_INVERT"}

_session = None
_classes: List[str] = []
_size = 224
_gray = False
_load_error: Optional[str] = None
_loaded = False


def _load():
    """세션을 한 번만 만든다. 실패는 기억해뒀다가 조용히 비활성화한다."""
    global _session, _classes, _size, _gray, _load_error, _loaded
    if _loaded:
        return _session
    _loaded = True

    path = Path(CLASSIFIER_MODEL_PATH)
    if not path.exists():
        _load_error = f"분류기 모델이 없습니다: {path}"
        return None

    # **클래스 이름표가 없으면 열지 않는다.** 인덱스가 어떤 코드인지 모르면
    # 이름을 붙일 수 없고, 짐작으로 붙이면 한 칸씩 밀린 채 조용히 틀린다
    # (D_yolo_mc 학습에서 train 24종 / val 18종이 어긋나 실제로 겪었다).
    side = path.with_suffix(".classes.json")
    if not side.exists():
        _load_error = f"클래스 이름표가 없습니다: {side}"
        return None
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
        _classes = list(meta["classes"])
        _size = int(meta.get("img", 224))
        _gray = bool(meta.get("gray", False))
    except Exception as exc:
        _load_error = f"클래스 이름표를 읽지 못했습니다: {exc}"
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        _load_error = "onnxruntime이 설치돼 있지 않습니다 (pip install onnxruntime)"
        return None

    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        _load_error = f"분류기 모델을 열지 못했습니다: {exc}"
        return None
    return _session


def availability() -> Tuple[bool, str]:
    """(쓸 수 있는가, 이유). UI·로그에서 상태를 보여줄 때 쓴다."""
    sess = _load()
    return (sess is not None, "" if sess is not None else (_load_error or "알 수 없음"))


def _preprocess(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((_SIZE_STORE, _SIZE_STORE), Image.BILINEAR)
            im = im.resize((_size, _size), Image.BILINEAR)
            if _gray:
                im = im.convert("L").convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None
    return (arr.transpose(2, 0, 1) - _MEAN) / _STD


def analyze(frames: List[Path]) -> Tuple[Dict[str, float], Dict[str, Tuple[str, float]]]:
    """한 번 돌려 **결함 확률과 이름을 함께** 얻는다.

    반환 (probs, names)
        probs  프레임 -> 결함 확률(= 1 - P(정상 계열)). 필터로 쓴다.
        names  프레임 -> (결함코드, 확률). 정상으로 본 프레임과 확신이 모자란
               프레임은 빠진다. 이름 붙이기에 쓴다.

    **두 일을 한 모델이 하는 것이 핵심이다.** 필터와 분류기가 따로면 "필터는
    결함이라는데 분류기는 정상이라 이름을 안 붙임" 같은 모순이 생긴다. 같은
    확률 분포에서 둘 다 나오면 그런 어긋남이 없다.

    테스트셋(655장) 실측 — 필터로서 AUC 0.9906, 이진 전용 모델(0.9920)과
    잡음(±0.008) 안에서 같다.
    """
    sess = _load()
    if sess is None or not frames:
        return {}, {}

    probs: Dict[str, float] = {}
    names: Dict[str, Tuple[str, float]] = {}
    normal_idx = [i for i, c in enumerate(_classes) if c in NORMAL_CLASSES]
    name = sess.get_inputs()[0].name
    batch: List[np.ndarray] = []
    keys: List[str] = []

    def flush():
        if not batch:
            return
        logits = sess.run(None, {name: np.stack(batch)})[0]
        m = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - m)
        p_all = exp / exp.sum(axis=1, keepdims=True)
        for k, p in zip(keys, p_all):
            probs[k] = float(1.0 - p[normal_idx].sum())
            j = int(p.argmax())
            code = _classes[j]
            if code not in NORMAL_CLASSES and float(p[j]) >= CLASSIFIER_MIN_CONF:
                names[k] = (code, float(p[j]))
        batch.clear()
        keys.clear()

    for fp in frames:
        arr = _preprocess(Path(fp))
        if arr is None:
            continue
        batch.append(arr)
        keys.append(str(fp))
        if len(batch) >= CLASSIFIER_BATCH_SIZE:
            flush()
    flush()
    return probs, names


def classify(frames: List[Path]) -> Dict[str, Tuple[str, float]]:
    """이름만 필요할 때. analyze()의 두 번째 값과 같다."""
    return analyze(frames)[1]

"""Stage-1 정상/결함 필터 — 프레임 하나가 결함인지 아닌지만 판정한다.

YOLO는 "결함이 어디 있는가"를 풀고, 이 모델은 "결함이 있는가"만 푼다. 후자가 훨씬 쉬운
문제라 정확도가 높고, 그래서 YOLO의 판단을 교차 검증하는 데 쓸 수 있다.

**onnxruntime을 쓰는 이유** — `requirements.txt`에 "Colab 원격 추론을 쓰면 로컬에
ultralytics/torch가 없어도 동작한다"고 적혀 있다. 필터 하나 때문에 그 전제를 깨면 설치
요구사항이 2GB 늘어난다. onnxruntime은 50MB 수준이고 CPU 추론도 더 빠르다.

**전처리는 학습과 정확히 같아야 한다.** 학습 데이터는 원본 프레임을 256x256으로 찌그러뜨려
저장했고(`build_binary_dataset.py`), 검증 변환이 그걸 다시 224로 줄였다
(`train_binary.py`의 `Resize(224)` + `CenterCrop(224)` — 정사각 입력이라 크롭은 무동작).
그래서 여기서도 256 → 224 두 단계를 그대로 밟는다. 한 번에 224로 줄이면 리샘플링 결과가
미묘하게 달라진다.

모델이 없거나 onnxruntime이 없으면 조용히 비활성화된다 — 필터는 보조 장치이므로
없다고 분석 자체가 멈추면 안 된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import FILTER_BATCH_SIZE, FILTER_MODEL_PATH

# ImageNet 통계. 사전학습 백본을 파인튜닝했으므로 학습 때와 같은 값을 써야 한다.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_SIZE_STORE = 256   # build_binary_dataset.py가 저장한 크기
_SIZE_INPUT = 224   # train_binary.py가 모델에 넣은 크기

_session = None
_load_error: Optional[str] = None
_loaded = False


def _load():
    """세션을 한 번만 만든다. 실패는 기억해뒀다가 조용히 비활성화한다."""
    global _session, _load_error, _loaded
    if _loaded:
        return _session
    _loaded = True

    path = Path(FILTER_MODEL_PATH)
    if not path.exists():
        _load_error = f"필터 모델이 없습니다: {path}"
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
        _load_error = f"필터 모델을 열지 못했습니다: {exc}"
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
            im = im.resize((_SIZE_INPUT, _SIZE_INPUT), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None
    return (arr.transpose(2, 0, 1) - _MEAN) / _STD


def defect_probs(frames: List[Path]) -> Dict[str, float]:
    """프레임 경로 -> 결함 확률(0~1). 모델이 없으면 빈 dict.

    읽지 못한 프레임은 결과에서 빠진다. 호출한 쪽은 '판정 없음'으로 다뤄야 한다 —
    임의로 0이나 1로 채우면 필터가 조용히 오판한 것처럼 보인다.
    """
    sess = _load()
    if sess is None or not frames:
        return {}

    out: Dict[str, float] = {}
    name = sess.get_inputs()[0].name
    batch: List[np.ndarray] = []
    keys: List[str] = []

    def flush():
        if not batch:
            return
        logits = sess.run(None, {name: np.stack(batch)})[0]
        # softmax. 인덱스 1이 결함(CLASSES = ["normal", "defect"])
        m = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - m)
        probs = exp[:, 1] / exp.sum(axis=1)
        for k, p in zip(keys, probs):
            out[k] = float(p)
        batch.clear()
        keys.clear()

    for fp in frames:
        arr = _preprocess(Path(fp))
        if arr is None:
            continue
        batch.append(arr)
        keys.append(str(fp))
        if len(batch) >= FILTER_BATCH_SIZE:
            flush()
    flush()
    return out

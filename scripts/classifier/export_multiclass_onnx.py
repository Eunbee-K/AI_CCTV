"""다중 클래스 분류기(C)를 앱이 쓸 ONNX로 내보낸다.

왜 ONNX인가
    앱은 onnxruntime(50MB)만 쓰고 torch/ultralytics(2GB)는 안 깔아도 돌아야 한다.
    `defect_filter.py` 주석에 그 전제가 적혀 있다. 이진 필터가 이미 그 길을
    쓰고 있으므로 다중도 같은 길로 간다.

클래스 이름을 같이 저장한다
    이진은 인덱스 1이 결함이라고 못 박으면 됐지만, 다중은 인덱스가 어떤 결함
    코드인지 알아야 이름을 붙일 수 있다. ONNX 파일 옆에 `<이름>.classes.json`으로
    같이 떨어뜨린다 — 모델과 이름표가 따로 놀면 D_yolo_mc에서 겪은 것처럼
    한 칸씩 밀린 채로 조용히 틀린다.

사용:
    python export_multiclass_onnx.py --ckpt <best_acc.pt 경로>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_multiclass import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2]
                                         / "apps/AI_CCTV/assets/classifier.onnx"))
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    size = ck.get("img", 224)
    gray = bool(ck.get("gray", False))
    print(f"체크포인트: {args.ckpt}")
    print(f"  {len(classes)}클래스 · img {size} · gray {gray}")

    net = build_model(ck.get("arch", "effb0"), len(classes))
    net.load_state_dict(ck["model"])
    net.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        net, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        # 배치는 가변이어야 한다 — 앱이 프레임을 묶음으로 넣는다
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )

    meta = {"classes": classes, "img": size, "gray": gray,
            "arch": ck.get("arch", "effb0"), "src": str(args.ckpt)}
    side = out.with_suffix(".classes.json")
    side.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    mb = out.stat().st_size / 1024 / 1024
    print(f"\n{out}  ({mb:.1f}MB)")
    print(f"{side}")

    # 내보낸 것이 원본과 같은 답을 내는지 확인한다. 여기서 안 맞으면 앱에서
    # 조용히 다른 답이 나온다.
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = torch.randn(3, 3, size, size)
        with torch.no_grad():
            a = torch.softmax(net(x), 1).numpy()
        b = sess.run(None, {"input": x.numpy()})[0]
        b = np.exp(b - b.max(1, keepdims=True))
        b = b / b.sum(1, keepdims=True)
        d = float(np.abs(a - b).max())
        print(f"\n원본과 최대 차이 {d:.2e}  {'OK' if d < 1e-4 else '!! 어긋남'}")
    except ImportError:
        print("\n(onnxruntime 없음 — 대조 검증 건너뜀)")


if __name__ == "__main__":
    main()

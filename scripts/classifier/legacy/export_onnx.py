"""학습 체크포인트(best.pt)를 앱이 쓸 ONNX로 내보낸다.

앱에 torch를 들이지 않기 위해서다. `requirements.txt`에 "Colab 원격 추론을 쓰면
로컬에 ultralytics/torch가 없어도 동작한다"고 적혀 있는데, 필터 하나 때문에 그 전제를
깨면 설치 요구사항이 2GB 늘어난다. onnxruntime은 50MB 수준이고 CPU 추론도 더 빠르다.

체크포인트에는 옵티마이저·스케줄러 상태가 함께 들어 있어 48MB다. 추론에 필요한 것은
가중치뿐이라 ONNX로 뽑으면 20MB 안쪽으로 줄어든다.

사용:
    python export_onnx.py --ckpt E:/AI_CCTV_RESULTS/filter/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_binary import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"E:/AI_CCTV_RESULTS/filter/best.pt")
    ap.add_argument("--out", default="")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch, img = ck.get("arch", "effb0"), ck.get("img", 224)
    gray = bool(ck.get("gray", False))
    print(f"체크포인트 epoch {ck.get('epoch')} | arch {arch} | img {img} | gray {gray}")
    if gray:
        # 흑백 학습본은 추론도 흑백이어야 한다. 전처리가 어긋나면 모델이 본 적
        # 없는 그림이 들어가고, 성능이 아니라 측정이 통째로 무너진다.
        print("  ** 흑백으로 학습된 모델이다. 앱에서 FILTER_GRAYSCALE=1 로 켤 것 **")

    model = build_model(arch)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.randn(1, 3, img, img)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        # 프레임 수가 매번 다르므로 배치 축을 열어둔다
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    size_mb = out.stat().st_size / 1e6
    print(f"내보냄: {out} ({size_mb:.1f}MB, 원본 {ckpt_path.stat().st_size/1e6:.1f}MB)")

    # torch 결과와 대조해 변환이 값을 바꾸지 않았는지 확인한다
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, img, img)
    with torch.no_grad():
        ref = torch.softmax(model(x), 1)[:, 1].numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    got = np.exp(got) / np.exp(got).sum(1, keepdims=True)
    got = got[:, 1]
    diff = float(np.abs(ref - got).max())
    print(f"torch 대비 최대 오차 {diff:.2e}  ->  {'일치' if diff < 1e-4 else '!! 확인 필요'}")
    print(f"샘플 확률 torch {ref.round(5)}")
    print(f"샘플 확률 onnx  {got.round(5)}")


if __name__ == "__main__":
    main()

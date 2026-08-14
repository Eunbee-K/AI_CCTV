"""YOLO11 분류기로 같은 데이터를 학습한다 — EfficientNet-B0과 비교하기 위해.

왜 비교하나
    지금 필터는 EfficientNet-B0인데, ImageNet 특징만으로 선형 분류기를 얹어도
    야장 AUC 0.866이 나왔다(학습 후 0.9503). 학습이 기여하는 몫이 작다는 뜻이라
    백본을 바꾸면 달라지는지 볼 값어치가 있다.

    **같은 조건으로 맞춘다** — 둘 다 ImageNet 사전학습에서 시작하고, 같은 데이터,
    같은 해상도, 흑백. 그래야 순수하게 구조 차이만 비교된다.

데이터 형식
    YOLO-cls는 `train/<클래스>/`, `val/<클래스>/` 구조를 그대로 쓴다.
    우리 clsdata_old_v5_384가 이미 그 모양이라 변환이 필요 없다.

흑백
    ultralytics에는 흑백 옵션이 없다. `hsv_s=0.0`으로 채도 증강을 끄는 것만으로는
    입력이 흑백이 되지 않으므로, **데이터셋을 미리 흑백으로 만들어 쓴다**
    (resize_dataset.py --gray). 그래야 EfficientNet 쪽과 같은 입력이 된다.

사용:
    python train_yolo_cls.py --data E:/AI_CCTV_DATASET/clsdata_old_v5_384_gray
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import DATASET, RESULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET / "clsdata_old_v5_384"))
    ap.add_argument("--model", default="yolo11n-cls.pt",
                    help="사전학습 가중치. 백지로 하려면 yolo11n-cls.yaml")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="")
    ap.add_argument("--project", default=str(RESULTS / "filter" / "yolo_cls"))
    args = ap.parse_args()

    from ultralytics import YOLO

    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=name,
        # 색 증강은 끈다 — 흑백 데이터셋이라 의미가 없고, EfficientNet 쪽
        # 학습(밝기·대비만)과 조건을 맞춘다.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5,
        # 수직 뒤집기는 쓰지 않는다 — 토사퇴적처럼 중력 방향에 의미가 있는 결함이 있다.
        flipud=0.0,
        degrees=0.0,
    )
    print(f"\n결과 -> {Path(args.project) / name}")
    print("  weights/best.pt · weights/last.pt")
    print("\n채점은 score_testset_bycode.py 로 한다 — 같은 판에서 EfficientNet과 비교.")


if __name__ == "__main__":
    main()

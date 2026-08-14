"""YOLO11 분류기 학습 — 같은 데이터로 EfficientNet-B0과 비교하기 위해.

두 가지에 쓴다
    **이진**       clsdata_old_v6            정상/결함 — 지금 필터와 같은 문제
    **다중 클래스** clsdata_old_v6_multiclass 결함 종류별 + 정상
                   bbox 없이 종류까지 맞힐 수 있는지 보는 실험. 지금은 YOLO 검출이
                   이름을 붙이는데 실영상에서 이름 정확도가 45%였다. 분류기가 그보다
                   나으면 검출 없이 조사표를 채울 수 있다.

    YOLO-cls는 `train/<클래스>/`, `val/<클래스>/` 구조를 그대로 읽고 클래스 수를
    자동으로 잡는다. 두 실험에 같은 스크립트를 쓸 수 있는 이유다.

조건 맞추기
    EfficientNet 쪽과 비교하려면 입력이 같아야 한다. ultralytics에는 흑백 옵션이
    없으므로 흑백으로 하려면 데이터셋을 미리 흑백으로 만든다
    (`resize_dataset.py --gray`).

사용:
    python train_yolo_cls.py --data E:/AI_CCTV_DATASET/clsdata_old_v6
    python train_yolo_cls.py --data E:/AI_CCTV_DATASET/clsdata_old_v6_multiclass \
        --name v6_multiclass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import DATASET, FILTER_DATA, RESULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FILTER_DATA / "clsdata_old_v6"))
    ap.add_argument("--model", default="yolo11n-cls.pt",
                    help="사전학습 가중치. 백지로 하려면 yolo11n-cls.yaml")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--name", default="")
    ap.add_argument("--project", default=str(RESULTS / "filter" / "yolo_cls"))
    args = ap.parse_args()

    data = Path(args.data)
    classes = sorted(d.name for d in (data / "train").iterdir() if d.is_dir())
    print(f"데이터 {data.name} · {len(classes)}클래스")
    print(f"  {', '.join(classes)}\n")

    from ultralytics import YOLO

    name = args.name or f"{Path(args.model).stem}_{data.name}_{args.imgsz}"
    model = YOLO(args.model)
    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=name,
        # 색 증강은 약하게 — 학습 데이터와 실영상의 색조가 계통적으로 다른데
        # (R-B −16~−27 vs −4~−8) 색에 기대는 것을 막고 싶다.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5,
        # 수직 뒤집기는 쓰지 않는다 — 토사퇴적처럼 중력 방향에 의미가 있는 결함이 있다.
        flipud=0.0,
        degrees=0.0,
    )
    print(f"\n결과 -> {Path(args.project) / name}/weights/")
    print("  best.pt · last.pt")


if __name__ == "__main__":
    main()

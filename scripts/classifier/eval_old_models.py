"""노후관로 필터 후보들을 야장·AIHub 양쪽으로 채점한다.

**best.pt를 그대로 믿으면 안 된다.** train_binary.py는 임계값 0.5에서의 F1이 가장 높은
epoch을 best.pt로 저장하는데, 우리가 원하는 것은 "결함을 95% 잡으면서 프레임을 얼마나
걸러내는가"다. 둘은 다른 지표다. 실제로 앞선 A~E 실험에서 F1으로는 C가 1등이었지만
AUC로는 D가 1등이었다 — F1 차이의 상당 부분이 임계값 착시였다.

그래서 best.pt와 last.pt를 모두 채점하고, F1이 아니라 **AUC와 제거율**로 고른다.

**두 데이터셋으로 잰다.**
  야장 val       실제 배포 도메인. 여기서 떨어지면 안 된다.
  AIHub holdout  학습에 넣지 않은 1,200장. v1은 정상 IN을 21.3%밖에 못 맞혔다.
                 한쪽을 얻고 다른 쪽을 잃는 건 아닌지 봐야 한다.

사용:
    python eval_old_models.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from paths import DATASET, RESULTS  # noqa: E402

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_binary import roc_auc
from train_binary import build_model

RUNS = RESULTS / "filter/runs"
MODELS = {
    "v1 best(ep9)": RUNS / "OLD_v1-20260811T135759Z-1-001/OLD_v1/best.pt",
    "v1 last(ep24)": RUNS / "OLD_v1-20260811T135759Z-1-001/OLD_v1/last.pt",
    "v2 best(ep3)": RUNS / "OLD_v2-20260811T152820Z-1-001/OLD_v2/best.pt",
    "v2 last(ep14)": RUNS / "OLD_v2-20260811T152820Z-1-001/OLD_v2/last.pt",
}
SETS = {
    "AIHub holdout": DATASET / "aihub_holdout/val",
    "야장 val": DATASET / "clsdata_old_v1/val",
}


class Simple(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def items_of(root: Path):
    out = []
    for y, name in enumerate(("normal", "defect")):
        out += [(f, y) for f in sorted((root / name).glob("*.jpg"))]
    return out


@torch.no_grad()
def predict(ckpt: Path, items, workers: int):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_model(ck.get("arch", "effb0"))
    m.load_state_dict(ck["model"])
    m.eval()
    img = ck.get("img", 224)
    tf = transforms.Compose([
        transforms.Resize(img), transforms.CenterCrop(img), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    for x, _ in DataLoader(Simple(items, tf), batch_size=64, num_workers=workers):
        out += torch.softmax(m(x).float(), 1)[:, 1].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    for sname, sroot in SETS.items():
        items = items_of(sroot)
        if not items:
            print(f"!! {sname}: 이미지 없음 ({sroot})")
            continue
        print(f"\n=== {sname} ({len(items):,}장) ===", flush=True)
        for mname, ckpt in MODELS.items():
            if not ckpt.exists():
                print(f"  {mname:<15} 체크포인트 없음")
                continue
            ps = predict(ckpt, items, args.workers)
            pos = [p for p, (_, y) in zip(ps, items) if y == 1]
            neg = [p for p, (_, y) in zip(ps, items) if y == 0]
            tp = sum(1 for p in pos if p >= 0.5)
            fp = sum(1 for p in neg if p >= 0.5)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(len(pos), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            # 필터로서의 실제 지표 — 결함 95%를 잡는 임계값에서 몇 %를 걸러내는가
            thr = sorted(pos)[int(0.05 * len(pos))]
            cut = 1 - sum(1 for p in ps if p >= thr) / len(ps)
            print(f"  {mname:<15} AUC {roc_auc(pos, neg):.4f}  F1 {f1:.4f}  "
                  f"결함95%에서 {cut:>5.1%} 제거", flush=True)
            if "AIHub" in sname:
                by = {}
                for (f, y), p in zip(items, ps):
                    c = f.stem.split("_")[1]
                    by.setdefault(c, [0, 0])
                    by[c][1] += 1
                    by[c][0] += int((p >= 0.5) == (y == 1))
                print("                  코드별: "
                      + "  ".join(f"{c} {h/t:.0%}" for c, (h, t) in sorted(by.items())),
                      flush=True)


if __name__ == "__main__":
    main()

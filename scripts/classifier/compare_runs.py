"""필터 실험들을 같은 현장 val로 다시 채점해 나란히 비교한다.

학습 로그의 F1은 임계값 0.5에서 잰 값이라 임계값 선택에 흔들린다. 어느 구성이
실제로 판별력이 좋은지는 **AUC**로 봐야 한다 — 임계값과 무관하고, 순위를 얼마나
잘 매기는지만 잰다.

각 run의 학습 해상도가 다르므로(224/384) 체크포인트에 저장된 img를 그대로 쓴다.
데이터는 현장 관로 5개(clsdata_v2/val)로 통일한다 — 모든 실험이 이 val을 공유했다.

사용:
    python compare_runs.py --runs E:/AI_CCTV_RESULTS/filter/runs
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_binary import roc_auc
from train_binary import build_model


def load_meta(root: Path) -> dict[str, dict]:
    """저장 경로 -> manifest 한 줄. 항목·등급별로 성능을 쪼개 보는 데 쓴다."""
    import csv as _csv
    f = root / "manifest.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8-sig") as fp:
        return {Path(r["dst"]).name: r for r in _csv.DictReader(fp)}


class Simple(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def collect(root: Path):
    out = []
    for label, name in enumerate(("normal", "defect")):
        for f in sorted((root / "val" / name).glob("*.jpg")):
            out.append((f, label))
    return out


@torch.no_grad()
def score(ckpt: Path, items, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    img = ck.get("img", 224)
    model = build_model(ck.get("arch", "effb0"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(img), transforms.CenterCrop(img), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ld = DataLoader(Simple(items, tf), batch_size=32, num_workers=2)
    ps, ys = [], []
    for x, y in ld:
        p = torch.softmax(model(x.to(device)).float(), 1)[:, 1]
        ps += p.cpu().tolist()
        ys += y.tolist()
    return ps, ys, img, ck.get("epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=r"E:/AI_CCTV_RESULTS/filter/runs")
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v2")
    args = ap.parse_args()

    items = collect(Path(args.data))
    n_pos = sum(1 for _, y in items if y == 1)
    print(f"현장 val {len(items)}장 (정상 {len(items)-n_pos} / 결함 {n_pos})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'run':<14}{'img':>5}{'AUC':>8}{'F1@.5':>8}{'정밀도':>8}{'재현율':>8}"
          f"{'  결함95%일때 정상통과율'}")
    rows = []
    for d in sorted(Path(args.runs).iterdir()):
        ck = d / "best.pt"
        if not ck.exists():
            continue
        ps, ys, img, ep = score(ck, items, device)
        pos = [p for p, y in zip(ps, ys) if y == 1]
        neg = [p for p, y in zip(ps, ys) if y == 0]
        auc = roc_auc(pos, neg)
        tp = sum(1 for p in pos if p >= 0.5)
        fp = sum(1 for p in neg if p >= 0.5)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(len(pos), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        # 결함을 95% 잡는 임계값에서 정상이 얼마나 새는가 = 필터로서의 쓸모
        thr = sorted(pos)[int(0.05 * len(pos))]
        leak = sum(1 for p in neg if p >= thr) / len(neg)
        print(f"{d.name:<14}{img:>5}{auc:>8.4f}{f1:>8.4f}{prec:>8.4f}{rec:>8.4f}{leak:>18.1%}")
        rows.append({"run": d.name, "img": img, "auc": auc, "f1": f1,
                     "precision": prec, "recall": rec, "leak_at_95": leak})

    # 가장 좋은 모델이 어느 항목에서 틀리는지 본다.
    # 등급(대/중/소)은 심각도일 뿐 정상/결함 구분이 아니다 — 야장에 적힌 것은
    # 소 등급이라도 결함이다. 다만 항목별로 보면 어떤 결함 유형이 약한지 드러난다.
    meta = load_meta(Path(args.data))
    if meta and rows:
        top = max(rows, key=lambda r: r["auc"])
        ps, _, _, _ = score(Path(args.runs) / top["run"] / "best.pt", items, device)
        agg = defaultdict(lambda: [0, 0])
        for (path, y), prob in zip(items, ps):
            m = meta.get(path.name)
            if not m or not m.get("name"):
                continue
            k = (m["name"], "결함" if y == 1 else "정상")
            agg[k][1] += 1
            agg[k][0] += int((prob >= 0.5) == (y == 1))
        if agg:
            print(f"\n=== {top['run']} 항목별 정답률 ===")
            for k in sorted(agg, key=lambda k: -agg[k][1]):
                hit, tot = agg[k]
                print(f"  {k[1]:<4} {k[0]:<22} {hit/tot:>6.1%}  ({tot}장)")

    best = max(rows, key=lambda r: r["auc"])
    print(f"\nAUC 최고: {best['run']} ({best['auc']:.4f})")
    print("참고 — v1(AIHub 전용) 모델의 현장 AUC는 0.516이었다(동전 던지기).")
    Path(args.runs, "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

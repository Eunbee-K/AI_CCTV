"""EfficientNet-B0로 **다중 클래스**(결함 종류별 + 정상)를 학습한다.

왜 따로 만드나
    `train_binary.py`는 2클래스 전용이다 — 헤드가 `Linear(..., 2)`로 고정이고,
    지표도 `probs[:, 1]`(결함 확률) 기준으로 짜여 있다. 거기에 다중 클래스를
    끼워 넣으면 이진 쪽이 망가질 위험이 크다.

무엇을 보려는 것인가
    지금 파이프라인은 필터가 위치를 잡고 YOLO 검출이 이름을 붙이는데, 실영상에서
    **이름 정확도가 45%**였다. 분류기가 bbox 없이 그보다 잘 맞히면 검출 모델 없이
    조사표를 채울 수 있고 라벨링 비용(bbox)도 사라진다.

    같은 데이터로 YOLO11-cls도 돌려 백본을 비교한다(train_yolo_cls.py).

지표
    다중 클래스라 AUC 대신 **정확도 / 클래스별 재현율 / 이진 환산 성능**을 본다.
    이진 환산 = "정상 클래스가 아닌 것으로 예측했나" — 지금 필터와 같은 문제로
    바꿔 재는 것이라 이진 모델과 직접 비교할 수 있다.

사용:
    python train_multiclass.py --data E:/AI_CCTV_DATASET/clsdata_old_v6_multiclass
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights,
                                convnext_tiny, efficientnet_b0)

NORMAL_CLASSES = {"normal", "IN", "PJ", "OUT_MH", "OUT_CAR", "OUT_INVERT"}


class FolderDataset(Dataset):
    """`<split>/<클래스>/*.jpg` 구조를 읽는다."""

    def __init__(self, root: Path, split: str, tf, classes: list[str]):
        self.tf = tf
        self.classes = classes
        self.samples: list[tuple[Path, int]] = []
        for i, name in enumerate(classes):
            d = root / split / name
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((f, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        # 외장드라이브는 동시 접근에서 간헐적으로 OSError(22)를 낸다. 파일은 멀쩡하다.
        for attempt in range(5):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                return self.tf(Image.open(io.BytesIO(data)).convert("RGB")), label
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))


def build_model(arch: str, n_class: int):
    if arch == "effb0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_class)
    elif arch == "convnext_t":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_class)
    else:
        raise ValueError(arch)
    return m


def make_sampler(samples, n_class: int):
    """클래스가 고르지 않아도 손실에 고르게 반영되도록 가중 샘플링.

    데이터를 종류당 400장으로 맞췄지만 원본이 부족한 종류(CX 16 · NS 4)가 남아
    있고, 정상은 결함 전체와 1:1이라 클래스 하나로 보면 훨씬 크다.
    """
    cnt = Counter(lab for _, lab in samples)
    # sqrt로 완만하게 — 4장짜리 클래스가 과대표집되면 그것만 외운다
    w = {c: 1.0 / math.sqrt(n) for c, n in cnt.items()}
    weights = [w[lab] for _, lab in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, classes: list[str]):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x).float()
        preds.append(out.argmax(1).cpu())
        labels.append(y)
    preds = torch.cat(preds)
    labels = torch.cat(labels)

    acc = (preds == labels).float().mean().item()

    per_class = {}
    for i, name in enumerate(classes):
        m = labels == i
        if m.sum():
            per_class[name] = (preds[m] == i).float().mean().item()

    # 이진 환산 — "정상이 아닌 것으로 예측했나". 지금 필터와 같은 문제로 바꿔
    # 재는 것이라 이진 모델과 직접 비교할 수 있다.
    normal_idx = {i for i, c in enumerate(classes) if c in NORMAL_CLASSES}
    is_def_true = torch.tensor([int(l.item()) not in normal_idx for l in labels])
    is_def_pred = torch.tensor([int(p.item()) not in normal_idx for p in preds])
    tp = (is_def_pred & is_def_true).sum().item()
    fp = (is_def_pred & ~is_def_true).sum().item()
    fn = (~is_def_pred & is_def_true).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    return {"acc": acc, "per_class": per_class,
            "bin_precision": prec, "bin_recall": rec,
            "bin_f1": 2 * prec * rec / max(prec + rec, 1e-9)}


@torch.no_grad()
def recall_at_fpr(model, loader, device, classes: list[str], fpr: float = 0.05):
    """정상 오탐률을 고정한 지점의 결함 재현율 — **필터로서의 성능**.

    위 `evaluate`의 bin_recall은 argmax 기준이라 임계값을 못 옮긴다. 필터는
    "정상을 5%만 버리는 대신 결함을 얼마나 건지나"로 평가해야 이진 모델과
    같은 잣대에 오른다(`score_testset_bycode.py`가 쓰는 기준과 같다).
    """
    model.eval()
    normal_idx = [i for i, c in enumerate(classes) if c in NORMAL_CLASSES]
    scores, is_def = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.softmax(model(x).float(), 1)
        scores.append((1.0 - p[:, normal_idx].sum(1)).cpu())   # 정상이 아닐 확률
        is_def.append(torch.tensor([int(v.item()) not in set(normal_idx) for v in y]))
    s = torch.cat(scores)
    d = torch.cat(is_def)
    neg = sorted(s[~d].tolist(), reverse=True)
    if not neg or not d.any():
        return 0.0
    thr = neg[max(0, int(len(neg) * fpr) - 1)]
    return (s[d] >= thr).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/multiclass")
    ap.add_argument("--arch", default="effb0", choices=["effb0", "convnext_t"])
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--gray", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
    print(f"device={device} arch={args.arch} · {len(classes)}클래스")
    print(f"  {', '.join(classes)}\n")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    gray = [transforms.Grayscale(num_output_channels=3)] if args.gray else []
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        *gray,
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.img), transforms.CenterCrop(args.img),
        *gray, transforms.ToTensor(), norm,
    ])

    train_ds = FolderDataset(root, "train", train_tf, classes)
    val_ds = FolderDataset(root, "val", val_tf, classes)
    print(f"train {len(train_ds):,} / val {len(val_ds):,}")

    train_ld = DataLoader(train_ds, batch_size=args.batch,
                          sampler=make_sampler(train_ds.samples, len(classes)),
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = {"acc": 0.0, "bin_f1": 0.0, "rec@fpr": 0.0}
    start_epoch = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best.update(ck.get("best", {}))
        print(f"이어서 학습: epoch {start_epoch}부터")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_ld):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
            if (i + 1) % 100 == 0:
                print(f"  e{epoch} {i+1}/{len(train_ld)} loss {run_loss/seen:.4f}")
        sched.step()

        m = evaluate(model, val_ld, device, classes)
        m["rec@fpr"] = recall_at_fpr(model, val_ld, device, classes)
        print(f"[epoch {epoch}] loss {run_loss/max(seen,1):.4f} | "
              f"acc {m['acc']:.4f} | 이진환산 P {m['bin_precision']:.4f} "
              f"R {m['bin_recall']:.4f} F1 {m['bin_f1']:.4f} | "
              f"오탐5% 재현율 {m['rec@fpr']:.4f} | {time.time()-t0:.0f}s")
        worst = sorted(m["per_class"].items(), key=lambda kv: kv[1])[:6]
        print("    재현율 낮은 클래스:", ", ".join(f"{c} {r:.2f}" for c, r in worst))

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch,
              "best": {k: max(v, m[k]) for k, v in best.items()},
              "arch": args.arch, "img": args.img, "gray": bool(args.gray),
              "classes": classes}
        torch.save(ck, ckpt_path)
        for key, fname in (("acc", "best_acc.pt"), ("bin_f1", "best_binF1.pt"),
                           ("rec@fpr", "best_recall.pt")):
            if m[key] > best[key]:
                best[key] = m[key]
                torch.save(ck, out / fname)
                (out / f"{fname[:-3]}_metrics.json").write_text(
                    json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"    {key} 갱신 {m[key]:.4f} → {fname}")

    print(f"\n완료. best_acc {best['acc']:.4f} · best_binF1 {best['bin_f1']:.4f}")


if __name__ == "__main__":
    main()

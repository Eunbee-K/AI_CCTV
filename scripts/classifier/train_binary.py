"""Stage-1 이진 분류기(정상/결함) 학습 — EfficientNet-B0.

build_binary_dataset.py가 만든 clsdata_v1을 학습한다.

Colab 사용법:
    !pip -q install torch torchvision   # 대개 이미 있음
    from google.colab import drive; drive.mount('/content/drive')
    !unzip -q /content/drive/MyDrive/clsdata_v1.zip -d /content/
    !python train_binary.py --data /content/clsdata_v1 \
        --out /content/drive/MyDrive/cls_runs/effb0 --epochs 15

세션이 끊기면 같은 명령을 다시 돌리면 된다 (--out의 last.pt에서 이어서 학습).

설계 메모
  - 정상:결함은 데이터 단계에서 1:1로 맞췄지만, 결함 안에서는 코드별 편차가 크다
    (BK 3,000 vs NS 4). 가중 샘플링으로 희소 결함이 손실에 묻히지 않게 한다.
  - 수직 뒤집기는 쓰지 않는다. 토사퇴적(DS)처럼 중력 방향에 의미가 있는 결함이 있다.
  - Stage-1은 필터다. 정확도보다 "결함을 놓치지 않는 것"(recall)이 중요하므로
    임계값별 recall / 프레임 감축률을 같이 보고한다.
"""

from __future__ import annotations

import argparse
import csv
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
from torchvision.models import (
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
    ConvNeXt_Tiny_Weights,
)

CLASSES = ["normal", "defect"]  # 인덱스 1 = 결함(양성)


class FrameDataset(Dataset):
    def __init__(self, root: Path, split: str, tf, manifest: dict[str, str]):
        self.tf = tf
        self.samples: list[tuple[Path, int, str]] = []
        for label, name in enumerate(CLASSES):
            d = root / split / name
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() == ".jpg":
                    rel = f"{split}\\{name}\\{f.name}"
                    code = manifest.get(rel) or manifest.get(rel.replace("\\", "/"), "?")
                    self.samples.append((f, label, code))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        # 외장드라이브는 동시 접근에서 간헐적으로 OSError(22)를 낸다. 파일은 멀쩡하므로
        # 잠깐 쉬었다 다시 연다. Colab 로컬 디스크에서는 걸리지 않는 경로다.
        for attempt in range(5):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                return self.tf(Image.open(io.BytesIO(data)).convert("RGB")), label
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))


def load_manifest(root: Path) -> dict[str, str]:
    """저장 경로 -> 묶음 이름. 묶음별 성능을 따로 보기 위해 필요하다.

    clsdata_v1은 결함코드(`code`)로, 현장 데이터셋(clsdata_v2)은 관로(`pipe`)로
    묶는다. 관로별로 보면 어느 현장에서 실패하는지가 드러난다.
    없으면 빈 dict — 묶음별 보고만 빠지고 학습은 그대로 돈다.
    """
    path = root / "manifest.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    key = "code" if "code" in rows[0] else ("pipe" if "pipe" in rows[0] else None)
    if key is None:
        return {}
    return {r["dst"]: r[key] for r in rows}


def build_model(arch: str):
    if arch == "effb0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif arch == "convnext_t":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    else:
        raise ValueError(arch)
    return m


def make_sampler(samples):
    """정상:결함 1:1을 유지하면서, 결함 안에서는 코드별로 고르게 뽑는다."""
    code_count = Counter(code for _, label, code in samples if label == 1)
    n_defect = sum(code_count.values())
    n_normal = len(samples) - n_defect
    weights = []
    for _, label, code in samples:
        if label == 0:
            weights.append(1.0 / max(n_normal, 1))
        else:
            # 코드별 가중치를 sqrt로 완만하게 — 4장짜리 NS가 과대표집되지 않도록
            w = 1.0 / math.sqrt(code_count[code])
            weights.append(w)
    total_defect_w = sum(w for w, (_, l, _) in zip(weights, samples) if l == 1)
    if total_defect_w > 0:
        scale = 1.0 / total_defect_w
        weights = [
            w * scale if l == 1 else w for w, (_, l, _) in zip(weights, samples)
        ]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def _roc_auc(pos: list[float], neg: list[float]) -> float:
    """양성 점수가 음성보다 높을 확률(Mann-Whitney U). 임계값과 무관하다.

    eval_binary.py에도 같은 함수가 있지만 그쪽이 이 파일을 import 하므로,
    순환 import를 피하려고 여기 따로 둔다.
    """
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    ranks, i = {}, 0
    vals = [s for s, _ in scored]
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2 + 1          # 동점은 평균 순위
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(scored) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    if not n_pos or not n_neg:
        return 0.0
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def evaluate(model, loader, device, samples):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.softmax(model(x).float(), dim=1)[:, 1]
        probs.append(p.cpu())
        labels.append(y)
    probs = torch.cat(probs)
    labels = torch.cat(labels)

    pred = (probs >= 0.5).long()
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()
    acc = (tp + tn) / max(len(labels), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    # 임계값별: 결함 recall을 목표치로 맞췄을 때 몇 %의 프레임을 걸러낼 수 있는가
    thresholds = {}
    defect_probs = probs[labels == 1].sort().values
    for target in (0.95, 0.98, 0.99):
        idx = int((1 - target) * len(defect_probs))
        thr = defect_probs[min(idx, len(defect_probs) - 1)].item() if len(defect_probs) else 0.5
        kept = (probs >= thr).float().mean().item()
        thresholds[target] = (thr, 1 - kept)

    # 임계값과 무관한 판별력. best 체크포인트는 이걸로 고른다 — F1은 임계값 0.5
    # 한 점에서만 잰 값이라, F1이 높아도 순위를 잘 못 매기는 경우가 있다.
    pos_l = probs[labels == 1].tolist()
    neg_l = probs[labels == 0].tolist()
    auc = _roc_auc(pos_l, neg_l) if pos_l and neg_l else 0.0

    # 결함 코드별 recall — 어떤 결함 유형을 놓치는지가 Stage-1의 핵심 지표
    by_code = defaultdict(lambda: [0, 0])
    for (_, label, code), p in zip(samples, probs.tolist()):
        if label != 1:
            continue
        by_code[code][1] += 1
        by_code[code][0] += p >= 0.5
    code_recall = {c: hit / tot for c, (hit, tot) in sorted(by_code.items())}

    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "thresholds": thresholds, "code_recall": code_recall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/effb0")
    ap.add_argument("--arch", default="effb0", choices=["effb0", "convnext_t"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--init", default="", help="이 체크포인트의 가중치에서 시작(백본 재사용)")
    ap.add_argument("--reset-head", action="store_true",
                    help="--init과 함께. 분류층만 초기화한다")
    args = ap.parse_args()

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} arch={args.arch}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.img), transforms.CenterCrop(args.img),
        transforms.ToTensor(), norm,
    ])

    manifest = load_manifest(root)
    train_ds = FrameDataset(root, "train", train_tf, manifest)
    val_ds = FrameDataset(root, "val", val_tf, manifest)
    print(f"train {len(train_ds):,} / val {len(val_ds):,}")

    train_ld = DataLoader(
        train_ds, batch_size=args.batch, sampler=make_sampler(train_ds.samples),
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = build_model(args.arch)
    if args.init:
        # 큰 데이터로 배운 백본을 재사용하고 현장 데이터로 파인튜닝한다.
        # v1 모델은 백본이 아니라 마지막 분류층이 "관 밖=정상, 관 안=결함"이라는
        # 잘못된 매핑을 외운 상태라, --reset-head로 그 층만 다시 시작할 수 있다.
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        if args.reset_head:
            fresh = build_model(args.arch).state_dict()
            for k in list(sd):
                if k.startswith("classifier."):
                    sd[k] = fresh[k]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"가중치 이어받음: {args.init} (epoch {ck.get('epoch', '?')})"
              f"{' · 분류층 초기화' if args.reset_head else ''}")
        if missing or unexpected:
            print(f"  맞지 않는 키 — 없음 {len(missing)}, 남음 {len(unexpected)}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    # best 체크포인트는 AUC로 고른다. F1은 임계값 0.5 한 점에서만 잰 값이라,
    # 필터로 쓸 때 중요한 "순위를 얼마나 잘 매기는가"를 반영하지 못한다.
    start_epoch, best_auc = 0, 0.0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():  # Colab 세션이 끊겼을 때 이어받기
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_auc = ck.get("best_auc", ck.get("best_f1", 0.0))
        print(f"이어서 학습: epoch {start_epoch}부터 (best_auc={best_auc:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_ld, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
            if i % 100 == 0:
                print(f"  e{epoch} {i}/{len(train_ld)} loss {run_loss / seen:.4f}")
        sched.step()

        m = evaluate(model, val_ld, device, val_ds.samples)
        print(
            f"[epoch {epoch}] loss {run_loss / max(seen,1):.4f} | "
            f"AUC {m['auc']:.4f} acc {m['acc']:.4f} P {m['precision']:.4f} "
            f"R {m['recall']:.4f} F1 {m['f1']:.4f} "
            f"| {time.time() - t0:.0f}s"
        )
        for target, (thr, cut) in m["thresholds"].items():
            print(f"    결함 recall {target:.0%} 지점: 임계값 {thr:.3f} → 프레임 {cut:.1%} 제거")
        worst = sorted(m["code_recall"].items(), key=lambda kv: kv[1])[:6]
        print("    recall 낮은 결함코드:", ", ".join(f"{c} {r:.2f}" for c, r in worst))

        ck = {
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "epoch": epoch,
            "best_auc": max(best_auc, m["auc"]), "f1": m["f1"],
            "arch": args.arch, "img": args.img,
        }
        torch.save(ck, ckpt_path)
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(ck, out / "best.pt")
            (out / "best_metrics.json").write_text(
                json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"    best 갱신 → {out / 'best.pt'}")

    print(f"\n완료. best AUC {best_auc:.4f} → {out / 'best.pt'}")


if __name__ == "__main__":
    main()

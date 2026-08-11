"""Stage-1 이진 분류기 성능 평가 — 부풀려진 숫자와 진짜 숫자를 분리해서 본다.

학습 로그의 전체 정확도는 두 가지 이유로 후하게 나온다.

1. 출처 지름길
   정상은 전부 AIHub인데 결함 25종 중 16종은 S20/S22에만 있다. 화질·압축 특성만으로
   갈라낼 수 있어 "결함을 배운 것"이 아닐 수 있다. AIHub 결함 vs AIHub 정상만 따로
   보면 이 지름길이 원천 차단된 값이 나온다 — 이게 실전에 가까운 숫자다.

2. 쉬운 정상
   OUT_MH(맨홀)·OUT_INVERT(인버트)·OUT_CAR(자동차)는 관 밖 장면이라 관 내부 결함과
   헷갈릴 여지가 거의 없다. 관 안 정상(IN·PJ)만 남기면 진짜 난이도가 드러난다.

또한 val은 정상:결함을 1:1로 맞춰놨으므로 "프레임 X% 제거"는 실제 영상과 무관하다.
대신 목표 recall에서의 정상 통과율(오탐률)을 보고, 실제 영상의 정상 비율을 가정해
기대 감축률을 따로 계산한다.

사용:
    python eval_binary.py --ckpt E:/AI_CCTV_RESULTS/filter/best.pt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_binary import build_model


class EvalSet(Dataset):
    def __init__(self, root: Path, rows, tf):
        self.root, self.rows, self.tf = root, rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # 외장드라이브는 동시 접근에서 간헐적으로 OSError(22)를 낸다. 파일은 멀쩡하므로 재시도한다.
        path = self.root / self.rows[i]["dst"]
        for attempt in range(5):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                return self.tf(Image.open(io.BytesIO(data)).convert("RGB")), i
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))


def roc_auc(pos, neg):
    """Mann-Whitney U 기반. 양성 점수가 음성보다 높을 확률."""
    scores = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    ranks, i = {}, 0
    ordered = [s for s, _ in scores]
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        avg = (i + j) / 2 + 1  # 동점은 평균 순위
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(scores) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    if not n_pos or not n_neg:
        return float("nan")
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarize(name, pos, neg, thr=0.5):
    """pos=결함 확률들, neg=정상 확률들."""
    tp = sum(p >= thr for p in pos)
    fn = len(pos) - tp
    fp = sum(p >= thr for p in neg)
    tn = len(neg) - fp
    rec = tp / max(len(pos), 1)
    spec = tn / max(len(neg), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(len(pos) + len(neg), 1)
    print(f"\n  [{name}]  결함 {len(pos):,} / 정상 {len(neg):,}")
    print(f"    정확도 {acc:.4f}  정밀도 {prec:.4f}  재현율 {rec:.4f}  F1 {f1:.4f}")
    print(f"    결함 놓침 {fn:,}장 ({1-rec:.2%})   정상 오탐 {fp:,}장 ({1-spec:.2%})")
    print(f"    ROC-AUC {roc_auc(pos, neg):.5f}")
    return {"recall": rec, "specificity": spec, "f1": f1, "auc": roc_auc(pos, neg)}


def threshold_table(pos, neg, targets=(0.95, 0.98, 0.99, 0.995)):
    print("\n=== 목표 결함 재현율별 운용점 ===")
    print("  목표recall  임계값   실제recall  정상통과율  영상정상90%시 감축  95%시 감축")
    ordered = sorted(pos)
    for t in targets:
        idx = max(0, min(len(ordered) - 1, int((1 - t) * len(ordered))))
        thr = ordered[idx]
        rec = sum(p >= thr for p in pos) / len(pos)
        fpr = sum(p >= thr for p in neg) / len(neg)  # 정상인데 결함으로 통과
        cut90 = 0.90 * (1 - fpr)
        cut95 = 0.95 * (1 - fpr)
        print(
            f"  {t:>8.1%}  {thr:>8.4f}  {rec:>9.4f}  {fpr:>9.2%}  "
            f"{cut90:>16.1%}  {cut95:>10.1%}"
        )
    print("  * 정상통과율 = 정상 프레임이 결함으로 잘못 넘어가는 비율(낮을수록 좋음)")
    print("  * 감축 = 실제 영상의 정상 비율을 90%/95%로 가정했을 때 Stage 2가 안 봐도 되는 프레임 비율")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"E:/AI_CCTV_RESULTS/filter/best.pt")
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v1")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="", help="확률 캐시 csv (있으면 추론 생략)")
    args = ap.parse_args()

    root = Path(args.data)
    with open(root / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    rows = [r for r in rows if (root / r["dst"]).exists()]
    print(f"val {len(rows):,}장")

    cache = Path(args.cache) if args.cache else Path(args.ckpt).with_name("val_probs.csv")
    if cache.exists():
        print(f"캐시 사용: {cache}")
        with open(cache, encoding="utf-8") as f:
            probs = [float(r["prob"]) for r in csv.DictReader(f)]
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(ck.get("arch", "effb0"))
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        img = ck.get("img", 224)
        print(f"체크포인트 epoch {ck.get('epoch')} | arch {ck.get('arch')} | device {device}")

        tf = transforms.Compose([
            transforms.Resize(img), transforms.CenterCrop(img), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ld = DataLoader(
            EvalSet(root, rows, tf), batch_size=args.batch, shuffle=False,
            num_workers=args.workers,
        )
        probs = [0.0] * len(rows)
        with torch.no_grad():
            for n, (x, idx) in enumerate(ld, 1):
                p = torch.softmax(model(x.to(device)).float(), 1)[:, 1].cpu()
                for j, i in zip(p.tolist(), idx.tolist()):
                    probs[i] = j
                if n % 20 == 0:
                    print(f"  {n * args.batch:,}/{len(rows):,}")
        with open(cache, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dst", "binary", "code", "source", "prob"])
            for r, p in zip(rows, probs):
                w.writerow([r["dst"], r["binary"], r["code"], r["source"], p])
        print(f"확률 저장: {cache}")

    pos = [p for r, p in zip(rows, probs) if r["binary"] == "defect"]
    neg = [p for r, p in zip(rows, probs) if r["binary"] == "normal"]

    print("\n" + "=" * 68)
    print("1. 전체 (학습 로그와 같은 값 — 부풀려진 숫자)")
    print("=" * 68)
    overall = summarize("전체 val", pos, neg)

    print("\n" + "=" * 68)
    print("2. 출처 지름길 분리")
    print("=" * 68)
    aihub_pos = [p for r, p in zip(rows, probs) if r["binary"] == "defect" and r["source"] == "aihub"]
    s20_pos = [p for r, p in zip(rows, probs) if r["binary"] == "defect" and r["source"] != "aihub"]
    matched = summarize("AIHub 결함 vs AIHub 정상 (지름길 차단 — 진짜 실력)", aihub_pos, neg)
    summarize("S20/S22 결함 vs AIHub 정상 (지름길 있음 — 부풀려짐)", s20_pos, neg)

    print("\n" + "=" * 68)
    print("3. 쉬운 정상(관 밖 장면) 제외")
    print("=" * 68)
    inside = [p for r, p in zip(rows, probs) if r["binary"] == "normal" and r["code"] in ("IN", "PJ")]
    outside = [p for r, p in zip(rows, probs) if r["binary"] == "normal" and r["code"].startswith("OUT")]
    hard = summarize("전체 결함 vs 관 안 정상(IN·PJ)", pos, inside)
    summarize("AIHub 결함 vs 관 안 정상 (가장 엄격)", aihub_pos, inside)
    summarize("전체 결함 vs 관 밖 정상(OUT_*)", pos, outside)

    print("\n" + "=" * 68)
    print("4. 결함 코드별 놓침")
    print("=" * 68)
    by_code = defaultdict(list)
    src_of = {}
    for r, p in zip(rows, probs):
        if r["binary"] == "defect":
            by_code[r["code"]].append(p)
            src_of[r["code"]] = r["source"]
    print("  코드   출처    장수   재현율   놓침")
    for code, ps in sorted(by_code.items(), key=lambda kv: sum(p >= 0.5 for p in kv[1]) / len(kv[1])):
        rec = sum(p >= 0.5 for p in ps) / len(ps)
        miss = len(ps) - sum(p >= 0.5 for p in ps)
        print(f"  {code:<6} {src_of[code]:<6} {len(ps):>5}   {rec:>6.3f}   {miss:>4}장")

    print("\n" + "=" * 68)
    print("5. 정상 코드별 오탐")
    print("=" * 68)
    by_norm = defaultdict(list)
    for r, p in zip(rows, probs):
        if r["binary"] == "normal":
            by_norm[r["code"]].append(p)
    print("  코드         장수   오탐률   오탐")
    for code, ps in sorted(by_norm.items(), key=lambda kv: -sum(p >= 0.5 for p in kv[1]) / len(kv[1])):
        fpr = sum(p >= 0.5 for p in ps) / len(ps)
        print(f"  {code:<11} {len(ps):>5}   {fpr:>6.2%}   {sum(p >= 0.5 for p in ps):>4}장")

    print("\n" + "=" * 68)
    print("6. 운용 임계값 (지름길 차단 조건: AIHub 결함 vs 관 안 정상)")
    print("=" * 68)
    threshold_table(aihub_pos, inside)

    out = Path(args.ckpt).with_name("eval_report.json")
    out.write_text(
        json.dumps(
            {"overall": overall, "source_matched": matched, "inside_only": hard},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n요약 저장: {out}")


if __name__ == "__main__":
    main()

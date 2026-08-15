"""신설관로 필터 v2 — 정상에서 관 밖(맨홀 진입) 프레임을 제거한다.

**왜 빼야 하나**

영상 도입부 몇 초는 로봇이 관에 들어가기 전이라 맨홀·도로·하늘이 찍힌다. 이 프레임이
정상에 들어가면 모델이 "관 밖 = 정상"만 배우고 관 내부 판별은 못 배운다. 첫 필터
(AIHub 78,388장, 현장 AUC 0.516)가 정확히 그렇게 망가졌고, 그때 정상으로 나온 구간이
영상 첫 0~10초뿐이었다.

AIHub 쪽은 build_new_filter.py에서 OUT_*을 아예 안 넣었지만, **현장 프레임의 도입부**가
그대로 남아 있었다.

**어떻게 찾나** — 그 첫 필터(v1)를 쓴다. 현장 영상에서는 무작위지만 "관 밖"만은
확실히 알아본다(관 밖이면 확률이 0.5 아래로 뚝 떨어진다). 망가진 모델의 그 한 가지
능력만 빌린다.

사용:
    python build_new_v2.py --dry-run      # 몇 장이 지워지는지 먼저 확인
    python build_new_v2.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from paths import DATASET, RESULTS  # noqa: E402

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_binary import build_model

DS = DATASET
V1_CKPT = RESULTS / "filter/best.pt"
OUTSIDE_THRESHOLD = 0.5    # v1이 이 아래를 주면 관 밖으로 본다


class Simple(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), i


@torch.no_grad()
def outside_probs(files, workers: int):
    ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    m = build_model(ck.get("arch", "effb0"))
    m.load_state_dict(ck["model"])
    m.eval()
    img = ck.get("img", 224)
    tf = transforms.Compose([
        transforms.Resize(img), transforms.CenterCrop(img), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = [1.0] * len(files)
    for x, idx in DataLoader(Simple(files, tf), batch_size=64, num_workers=workers):
        p = torch.softmax(m(x).float(), 1)[:, 1]
        for j, i in zip(p.tolist(), idx.tolist()):
            out[i] = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DS / "clsdata_new_v1"))
    ap.add_argument("--out", default=str(DS / "clsdata_new_v2"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    if not V1_CKPT.exists():
        print(f"!! v1 모델이 없습니다: {V1_CKPT}")
        return

    # **현장 프레임만** 검사한다. AIHub IN은 관 내부 정상이고, v1이 유일하게 제대로
    # 배운 것이 바로 그 IN이라 확률을 낮게 준다. 그걸 관 밖으로 오해하면 정상 데이터의
    # 핵심을 통째로 지우게 된다(첫 시도에서 2,099장이 걸렸다).
    # 관 밖 프레임은 영상 도입부에만 있으므로 현장 프레임에서만 나온다.
    targets = []
    for split in ("train", "val"):
        targets += [f for f in sorted((base / split / "normal").glob("*.jpg"))
                    if not f.name.startswith("aihub_")]
    print(f"현장 정상 프레임 {len(targets):,}장 검사 중 (AIHub는 대상 아님)...")

    probs = outside_probs(targets, args.workers)
    drop = [f for f, p in zip(targets, probs) if p < OUTSIDE_THRESHOLD]

    by = Counter()
    for f in drop:
        split = f.parent.parent.name
        src = "aihub" if f.name.startswith("aihub_") else (
            "fieldset(오수)" if f.name.startswith("fs__") else "상면(우수)")
        by[(split, src)] += 1
    print(f"\n관 밖으로 판정된 정상 프레임 {len(drop):,}장")
    for k, v in sorted(by.items()):
        print(f"  [{k[0]:<5}] {k[1]:<15} {v:>4}장")

    if args.dry_run:
        print("\n(dry-run — 파일 손대지 않음)")
        print("예시:", [f.name[:52] for f in drop[:5]])
        return

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)
    dropped = {f.name for f in drop}
    n = 0
    for split in ("train", "val"):
        for f in list((out / split / "normal").glob("*.jpg")):
            if f.name in dropped:
                f.unlink()
                n += 1

    # manifest도 남은 파일 기준으로 다시 쓴다
    with open(out / "manifest.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (out / r["dst"]).exists()]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{n:,}장 삭제 · manifest {len(rows):,}행")
    print("=== 최종 ===")
    for split in ("train", "val"):
        a = len(list((out / split / "normal").glob("*.jpg")))
        b = len(list((out / split / "defect").glob("*.jpg")))
        print(f"  {split:<6} 정상 {a:>5}  결함 {b:>5}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

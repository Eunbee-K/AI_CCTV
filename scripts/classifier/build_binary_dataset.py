"""Stage-1 이진 분류기(정상/결함) 학습 데이터셋 구성.

두 출처를 합쳐 정상/결함 1:1 데이터셋을 만들고, 256x256 JPEG으로 줄여
Colab 업로드가 가능한 크기(~1.6GB)로 떨어뜨린다.

출처
  A. AI_CCTV_DATASET/original/aihub_data_bbox/image/<CODE>/
     - 결함 9종(BK CC CL DS ETC JD JF LP SD), 정상 5종(IN PJ OUT_MH OUT_INVERT OUT_CAR)
  B. AI_CCTV_DATASET/original/rename_data_s20_s22_bbox/<폴더>/images/
     - 결함 24종. A에 없는 16종(BC CM CX DE DF DG HL IF JS LD LS NS PO RT SG TO)을 채운다.

분할
  S20 파일명에는 소스 영상이 들어있다(`JF_S20_01-00510.mp4_000213.734.jpg`).
  같은 영상 프레임이 train/val에 섞이면 검증 점수가 부풀려지므로 영상 단위로 나눈다.
  영상 정보가 없는 출처(AIHub, S22)는 파일 단위 무작위 분할.

출력
  <OUT>/train/{normal,defect}/*.jpg
  <OUT>/val/{normal,defect}/*.jpg
  <OUT>/manifest.csv     원본 경로·결함코드·출처·분할 기록

현장 영상 프레임(fieldset_v1_label)은 여기 넣지 않는다. 별도 모델의 파인튜닝용.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from paths import DATASET  # noqa: E402

from PIL import Image

DATASET_ROOT = DATASET
AIHUB = DATASET_ROOT / "original/aihub_data_bbox/image"
S20S22 = DATASET_ROOT / "original/rename_data_s20_s22_bbox"

# AIHub 폴더명은 그 자체가 코드다.
AIHUB_NORMAL = ["IN", "PJ", "OUT_MH", "OUT_INVERT", "OUT_CAR"]
AIHUB_DEFECT = ["BK", "CC", "CL", "DS", "ETC", "JD", "JF", "LP", "SD"]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_RE = re.compile(r"_(.+?\.mp4)_")


def defect_code_of(folder_name: str) -> str:
    """S20/S22 폴더명에서 결함 코드를 뽑는다. `BC_S20_좌굴` / `S22_BK_05파손(BK)`."""
    parts = folder_name.split("_")
    return parts[1] if folder_name.startswith("S22") else parts[0]


def group_key(path: Path) -> str:
    """분할 단위. 소스 영상을 알 수 있으면 영상, 아니면 파일 자체."""
    m = VIDEO_RE.search(path.name)
    return m.group(1) if m else f"__file__:{path}"


def collect_sources() -> list[tuple[Path, str, str, str]]:
    """(경로, 이진라벨, 결함코드, 출처) 목록."""
    items: list[tuple[Path, str, str, str]] = []

    for code in AIHUB_NORMAL + AIHUB_DEFECT:
        d = AIHUB / code
        if not d.is_dir():
            print(f"  ! 없음: {d}", file=sys.stderr)
            continue
        binary = "normal" if code in AIHUB_NORMAL else "defect"
        for f in d.iterdir():
            if f.suffix.lower() in IMG_EXT:
                items.append((f, binary, code, "aihub"))

    for d in sorted(S20S22.iterdir()) if S20S22.is_dir() else []:
        img_dir = d / "images"
        if not img_dir.is_dir():
            continue
        code = defect_code_of(d.name)
        src = "s22" if d.name.startswith("S22") else "s20"
        for f in img_dir.iterdir():
            if f.suffix.lower() in IMG_EXT:
                items.append((f, "defect", code, src))

    return items


def sample_by_class(items, cap, rng):
    """(이진라벨, 코드)별로 cap장까지, 영상 단위를 깨지 않고 뽑는다."""
    by_class = defaultdict(list)
    for it in items:
        by_class[(it[1], it[2])].append(it)

    picked = []
    for key, group in sorted(by_class.items()):
        if len(group) <= cap:
            picked.extend(group)
            continue
        by_group = defaultdict(list)
        for it in group:
            by_group[group_key(it[0])].append(it)
        keys = list(by_group)
        rng.shuffle(keys)
        taken = []
        for k in keys:
            if len(taken) >= cap:
                break
            taken.extend(by_group[k][: cap - len(taken)])
        picked.extend(taken)
    return picked


def plan_splits(records, val_ratio, rng):
    """영상 -> train/val 배정. 영상 하나는 반드시 한쪽에만 들어간다.

    영상 하나가 여러 결함 코드에 걸치기 때문(1,569개 중 685개가 2코드 이상),
    코드별로 따로 나누면 같은 관로가 train과 val 양쪽에 들어가 검증이 오염된다.
    희소 코드부터 val 몫을 채워, 코드 수가 적은 결함도 검증에 남게 한다.

    records: (group, cls) 리스트. cls는 (binary, code).
    반환: {group: "train"|"val"}
    """
    by_group = defaultdict(list)
    for group, cls in records:
        by_group[group].append(cls)
    cls_total = Counter(cls for _, cls in records)
    cls_groups = defaultdict(set)
    for group, classes in by_group.items():
        for cls in classes:
            cls_groups[cls].add(group)

    assigned: dict[str, str] = {}
    val_count: Counter = Counter()

    for cls in sorted(cls_total, key=lambda c: cls_total[c]):
        target = round(cls_total[cls] * val_ratio)
        if target < 1 or len(cls_groups[cls]) < 2:
            continue
        # set 순회 순서는 실행마다 달라진다(문자열 해시 랜덤화). 정렬 후 섞어야 재현된다.
        cand = sorted(g for g in cls_groups[cls] if g not in assigned)
        rng.shuffle(cand)
        for g in cand:
            if val_count[cls] >= target:
                break
            # 이 영상의 모든 코드를 함께 val로 보내게 되므로 같이 집계한다
            assigned[g] = "val"
            for c in by_group[g]:
                val_count[c] += 1

    for g in by_group:
        assigned.setdefault(g, "train")
    return assigned


def resolve_groups(records):
    """소스 영상이 1개뿐인 코드는 영상 단위로 못 나눈다. 그 코드만 파일 단위로 되돌린다.

    records: (group, cls, uid) 리스트. 반환: 같은 순서의 최종 group 리스트.
    """
    cls_groups = defaultdict(set)
    for group, cls, _ in records:
        cls_groups[cls].add(group)
    degenerate = {cls for cls, gs in cls_groups.items() if len(gs) < 2}
    for cls in sorted(degenerate):
        print(
            f"  ! {cls[1]}: 소스 영상 1개뿐 — 파일 단위로 분할(같은 관로가 train/val에 걸침)",
            file=sys.stderr,
        )
    return [uid if cls in degenerate else group for group, cls, uid in records]


def assign_splits(items, val_ratio, rng):
    raw = [(group_key(it[0]), (it[1], it[2]), f"__file__:{it[0]}") for it in items]
    groups = resolve_groups(raw)
    plan = plan_splits(list(zip(groups, (r[1] for r in raw))), val_ratio, rng)
    return [(*it, plan[g]) for it, g in zip(items, groups)]


def convert_one(job):
    src, dst, size = job
    try:
        with Image.open(src) as im:
            im.convert("RGB").resize((size, size), Image.BILINEAR).save(
                dst, "JPEG", quality=88, optimize=True
            )
        return True
    except Exception as exc:  # 손상 파일은 건너뛰고 계속
        print(f"  ! {src}: {exc}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATASET_ROOT / "clsdata_v1"))
    ap.add_argument("--cap", type=int, default=2000, help="결함 클래스당 최대 장수")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="집계만 하고 파일은 만들지 않음")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out)

    print("원본 스캔 중...")
    items = collect_sources()
    print(f"  전체 후보 {len(items):,}장")

    defects = sample_by_class([i for i in items if i[1] == "defect"], args.cap, rng)
    n_defect = len(defects)

    # 정상은 결함 총량에 1:1로 맞춰, 5개 클래스에서 보유량 비례로 뽑는다.
    normals_all = [i for i in items if i[1] == "normal"]
    pool = defaultdict(list)
    for it in normals_all:
        pool[it[2]].append(it)
    total_norm = sum(len(v) for v in pool.values())
    normals = []
    for code in AIHUB_NORMAL:
        group = pool.get(code, [])
        if not group:
            continue
        quota = min(len(group), round(n_defect * len(group) / total_norm))
        rng.shuffle(group)
        normals.extend(group[:quota])
    # 반올림 오차 보정
    if len(normals) < n_defect:
        chosen = {id(i) for i in normals}
        leftover = [i for i in normals_all if id(i) not in chosen]
        rng.shuffle(leftover)
        normals.extend(leftover[: n_defect - len(normals)])

    selected = assign_splits(defects + normals, args.val_ratio, rng)

    stat = Counter((s, b) for _, b, _, _, s in selected)
    per_code = defaultdict(Counter)
    for _, b, code, _, s in selected:
        per_code[(b, code)][s] += 1

    for label in ("defect", "normal"):
        print(f"\n=== {label} 코드별 (train/val) ===")
        for (b, code), c in sorted(per_code.items()):
            if b != label or c["train"] + c["val"] == 0:
                continue
            print(f"  {code:<11} train {c['train']:>6}  val {c['val']:>5}")
    print("\n=== 분할 합계 ===")
    for split in ("train", "val"):
        n = stat[(split, "normal")], stat[(split, "defect")]
        print(f"  {split:<10} normal {n[0]:>6}  defect {n[1]:>6}  합계 {sum(n):>6}")
    print(f"\n  총 {len(selected):,}장")

    if args.dry_run:
        print("\n(dry-run — 파일 생성 안 함)")
        return

    for split in ("train", "val"):
        for b in ("normal", "defect"):
            (out_root / split / b).mkdir(parents=True, exist_ok=True)

    jobs, rows, seen = [], [], Counter()
    for src, binary, code, source, split in selected:
        seen[code] += 1
        dst = out_root / split / binary / f"{source}_{code}_{seen[code]:06d}.jpg"
        jobs.append((src, dst, args.size))
        rows.append(
            {
                "split": split,
                "binary": binary,
                "code": code,
                "source": source,
                "video": group_key(src) if ".mp4" in src.name else "",
                "dst": str(dst.relative_to(out_root)),
                "src": str(src),
            }
        )

    print(f"\n{len(jobs):,}장 변환 중 ({args.size}px JPEG, worker {args.workers})...")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ok in ex.map(convert_one, jobs, chunksize=64):
            done += ok
            if done % 5000 == 0:
                print(f"  {done:,}/{len(jobs):,}")

    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    size_mb = sum(p.stat().st_size for p in out_root.rglob("*.jpg")) / 1e6
    print(f"\n완료: {done:,}장 / {size_mb:,.0f}MB -> {out_root}")


if __name__ == "__main__":
    main()

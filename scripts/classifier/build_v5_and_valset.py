"""OLD_v5 학습셋 + 검증셋 — **야장은 학습에 쓰지 않고 검증에만 쓴다.**

왜 이렇게 나누나
    야장(북면·상면·설악면 조사표 프레임)은 조사원이 현장에서 판정한 자료다.
    학습에 쓰면 "배운 걸 다시 맞히는" 셈이라 실력을 잴 수 없다. 그래서 학습은
    AIHub/S20/S22 공개 데이터로만 하고 야장은 검증에 쓴다.

검증셋 `valset_bycode` — 하나로 통합
    결함 종류당 VAL_PER_CODE장. **야장을 먼저 쓰고, 모자란 만큼만 원본에서 채운다.**
    야장이 실제 조사 상황에 가까우므로 우선한다. 야장에 없거나(DE·DG·LD·NS)
    10장이 안 되는 종류(BC 7 · CX 2 · HL 3)만 S20/S22·AIHub에서 보충한다.
    정상도 종류당 VAL_PER_CODE장(IN·PJ·OUT_MH·OUT_CAR·OUT_INVERT).

    ETC(기타)는 결함 종류가 아니므로 검증 대상에서 뺀다 — 학습에는 넣는다.

학습셋 `clsdata_old_v5`
    결함: 24종 × TRAIN_PER_CODE장(부족하면 있는 만큼) + ETC TRAIN_ETC장
    정상: 결함과 1:1, IN 30% · PJ 40% · OUT_MH 10% · OUT_CAR 10% · OUT_INVERT 10%
          **관 밖(OUT_*) 30%가 핵심이다.** 필터가 관 밖을 100% 결함으로 오판하는
          것이 확인됐는데, v1~v4 학습 데이터에 관 밖 정상이 한 장도 없었기 때문이다.

사용:
    python build_v5_and_valset.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import DATASET, FILTER_DATA  # noqa: E402

ORIG = DATASET / "original"
S20 = ORIG / "rename_data_s20_s22_bbox"
AIHUB = ORIG / "aihub_data_bbox" / "image"
V3 = FILTER_DATA / "clsdata_old_v3"                  # 야장 프레임이 복사돼 있는 곳

VAL_PER_CODE = 10
TRAIN_PER_CODE = 100
TRAIN_ETC = 500
CAP = 4000                                       # 폴더 하나에서 읽을 상한

NORMAL_MIX = {"IN": 0.30, "PJ": 0.40, "OUT_MH": 0.10, "OUT_CAR": 0.10, "OUT_INVERT": 0.10}

# 야장 캡션의 결함명 -> 코드. 야장은 한글 이름으로 적혀 있다.
YAJANG_NAME_TO_CODE = {
    "균열원주": "CC", "균열길이": "CL", "균열복합": "CM", "표면손상": "SD",
    "좌굴": "BC", "라이닝결함": "LD", "변형": "DF", "파손": "BK", "붕괴": "CX",
    "영구장애물": "PO", "천공": "HL", "연결관돌출": "LP", "연결관접합부": "LS",
    "이음부이탈": "JS", "이음부손상": "JF", "이음부단차": "JD", "역경사": "NS",
    "침하": "SG", "내피생성": "DE", "토사퇴적": "DS", "폐유부착": "DG",
    "임시장애물": "TO", "뿌리침입": "RT", "침입수": "IF",
}
# ETC(기타)는 결함 종류가 아니라 검증 대상에서 제외한다.
VAL_EXCLUDE = {"ETC"}


def code_of(folder: str):
    if folder in NORMAL_MIX:
        return folder
    m = re.match(r"^S22_([A-Z]{2})_", folder) or re.match(r"^([A-Z]{2,3})_S20_", folder)
    if m:
        return m.group(1)
    return folder if re.fullmatch(r"[A-Z]{2,3}", folder) else None


def take_from(d: Path, tag: str, cap: int = CAP):
    """폴더 앞에서부터 cap개만 읽는다.

    S20/S22는 `<코드폴더>/images/`, AIHub는 코드폴더 바로 아래에 파일이 있다.
    **전체를 훑지 않는다** — AIHub IN(9만 장)은 외장하드에서 나열만 5분이 넘는다.
    """
    out = []
    for sub in (d / "images", d):
        if not sub.is_dir():
            continue
        with os.scandir(sub) as it:
            for e in it:
                if len(out) >= cap:
                    return out
                if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    out.append((Path(e.path), tag))
        if out:
            break
    return out


def scan_all() -> dict:
    pool = defaultdict(list)
    for root, tag in ((S20, "s20"), (AIHUB, "aihub")):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            code = code_of(d.name)
            if code:
                pool[code] += take_from(d, tag)
    return pool


def interleave(items, seed: int):
    """출처(s20/aihub)가 골고루 섞이도록 번갈아 뽑는다."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    for v in by.values():
        rng.shuffle(v)
    out, idx, tags = [], defaultdict(int), list(by)
    while any(idx[t] < len(by[t]) for t in tags):
        for t in tags:
            if idx[t] < len(by[t]):
                out.append(by[t][idx[t]])
                idx[t] += 1
    return out


def rmtree_retry(d: Path):
    """외장하드에서 rmtree가 자주 실패한다 — 삭제 지연, 파일 잠금 둘 다.

    반쯤 지워진 채 죽으면 데이터셋이 망가지므로, 끝까지 못 지우면 **지우지 말고
    옆으로 치운다.** 이름만 바꾸면 잠긴 파일이 있어도 성공한다.
    """
    if not d.exists():
        return
    for attempt in range(3):
        try:
            shutil.rmtree(d)
            return
        except OSError:
            time.sleep(1.0 * (attempt + 1))
    trash = d.with_name(f"{d.name}__old_{int(time.time())}")
    d.rename(trash)
    print(f"  (지우지 못해 옆으로 치움: {trash.name} — 나중에 수동 삭제)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    train_out = FILTER_DATA / "clsdata_old_v5"
    val_out = FILTER_DATA / "valset_bycode"
    for d in (train_out, val_out):
        rmtree_retry(d)
    for lab in ("defect", "normal"):
        (val_out / lab).mkdir(parents=True)
        (train_out / "train" / lab).mkdir(parents=True)

    pool = scan_all()
    defect_codes = sorted(k for k in pool if k not in NORMAL_MIX)
    print(f"원본 결함 {len(defect_codes)}종\n")

    # 야장 프레임을 코드별로 모아둔다 — 검증의 1순위 재료
    yajang = defaultdict(list)
    with open(V3 / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["area"] not in ("북면", "상면", "설악면") or r["label"] != "defect":
                continue
            code = YAJANG_NAME_TO_CODE.get(r["name"])
            src = V3 / r["dst"].replace("\\", "/")
            if code and src.exists():
                yajang[code].append(src)
    for v in yajang.values():
        random.shuffle(v)

    # ── 1) 검증셋 — 야장 우선, 모자라면 원본에서 보충 ──
    used = defaultdict(set)
    vman = []
    val_codes = [c for c in sorted(set(defect_codes) | set(YAJANG_NAME_TO_CODE.values()))
                 if c not in VAL_EXCLUDE]
    print(f"{'코드':<6}{'야장':>6}{'보충':>6}{'합계':>6}")
    for code in val_codes:
        take_ya = yajang.get(code, [])[:VAL_PER_CODE]
        for src in take_ya:
            dst = val_out / "defect" / f"{code}__yajang__{src.name}"
            shutil.copy2(src, dst)
            vman.append({"label": "defect", "code": code, "from": "yajang",
                         "file": dst.name, "src": str(src)})
        need = VAL_PER_CODE - len(take_ya)
        extra = interleave(pool.get(code, []), args.seed)[:need] if need > 0 else []
        for f, tag in extra:
            dst = val_out / "defect" / f"{code}__{tag}__{f.name}"
            shutil.copy2(f, dst)
            vman.append({"label": "defect", "code": code, "from": tag,
                         "file": dst.name, "src": str(f)})
            used[code].add(f)
        print(f"{code:<6}{len(take_ya):>6}{len(extra):>6}{len(take_ya)+len(extra):>6}"
              + ("" if len(take_ya) + len(extra) == VAL_PER_CODE else "  <- 부족"))

    # 정상도 결함 총량만큼 — 학습과 같은 비율로.
    # **야장 정상을 먼저 쓰고 모자란 만큼만 AIHub로 채운다.** 결함이 야장 우선인데
    # 정상만 AIHub면 모델이 결함/정상이 아니라 출처를 보고 갈라도 점수가 나온다
    # (v1이 그렇게 망가졌다). 야장 정상은 전부 관 안쪽이고 89%가 '이음부(접합부)존재'라
    # AIHub의 PJ에 대응하므로 PJ부터 채운다. IN·OUT_*은 야장에 없어 AIHub만 쓴다.
    ya_normal = []
    with open(V3 / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["area"] in ("북면", "상면", "설악면") and r["label"] == "normal":
                src = V3 / r["dst"].replace("\\", "/")
                if src.exists():
                    ya_normal.append(src)
    random.shuffle(ya_normal)

    n_val_defect = sum(1 for r in vman if r["label"] == "defect")
    print(f"\n검증 정상 목표 {n_val_defect}장 (결함과 같은 수) — 야장 우선")
    print(f"  {'코드':<12}{'목표':>6}{'야장':>6}{'AIHub':>7}")
    ya_i = 0
    for code, ratio in NORMAL_MIX.items():
        want = round(n_val_defect * ratio)
        n_ya = 0
        if code == "PJ":                       # 야장 정상은 이음부 계열이다
            for src in ya_normal[ya_i:ya_i + want]:
                dst = val_out / "normal" / f"{code}__yajang__{src.name}"
                shutil.copy2(src, dst)
                vman.append({"label": "normal", "code": code, "from": "yajang",
                             "file": dst.name, "src": str(src)})
                n_ya += 1
            ya_i += n_ya
        need = want - n_ya
        take = interleave(pool.get(code, []), args.seed)[:need] if need > 0 else []
        for f, tag in take:
            dst = val_out / "normal" / f"{code}__{tag}__{f.name}"
            shutil.copy2(f, dst)
            vman.append({"label": "normal", "code": code, "from": tag,
                         "file": dst.name, "src": str(f)})
            used[code].add(f)
        print(f"  {code:<12}{want:>6}{n_ya:>6}{len(take):>7}")

    with open(val_out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["label", "code", "from", "file", "src"])
        w.writeheader()
        w.writerows(vman)
    print(f"\n검증셋 -> {val_out}  결함 {n_val_defect} · "
          f"정상 {sum(1 for r in vman if r['label'] == 'normal')}\n")

    # ── 3) 학습 결함 — AIHub/S20/S22만 ──
    tman, n_defect = [], 0
    print(f"{'코드':<6}{'목표':>6}{'뽑음':>6}")
    for code in defect_codes:
        target = TRAIN_ETC if code == "ETC" else TRAIN_PER_CODE
        remain = [x for x in interleave(pool[code], args.seed + 1) if x[0] not in used[code]]
        take = remain[:target]
        for f, tag in take:
            dst = train_out / "train" / "defect" / f"{tag}_{code}__{f.name}"
            shutil.copy2(f, dst)
            tman.append({"split": "train", "label": "defect", "area": tag, "code": code,
                         "dst": f"train\\defect\\{dst.name}"})
        n_defect += len(take)
        print(f"{code:<6}{target:>6}{len(take):>6}"
              + ("" if len(take) == target else "  <- 부족"))

    # ── 4) 학습 정상 — 결함과 1:1, 지정 비율 ──
    print(f"\n정상 목표 {n_defect:,}장 (결함과 1:1)")
    print(f"{'코드':<12}{'목표':>7}{'뽑음':>7}")
    for code, ratio in NORMAL_MIX.items():
        want = round(n_defect * ratio)
        remain = [x for x in interleave(pool.get(code, []), args.seed + 2)
                  if x[0] not in used[code]]
        take = remain[:want]
        for f, tag in take:
            dst = train_out / "train" / "normal" / f"{tag}_{code}__{f.name}"
            shutil.copy2(f, dst)
            tman.append({"split": "train", "label": "normal", "area": tag, "code": code,
                         "dst": f"train\\normal\\{dst.name}"})
        print(f"{code:<12}{want:>7}{len(take):>7}"
              + ("" if len(take) == want else "  <- 부족"))

    with open(train_out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["split", "label", "area", "code", "dst"])
        w.writeheader()
        w.writerows(tman)

    print(f"\n학습셋 -> {train_out}")
    print(f"  결함 {sum(1 for r in tman if r['label'] == 'defect'):,} · "
          f"정상 {sum(1 for r in tman if r['label'] == 'normal'):,}")
    print("  (야장은 학습에 쓰지 않는다 — 전부 검증)")


if __name__ == "__main__":
    main()

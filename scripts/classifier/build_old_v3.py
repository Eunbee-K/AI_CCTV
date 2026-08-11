"""노후관로 필터 v3 — 야장에 없던 결함 유형을 S20/S22로 채운다.

**왜 필요한가**

실제 노후관로 영상(JB1-2-0413)에서 확인된 결함 2건이 모두 **변형**이었는데,
v2 필터는 둘 다 놓쳤다(128프레임 중 58위·109위). 야장 학습 데이터에 변형이
**10건(결함 5,307장의 0.19%)** 뿐이라 사실상 배운 적이 없기 때문이다.

필터의 역할은 "결함을 찾는 것"이고 이름 붙이는 것은 YOLO의 몫이다. 그러려면
필터가 **24종 결함을 모두 결함으로 알아야** 한다. 지금은 16종이 사실상 비어 있다.

    야장 0건    내피생성 · 폐유부착 · 라이닝결함 · 역경사
    10건 미만   붕괴 2 · 천공 3 · 좌굴 7 · 변형 10 · 침하 10
    100건 미만  뿌리침입 19 · 임시장애물 39 · 균열복합 42 · 침입수 79 · 표면손상 96

**비율을 지킨다** — 결함이 정상보다 많아지면 모델이 결함 쪽으로 치우쳐 오탐이 는다.
각 유형을 TARGET장까지만 올리고, 전체 결함이 정상을 넘지 않는 선에서 멈춘다.

**S20에는 정상이 없다.** 결함만 들어오므로 "S20처럼 생기면 결함" 지름길이 생길 수
있다. 다만 v2에 AIHub 정상이 7,000장 들어가 정상 쪽 다양성이 확보돼 있고, val이
야장 전용이라 지름길로 점수를 부풀릴 수는 없다. 실영상 효과는 돌려봐야 안다.

사용:
    python build_old_v3.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

DS = Path(r"E:/AI_CCTV_DATASET")
S20 = DS / "original/rename_data_s20_s22_bbox"

# 코드 -> 야장에서 쓰는 항목명 (야장 보유량 집계에 쓴다)
KO = {
    "BC": "좌굴", "BK": "파손", "CC": "균열원주", "CL": "균열길이", "CM": "균열복합",
    "CX": "붕괴", "DE": "내피생성", "DF": "변형", "DG": "폐유부착", "DS": "토사퇴적",
    "HL": "천공", "IF": "침입수", "JD": "이음부단차", "JF": "이음부손상",
    "JS": "이음부이탈", "LD": "라이닝결함", "LP": "연결관돌출", "LS": "연결관접합부",
    "NS": "역경사", "PO": "영구장애물", "RT": "뿌리침입", "SD": "표면손상",
    "SG": "침하", "TO": "임시장애물",
}
TARGET = 400   # 각 결함 유형을 이 장수까지 끌어올린다


def s20_files() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for d in sorted(S20.iterdir()):
        if not d.is_dir() or not (d / "images").is_dir():
            continue
        code = d.name.split("_")[1] if d.name.startswith("S22") else d.name.split("_")[0]
        out.setdefault(code, []).extend(sorted((d / "images").iterdir()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DS / "clsdata_old_v2"))
    ap.add_argument("--out", default=str(DS / "clsdata_old_v3"))
    ap.add_argument("--yajang", default=str(DS / "yajang_frames/yajang.csv"))
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    if not base.exists():
        print(f"!! {base} 없음")
        return
    if out.exists():
        shutil.rmtree(out)
    print(f"v2 복사: {base.name} -> {out.name}")
    shutil.copytree(base, out)

    with open(out / "manifest.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])

    have = Counter(r["name"] for r in csv.DictReader(
        open(args.yajang, encoding="utf-8-sig")) if r["kind"] == "관로")
    pool = s20_files()

    n_norm = len(list((out / "train" / "normal").glob("*.jpg")))
    n_def = len(list((out / "train" / "defect").glob("*.jpg")))
    budget = n_norm - n_def       # 결함이 정상을 넘지 않는 선까지만
    print(f"현재 train 정상 {n_norm:,} / 결함 {n_def:,}  -> 추가 여유 {budget:,}장\n")

    rng = random.Random(args.seed)
    added = Counter()
    plan = sorted(KO, key=lambda c: have.get(KO[c], 0))    # 부족한 것부터
    for code in plan:
        cur = have.get(KO[code], 0)
        need = max(0, args.target - cur)
        files = pool.get(code, [])
        take = min(need, len(files), budget)
        if take <= 0:
            continue
        rng.shuffle(files)
        ok = 0
        for f in files[:take]:
            dst = out / "train" / "defect" / f"s20_{code}_{f.stem}.jpg"
            try:
                Image.open(f).convert("RGB").resize(
                    (args.size, args.size), Image.BILINEAR).save(dst, "JPEG", quality=88)
            except Exception:
                continue
            rows.append({**{k: "" for k in fields},
                         "split": "train", "label": "defect", "area": "s20",
                         "code": f"s20_{code}", "name": KO[code],
                         "dst": f"train\\defect\\{dst.name}"})
            ok += 1
        budget -= ok
        added[code] = ok
        print(f"  {code:<4}{KO[code]:<10} 야장 {cur:>4} + S20 {ok:>4} = {cur+ok:>4}")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n추가 {sum(added.values()):,}장 (남은 여유 {budget:,})")
    print("=== 최종 ===")
    for s in ("train", "val"):
        n = len(list((out / s / "normal").glob("*.jpg")))
        d = len(list((out / s / "defect").glob("*.jpg")))
        print(f"  {s:<6} 정상 {n:>6}  결함 {d:>6}  (결함 비율 {d/(n+d):.0%})")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

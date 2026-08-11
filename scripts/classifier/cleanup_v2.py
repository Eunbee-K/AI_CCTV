"""검토 결과를 clsdata_v2에 반영한다.

사람이 폴더를 훑으며 결함↔정상을 옮기고 지운 뒤, 남은 두 가지를 정리한다.

1. **관 밖 프레임 제거**
   영상 첫 몇 초는 로봇이 관에 들어가기 전이라 도로·맨홀이 찍힌다.
   이 프레임이 정상에 들어가면 v1의 실패를 그대로 반복한다 — v1이 배운 정상은
   딱 "관 밖"이었고, 그래서 현장 영상에서 관 안을 전부 결함이라고 했다.
   결함 쪽에 들어간 관 밖 프레임은 관 결함일 수가 없으니 오라벨이다.

   찾는 방법: v1 필터가 유일하게 제대로 배운 것이 "관 밖 = 정상"이므로,
   그 모델이 0.5 미만을 주는 프레임을 관 밖으로 본다.

2. **관로 단위 라벨 교정**
   조사표의 지적사항은 보수가 필요한 항목만 적혀 있다. 토사퇴적(DS)이나
   임시장애물(TO)처럼 눈에 보이는 결함이 지적사항에 없으면, 그 구간이 통째로
   정상으로 들어간다. 검토에서 발견된 관로를 여기에 적어 교정한다.

사용:
    python cleanup_v2.py --dry-run
    python cleanup_v2.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"
sys.path.insert(0, str(APP))

# 검토에서 확인된 것 — 관로 전체가 해당 결함이라 정상으로 둘 수 없다.
# (2026-08-11 검토: SM2-139-001은 관 바닥에 자갈이 깔려 있어 토사퇴적)
RELABEL_ALL_TO_DEFECT = {
    "SM2-139-001(관 연결부 보수)": "DS 토사퇴적이 관로 전체에 있음",
}

OUTSIDE_THRESHOLD = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v2")
    ap.add_argument("--review", default=r"E:/AI_CCTV_DATASET/clsdata_v2_review")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root, review = Path(args.data), Path(args.review)
    from backend import defect_filter

    files = []
    for s in ("train", "val"):
        for l in ("normal", "defect"):
            files += [(s, l, f) for f in sorted((root / s / l).glob("*.jpg"))]
    print(f"현재 {len(files)}장")

    probs = defect_filter.defect_probs([f for _, _, f in files])
    acts: list[tuple[str, str, str, Path]] = []   # (동작, split, label, path)

    for s, l, f in files:
        pipe = f.stem.split("__")[0]
        p = probs.get(str(f))
        outside = p is not None and p < OUTSIDE_THRESHOLD

        if outside and l == "defect":
            acts.append(("삭제(관밖인데 결함)", s, l, f))
        elif outside and s == "val":
            # val은 관 안 판별 능력을 재야 한다. 관 밖은 쉬워서 점수를 부풀린다.
            acts.append(("삭제(val 관밖)", s, l, f))
        elif pipe in RELABEL_ALL_TO_DEFECT and l == "normal" and not outside:
            # 관 밖 프레임은 제외한다. 관 밖이 토사퇴적일 수는 없다.
            acts.append(("결함으로 이동", s, l, f))

    print("\n=== 처리 계획 ===")
    for k, v in Counter((a[0], a[1]) for a in acts).most_common():
        print(f"  [{k[1]:<5}] {k[0]:<22} {v:>4}장")
    for pipe, why in RELABEL_ALL_TO_DEFECT.items():
        print(f"  * {pipe} -> 전부 결함 ({why})")

    if args.dry_run:
        print("\n(dry-run — 파일 손대지 않음)")
        return

    for act, s, l, f in acts:
        for base in (root, review):
            src = base / s / l / f.name
            if not src.exists():
                continue
            if act.startswith("삭제"):
                src.unlink()
            else:
                dst = base / s / "defect" / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    print("\n=== 정리 후 ===")
    for s in ("train", "val"):
        row = []
        for l in ("normal", "defect"):
            row.append(f"{l} {len(list((root / s / l).glob('*.jpg'))):>4}")
        print(f"  {s:<6} " + "  ".join(row))


if __name__ == "__main__":
    main()

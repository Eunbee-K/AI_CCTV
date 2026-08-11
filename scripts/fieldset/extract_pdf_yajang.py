"""CCTV조사 야장 PDF에서 프레임과 라벨을 뽑는다.

야장 PDF는 한 쪽에 CCTV 프레임 6장(2열×3행)과, 각 프레임에 대응하는 기록이 함께 있다.
기록은 이런 형태다:

    (관로)이음부이탈(소) [00:06:20]      <- 결함
    (상태)이음부(접합부)존재(중) [00:05:26]  <- 정상 상태

접두어가 결함/정상을 가른다. `관로`는 보수가 필요한 결함, `상태`는 관 내부에서
관찰된 정상 특징(이음부가 있다, 연결관이 있다 등)이다. 필터 학습에 필요한
"현장 화질의 관 내부 정상"이 바로 이 `상태` 프레임이다.

**사진과 기록의 짝짓기는 좌표로 한다.** 기록은 대응하는 사진 **바로 아래**에 놓인다.
순서대로 짝지으면 안 된다 — 라벨이 없는 사진이 하나라도 끼면 그 뒤가 전부 한 칸씩
밀린다. 실제로 상면 6쪽은 사진 6장에 기록 5건이고, 첫 사진에 기록이 없다.
사진마다 바로 아래·같은 열에 있는 기록을 찾고, 못 찾으면 그 사진만 버린다.

**조사시작/완료/중단은 제외한다** — 로봇이 관 밖이나 맨홀에 있는 장면이라
"관 내부 정상"이 아니다. 이 프레임이 정상에 섞이면 모델이 "관 밖 = 정상"만
배우게 되고, 그게 v1 필터가 실패한 원인이었다.

사용:
    python extract_pdf_yajang.py --src "C:/.../2.2. 노후관로 정밀조사 성과품-CCTV조사 야장"
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import fitz

ENTRY = re.compile(r"\((관로|상태)\)(.+?)\((대|중|소)\)\s*\[(\d+:\d+:\d+)\]")
PIPE = re.compile(r"관로번호\s*:\s*([^\s/]+)")

# 관 안이 아니라 관 밖·맨홀 장면. 정상으로 쓰면 v1의 실패를 반복한다.
SKIP_STATUS = ("조사시작", "조사완료", "조사중단")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=r"E:/AI_CCTV_DATASET/yajang_frames")
    ap.add_argument("--min-width", type=int, default=600)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, stat = [], Counter()
    for pdf in sorted(src.glob("*.pdf")):
        area = pdf.stem.split("_")[-1]          # 북면 / 상면 / 설악면
        doc = fitz.open(pdf)
        kept = skipped = 0
        for pno in range(len(doc)):
            page = doc[pno]

            imgs = []
            for im in page.get_images(full=True):
                info = doc.extract_image(im[0])
                if info["ext"] != "jpeg" or info["width"] < args.min_width:
                    continue
                for r in page.get_image_rects(im[0]):
                    imgs.append((r, info["image"]))
            if not imgs:
                continue
            imgs.sort(key=lambda t: (t[0].y0, t[0].x0))

            text = page.get_text()
            # 기록을 좌표와 함께 모은다
            recs = []
            for b in page.get_text("blocks"):
                m = ENTRY.search(b[4].replace("\n", " "))
                if m:
                    recs.append({"y": b[1], "xc": (b[0] + b[2]) / 2,
                                 "kind": m.group(1), "name": m.group(2).strip(),
                                 "grade": m.group(3), "time": m.group(4), "used": False})

            pm = PIPE.search(text)
            pipe = pm.group(1) if pm else f"p{pno}"

            for i, (rect, blob) in enumerate(imgs):
                xc = (rect.x0 + rect.x1) / 2
                # 사진 아래·같은 열에서 가장 가까운 기록
                cand = [r for r in recs if not r["used"]
                        and rect.y1 - 5 <= r["y"] <= rect.y1 + 90
                        and abs(r["xc"] - xc) <= 140]
                if not cand:
                    stat["기록 없는 사진 버림"] += 1
                    skipped += 1
                    continue
                rec = min(cand, key=lambda r: (r["y"], abs(r["xc"] - xc)))
                rec["used"] = True
                kind, name, grade, t = rec["kind"], rec["name"], rec["grade"], rec["time"]
                if kind == "상태" and any(s in name for s in SKIP_STATUS):
                    stat["관 밖(조사시작/완료) 제외"] += 1
                    continue
                label = "defect" if kind == "관로" else "normal"
                safe = re.sub(r"[^\w가-힣]", "", name)[:14]
                fn = f"{area}__{pipe}__{pno:04d}_{i}__{label}__{safe}__{grade}.jpg"
                (out / "images" / fn).write_bytes(blob)
                rows.append({"area": area, "pipe": pipe, "page": pno, "idx": i,
                             "label": label, "kind": kind, "name": name,
                             "grade": grade, "time": t, "file": fn})
                stat[label] += 1
                kept += 1
        print(f"{pdf.stem[-3:]:<5} {len(doc):>5}쪽  사용 {kept:>6}장  버림 {skipped:>5}장")
        doc.close()

    with open(out / "yajang.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== 결과 ===")
    for k, v in stat.most_common():
        print(f"  {k:<26} {v:>6}")
    print(f"\n관로 {len({r['pipe'] for r in rows})}개 / 지역 {len({r['area'] for r in rows})}곳")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

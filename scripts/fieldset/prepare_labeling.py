"""fieldset_v1(자동 추출한 현장 프레임)을 라벨링할 수 있는 형태로 깐다.

`build_field_dataset.py`가 "이 프레임에 이 결함이 있다"까지는 자동으로 알아냈다
(문서의 미터 위치 + 자막 거리 OCR). 남은 일은 **박스를 그리는 것**뿐이다.

그래서 여기서는:
  1. YOLO 표준 배치(images/labels/train,val)로 복사하고
  2. v5 모델을 낮은 임계값으로 돌려 **박스 후보를 미리 깔아준다**.
     클래스는 모델 추측이 아니라 **문서에 적힌 코드**로 강제한다 — 어디에 무엇이
     있는지는 이미 알고 있고, 모델이 못 맞히는 게 그 클래스들이기 때문이다.
  3. 정상 프레임은 빈 txt로 둔다(음성 샘플 — 오탐을 줄이는 데 쓰인다).
  4. train/val은 **관로 단위로** 나눈다. 같은 결함에서 0.5초 간격으로 뽑은 프레임을
     섞어 나누면 val이 train을 외운 걸 재는 무의미한 숫자가 된다.

출력: out/{images,labels}/{train,val} + classes.txt + data.yaml + review.csv
review.csv의 needs_box=1 인 것부터 손보면 된다.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 재질에 따라 이름만 달라지는 쌍. 화면상 구분이 안 되고 BC는 표본도 너무 적다.
# 하나로 배우게 하고, 파손/좌굴 표기는 나중에 관종(OCR)으로 나눈다.
MERGE = {"BC": "BK"}


def _edited_labels(out: Path) -> int:
    """사람이 손댄 것으로 보이는 라벨 수.

    갓 만든 사전 라벨은 이미지보다 먼저 쓰인다. 라벨이 짝 이미지보다 나중에
    저장돼 있으면 라벨링 도구로 고친 것으로 본다 — 그걸 날리기 전에 물어봐야 한다.
    """
    n = 0
    for split in ("train", "val"):
        for lp in (out / "labels" / split).glob("*.txt"):
            ip = out / "images" / split / (lp.stem + ".jpg")
            try:
                if ip.exists() and lp.stat().st_mtime > ip.stat().st_mtime + 2:
                    n += 1
            except OSError:
                pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"E:\AI_CCTV_DATASET\fieldset_v1")
    ap.add_argument("--out", default=r"E:\AI_CCTV_DATASET\fieldset_v1_label")
    ap.add_argument("--weights",
                    default=r"E:\AI_CCTV_RESULTS\test3_18class_sweep01\test3_960_e50\weights\best.pt")
    ap.add_argument("--conf", type=float, default=0.05, help="사전 박스를 뽑을 임계값")
    ap.add_argument("--max-boxes", type=int, default=3, help="프레임당 사전 박스 최대 개수")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-merge", action="store_true", help="BC를 BK에 합치지 않는다")
    ap.add_argument("--no-prelabel", action="store_true", help="모델 없이 빈 라벨만 깐다")
    ap.add_argument("--force", action="store_true",
                    help="이미 만들어둔(작업 중일 수도 있는) 데이터를 덮어쓴다")
    args = ap.parse_args()

    src = Path(args.src)
    rows = list(csv.DictReader(open(src / "manifest.csv", encoding="utf-8-sig")))
    if not rows:
        print("manifest.csv가 비어 있습니다."); return 1
    merge = {} if args.no_merge else MERGE

    def code_of(r):
        c = (r["code"] or "").strip()
        return merge.get(c, c)

    # ── 클래스 목록: 실제로 등장하는 것만, 많은 순서대로
    counts = Counter(code_of(r) for r in rows if r["kind"] == "defect" and code_of(r))
    classes = [c for c, _ in counts.most_common()]
    cls_id = {c: i for i, c in enumerate(classes)}

    # ── 관로 단위 train/val 분할
    pipes = sorted({r["pipe"] for r in rows})
    random.Random(args.seed).shuffle(pipes)
    n_val = max(1, round(len(pipes) * args.val_ratio))
    val_pipes = set(pipes[:n_val])
    # 클래스가 train에서 통째로 빠지면 학습이 안 되므로 되돌린다
    for c in classes:
        tr = {r["pipe"] for r in rows if code_of(r) == c and r["pipe"] not in val_pipes}
        if not tr:
            owner = next(r["pipe"] for r in rows if code_of(r) == c)
            val_pipes.discard(owner)
            print(f"  ! {c}가 train에 없어 {owner}를 train으로 되돌림")

    out = Path(args.out)

    # 기존 결과물 보호. 예전에는 여기서 곧바로 rmtree를 해서, 뒤쪽에서 실패하면
    # (모델 로딩 실패 등) 사진도 라벨도 없는 빈 폴더만 남았다. 실제로 한 번 날렸다.
    if out.exists() and any(out.rglob("*.jpg")):
        edited = _edited_labels(out)
        if not args.force:
            print(f"이미 데이터가 있습니다: {out}")
            if edited:
                print(f"  ! 손으로 작업한 흔적이 있는 라벨 {edited}개 — 다시 만들면 사라집니다.")
            print("  덮어쓰려면 --force 를 붙이세요. 다른 곳에 만들려면 --out 을 주세요.")
            return 1
        print(f"--force: 기존 데이터를 덮어씁니다"
              + (f" (손댄 라벨 {edited}개 포함)" if edited else ""))

    # 임시 폴더에 다 만든 뒤 마지막에 바꿔치기한다. 도중에 죽어도 기존 것은 남는다.
    work = out.with_name(out.name + ".building")
    if work.exists():
        shutil.rmtree(work)
    for split in ("train", "val"):
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)

    model = None
    if not args.no_prelabel:
        from ultralytics import YOLO
        model = YOLO(args.weights)

    review, stats = [], Counter()
    for i, r in enumerate(rows, 1):
        img = src / "frames" / r["file"]
        if not img.exists():
            continue
        split = "val" if r["pipe"] in val_pipes else "train"
        shutil.copy(img, work / "images" / split / r["file"])
        lbl = work / "labels" / split / (Path(r["file"]).stem + ".txt")

        code = code_of(r)
        lines, n_box = [], 0
        if r["kind"] == "defect" and code in cls_id and model is not None:
            res = model.predict(str(img), imgsz=960, conf=args.conf, verbose=False)[0]
            boxes = sorted(res.boxes, key=lambda b: -float(b.conf[0]))[:args.max_boxes]
            h, w = res.orig_shape
            for b in boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                if bw <= 0 or bh <= 0:
                    continue
                # 클래스는 모델 추측이 아니라 문서에 적힌 코드로 못박는다
                lines.append(f"{cls_id[code]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_box += 1

        lbl.write_text("\n".join(lines), encoding="utf-8")
        stats[f"{split}/{r['kind']}"] += 1
        review.append({
            "split": split, "file": r["file"], "pipe": r["pipe"], "kind": r["kind"],
            "code": code, "raw_label": r["raw_label"], "doc_pos_m": r["doc_pos_m"],
            "prelabel_boxes": n_box,
            # 결함인데 박스가 하나도 안 깔린 것 = 손으로 그려야 하는 것
            "needs_box": int(r["kind"] == "defect" and n_box == 0),
        })
        if i % 60 == 0:
            print(f"  ...{i}/{len(rows)}", flush=True)

    # classes.txt는 루트뿐 아니라 **라벨 폴더 안에도** 있어야 한다.
    # labelImg의 YoloReader가 라벨과 같은 폴더에서 이걸 찾고, 없으면
    # 이미지를 여는 순간 FileNotFoundError로 창이 그냥 닫힌다.
    class_txt = "\n".join(classes) + "\n"
    (work / "classes.txt").write_text(class_txt, encoding="utf-8")
    for split in ("train", "val"):
        (work / "labels" / split / "classes.txt").write_text(class_txt, encoding="utf-8")
    (work / "data.yaml").write_text(
        f"# fieldset_v1 — 현장 영상 파인튜닝용 (관로 단위 분할)\n"
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(classes)}\nnames: {classes}\n", encoding="utf-8")

    with open(work / "review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0]))
        w.writeheader()
        w.writerows(sorted(review, key=lambda x: (-x["needs_box"], x["pipe"], x["file"])))

    # 다 만들어졌으니 이제야 자리를 바꾼다 (여기까지 왔으면 실패할 일이 없다)
    if out.exists():
        old = out.with_name(out.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        out.rename(old)
    work.rename(out)
    shutil.rmtree(out.with_name(out.name + ".old"), ignore_errors=True)

    print(f"\n클래스 {len(classes)}종: {classes}")
    if merge:
        print(f"통합 적용: {merge}  (--no-merge 로 끌 수 있음)")
    print(f"\n관로 {len(pipes)}개 → train {len(pipes)-len(val_pipes)} / val {len(val_pipes)}")
    print(f"  val 관로: {sorted(val_pipes)}")
    print(f"\n{'split/종류':<18} 장수")
    for k in sorted(stats):
        print(f"  {k:<16} {stats[k]}")
    need = sum(r["needs_box"] for r in review)
    print(f"\n사전 박스가 깔린 결함 프레임 : {sum(1 for r in review if r['prelabel_boxes'])}")
    print(f"손으로 그려야 하는 프레임    : {need}")
    print(f"\n-> {out}")
    print(f"   review.csv 의 needs_box=1 부터 작업하면 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

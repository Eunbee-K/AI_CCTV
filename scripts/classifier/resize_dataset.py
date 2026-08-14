"""데이터셋을 학습 해상도로 미리 줄여 사본을 만든다. **원본은 건드리지 않는다.**

왜 미리 줄이나
    원본은 PNG 1280x720이 섞여 있어 5,260장이 5.2GB다. Colab에 올리는 것도,
    매 epoch마다 디코딩하는 것도 부담이다. 학습에 쓸 크기로 한 번 줄여두면
    5.2GB -> 500MB 수준이 되고 학습도 빨라진다.

    **JPEG로 저장한다.** PNG 무손실을 유지할 이유가 없다 — 어차피 리사이즈로
    정보가 줄고, 추론 때 들어오는 영상 프레임도 JPEG다.

가로세로비
    정사각으로 찌그러뜨린다(기존 clsdata_* 방식과 동일). 추론 경로
    `backend/defect_filter.py`가 프레임을 정사각으로 resize하므로 학습도 같아야
    한다 — 전처리가 어긋나면 잰 값이 전부 무효다.

사용:
    python resize_dataset.py --src clsdata_old_v5 --size 384
    python resize_dataset.py --src valset_bycode  --size 384
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import FILTER_DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DATASET 아래 폴더 이름")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--gray", action="store_true",
                    help="흑백으로 저장(3채널 복제). ultralytics에는 흑백 옵션이 "
                         "없어 YOLO-cls와 조건을 맞추려면 데이터를 미리 흑백으로 만든다")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = FILTER_DATA / args.src
    suffix = f"{args.size}_gray" if args.gray else f"{args.size}"
    out = Path(args.out) if args.out else FILTER_DATA / f"{args.src}_{suffix}"
    if not src.is_dir():
        raise SystemExit(f"없는 폴더: {src}")
    if out.exists():
        shutil.rmtree(out)

    n, skipped, bytes_in, bytes_out = 0, 0, 0, 0
    for f in sorted(src.rglob("*")):
        if f.is_dir() or f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rel = f.relative_to(src)
        dst = (out / rel).with_suffix(".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(f) as im:
                bytes_in += f.stat().st_size
                im = im.convert("RGB").resize((args.size, args.size), Image.BILINEAR)
                if args.gray:
                    # 학습(Grayscale(3))·추론(convert("L").convert("RGB"))과
                    # 픽셀 단위로 같은 결과다.
                    im = im.convert("L").convert("RGB")
                im.save(dst, "JPEG", quality=args.quality)
            bytes_out += dst.stat().st_size
            n += 1
            if n % 500 == 0:
                print(f"  {n:,}장…")
        except Exception as ex:
            skipped += 1
            print(f"  ! {rel}: {ex}")

    # manifest는 그대로 옮기되 확장자만 .jpg로 맞춘다
    for m in src.rglob("manifest.csv"):
        rows = list(csv.DictReader(open(m, encoding="utf-8-sig")))
        for r in rows:
            for k in ("dst", "file"):
                if k in r and r[k]:
                    r[k] = str(Path(r[k]).with_suffix(".jpg"))
        dstm = out / m.relative_to(src)
        dstm.parent.mkdir(parents=True, exist_ok=True)
        with open(dstm, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    for extra in src.glob("*.py"):                       # train_binary.py 등
        shutil.copy2(extra, out / extra.name)

    print(f"\n{n:,}장 변환 (건너뜀 {skipped})")
    print(f"  {bytes_in/1024/1024:,.0f}MB -> {bytes_out/1024/1024:,.0f}MB "
          f"({bytes_out/max(bytes_in,1):.1%})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

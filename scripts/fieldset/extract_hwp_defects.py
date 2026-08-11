"""한글(.hwp) 조사표에서 결함 사진과 그 위치(미터)를 뽑아낸다.

현장 조사표는 관로마다 hwp 한 장이고, 그 안에 결함 지점의 CCTV 프레임이 통째로
박혀 있다. 프레임에는 촬영 당시 자막이 그대로 남아 있어 거리·관로번호·관경·시각을
읽을 수 있다. 즉 "몇 미터에 무엇이 있다"를 사람이 옮겨 적지 않아도 된다.

이 결과물의 쓸모는 두 가지다.

1. **현장 화질의 결함 프레임** — 사진 자체가 라벨된 결함 이미지다.
2. **정상 구간을 특정하는 근거** — 결함 위치를 알면, 그 지점에서 충분히 떨어진
   구간은 정상으로 볼 수 있다. Stage-1 필터가 현장에서 실패한 원인이 "현장 화질의
   관 내부 정상"을 배우지 못한 것이라, 이 정상 프레임이 재학습의 핵심 재료다.

hwp는 OLE 복합문서다. BodyText/Section0에 본문이, BinData/*에 그림이 들어 있고
FileHeader의 압축 플래그가 서면 zlib(raw)로 눌려 있다.

사용:
    python extract_hwp_defects.py --src "C:/.../20260128.가평군 상면 노후하수관로 정비공사"
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import struct
import sys
import zlib
from pathlib import Path

import olefile
from PIL import Image

IMG_MIN_PIXELS = 300_000   # 로고·아이콘 정도만 거른다

# 첫 그림은 대개 관로 위치도(도면)지만 **항상은 아니다**. 켜서 돌려보니 결함 사진이
# 59장 → 55장으로 줄고 4개 관로에서 지적사항과 짝이 어긋났다(SM1-121-002는 지적 4에
# 사진 3). hwp의 이미지 스트림 순서가 문서 배치 순서와 늘 일치하지는 않는다.
# 아래 자막 형식 검사만으로 도면이 정확히 걸러지므로 기본은 끈다.
SKIP_FIRST_IMAGE = False

# 촬영 자막의 거리는 항상 3자리 0채움이다(`004.1m`). 도면에서 잘못 읽힌 값은
# `82.7m`(도면의 L=47.70m 라벨), `3.45m`처럼 형식이 깨지므로 이걸로 한 번 더 거른다.
# 크기로 나누면 안 된다 — 실제 프레임에 0.90M짜리도 1.83M짜리도 섞여 있다.
DIST_RE = re.compile(r"^\d{3}\.\d{1,2}m$")

# 잘못 촬영되어 조사 자체가 무효인 관로. 사진·지적사항 수도 어긋난다.
EXCLUDE_DIRS = {"SM1-131-003(얼음으로인한 확인 불가)"}


def hwp_paragraphs(ole: olefile.OleFileIO) -> list[str]:
    """본문 문단 텍스트. HWPTAG_PARA_TEXT(67) 레코드만 모은다."""
    head = ole.openstream("FileHeader").read()
    data = ole.openstream("BodyText/Section0").read()
    if head[36] & 1:
        data = zlib.decompress(data, -15)

    out, pos = [], 0
    while pos < len(data) - 4:
        h = struct.unpack("<I", data[pos:pos + 4])[0]
        tag, size = h & 0x3FF, (h >> 20) & 0xFFF
        pos += 4
        if size == 0xFFF:
            size = struct.unpack("<I", data[pos:pos + 4])[0]
            pos += 4
        if tag == 67:
            t = data[pos:pos + size].decode("utf-16-le", errors="ignore")
            t = "".join(c for c in t if ord(c) >= 32)
            # 제어문자가 한자로 잘못 읽힌 조각을 걷어낸다
            t = re.sub(r"[\u4e00-\u9fff]+", " ", t).strip()
            if t:
                out.append(t)
        pos += size
    return out


def hwp_images(ole: olefile.OleFileIO):
    """(스트림명, 바이트). BinData는 눌려 있을 수도, 아닐 수도 있다."""
    for entry in ole.listdir():
        if entry[0] != "BinData":
            continue
        raw = ole.openstream("/".join(entry)).read()
        for attempt in (lambda d: zlib.decompress(d, -15), lambda d: zlib.decompress(d), lambda d: d):
            try:
                data = attempt(raw)
                Image.open(io.BytesIO(data)).verify()
                yield entry[1], data
                break
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="관로 폴더들이 들어 있는 현장 폴더")
    ap.add_argument("--out", default=r"E:/AI_CCTV_DATASET/fieldset_v2_defects")
    ap.add_argument("--no-ocr", action="store_true", help="거리 판독 없이 사진만 추출")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    (out / "photos").mkdir(parents=True, exist_ok=True)

    read_distance = None
    if not args.no_ocr:
        # 앱의 판독기를 그대로 쓴다. 자막 형식이 같으므로 따로 만들 이유가 없다.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"))
        from backend.ocr import ocr_distance_from_frame
        read_distance = ocr_distance_from_frame

    rows, n_pipe = [], 0
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        if d.name in EXCLUDE_DIRS:
            print(f"{d.name[:36]:<38} 제외 (잘못 촬영)")
            continue
        hwp = next(d.glob("*.hwp"), None)
        mp4 = next(d.glob("*.mp4"), None)
        if not hwp:
            continue
        n_pipe += 1
        try:
            ole = olefile.OleFileIO(str(hwp))
            paras = hwp_paragraphs(ole)
        except Exception as exc:
            print(f"  ! {d.name}: hwp를 열지 못했습니다 — {exc}", file=sys.stderr)
            continue

        # 문단은 `관경(mm) / D450 / 지적사항 / <내용>` 블록이 사진 수만큼 반복된다.
        # 순서가 곧 사진 순서라 i번째 지적사항이 i번째 CCTV 사진의 것이다.
        notes, dia = [], ""
        for i, p in enumerate(paras):
            if p == "지적사항" and i + 1 < len(paras):
                nxt = paras[i + 1]
                if nxt not in ("관경(mm)", "지적사항") and "Line 관로" not in nxt:
                    notes.append(nxt)
            elif re.fullmatch(r"D\d{3,4}", p) and not dia:
                dia = p

        # 사진을 먼저 다 뽑고 거리 판독으로 CCTV 프레임만 골라낸다.
        # 첫 그림은 위치도이므로 스트림 이름 순으로 정렬해 건너뛴다.
        images = sorted(hwp_images(ole), key=lambda x: x[0])
        if SKIP_FIRST_IMAGE:
            images = images[1:]

        frames = []
        for name, data in images:
            try:
                im = Image.open(io.BytesIO(data))
            except Exception:
                continue
            if im.size[0] * im.size[1] < IMG_MIN_PIXELS:
                continue
            dst = out / "photos" / f"{d.name[:40]}__{Path(name).stem}.png"
            dst.write_bytes(data)
            dist = read_distance(dst) if read_distance else ""
            is_frame = bool(DIST_RE.match(dist))
            if not is_frame:
                dst.unlink(missing_ok=True)   # 위치도는 남기지 않는다
                continue
            frames.append((dst, dist, im.size))

        for i, (dst, dist, size) in enumerate(frames):
            rows.append({
                "pipe_dir": d.name,
                "video": mp4.name if mp4 else "",
                "photo": dst.name,
                "distance_m": float(dist.rstrip("m")),
                "dia": dia,
                # 사진 수와 지적사항 수가 어긋나면 짝을 짓지 않는다(잘못 붙이는 것보다 낫다)
                "issue": notes[i] if len(notes) == len(frames) else "",
                "width": size[0],
                "height": size[1],
            })

        paired = "짝맞음" if len(notes) == len(frames) else f"어긋남(지적{len(notes)}/사진{len(frames)})"
        print(f"{d.name[:36]:<38} 결함사진 {len(frames):>2}장  {paired:<20} {dia}")

    with open(out / "defects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    paired = sum(1 for r in rows if r["issue"])
    pipes_with = len({r["pipe_dir"] for r in rows})
    print(f"\n관로 {n_pipe}개 중 {pipes_with}개에서 결함 사진 {len(rows)}장 확보 "
          f"/ 지적사항 짝지음 {paired}장")
    print(f"-> {out / 'defects.csv'}")


if __name__ == "__main__":
    main()

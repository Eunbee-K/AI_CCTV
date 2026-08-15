"""test3(YOLO 18종, mAP50 0.703)를 **유출 없이** 검증할 판을 만든다.

배경
    지금 앱에 배포된 test5는 testset_bycode로 이름 정확도 90%가 나왔는데,
    같은 종류를 야장 사진만으로 다시 재니 33.3%로 무너졌다(2026-08-15).
    이후 각 스윕 결과 폴더의 `used_files.csv`로 대조해 **유출이 확정**됐다
    (testset_bycode 결함 455장 중 259장이 test5 train에 있었다).

    test3는 다르다 — `configs/sweeps/test3_18class.yaml`이 저장소에 있고,
    분할이 결정적이다(`scripts/sweep/collect_dataset.py::deterministic_pick`,
    시드 42). 즉 **test3가 학습에 실제로 쓴 파일 목록을 이 자리에서 재현할 수
    있다.** 그 목록만 빼면 유출 걱정 없는 검증셋을 만들 수 있다.

구성
    결함: test3 18종 각 20장 — **야장 우선**, 모자라면 test3가 안 쓴 S20/AIHub
          (S20/AIHub는 아까 문제가 된 그 도메인이므로 최대한 야장으로 채운다).
    정상: PJ/IN — 마찬가지로 야장 우선.

    "test3가 쓴 파일"은 train+val+test 전부를 제외한다 — test3 자체의 test
    split도 train과 같은 풀에서 나온 셔플이라, 이미지 자체는 다르지만 같은
    사람이 찍은 같은 관로의 인접 프레임일 수 있다(파이프 단위 분리가 아니다).
    보수적으로 셋 다 뺀다.

사용:
    python build_test3_holdout.py
"""

from __future__ import annotations

import csv
import io
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))              # scripts/classifier/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sweep"))
from paths import DATASET, FILTER_DATA, RESULTS  # noqa: E402
from common import load_yaml  # noqa: E402
from collect_dataset import build_class_pool, deterministic_pick  # noqa: E402
# 이름표는 build_v5_and_valset의 것을 그대로 쓴다 — 검증된 매핑을 다시 베끼면
# 괄호 유무 같은 사소한 차이로 조용히 다 빠질 수 있다(실제로 처음에 그랬다).
from build_v5_and_valset import YAJANG_NAME_TO_CODE  # noqa: E402

CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "sweeps" / "test3_18class.yaml"
DEFECT_PER_CODE = 20
NORMAL_TOTAL = 200
NORMAL_MIX = {"PJ": 0.5, "IN": 0.5}

# 야장 정상/결함 스캔에 쓰는 clsdata_old_v3 (2026-08-11 세션, 야장 전용 구성).
V3 = FILTER_DATA / "clsdata_old_v3"


def test3_used_files() -> set:
    """test3가 학습/검증/테스트 어디든 실제로 쓴 원본 파일명 전체.

    yaml + deterministic_pick을 그대로 불러 재현한다 — 별도 기록 없이도
    "그때 실제로 뽑힌 파일"을 정확히 알 수 있는 것이 이 방식의 핵심이다.
    """
    cfg = load_yaml(CFG_PATH)
    ds = cfg["dataset"]
    external_root = Path(str(ds["external_root"]).replace(
        "/home/workstation/ai_cctv/AI_CCTV_DATASET", str(DATASET)))
    seed = ds.get("split_seed", 42)

    used = set()
    for class_code, spec in ds["classes"].items():
        pool = build_class_pool(external_root, class_code, spec["sources"])
        picked = deterministic_pick(pool, spec["quota"], seed, class_code)
        for split in ("train", "val", "test"):
            for img_path, _kind, _label_path, _code in picked[split]:
                used.add(img_path.name)
    return used


def scan_yajang() -> tuple[dict, list]:
    """clsdata_old_v3에서 야장 결함(코드별)과 야장 정상(PJ 대응) 원본 경로."""
    defects, normals = defaultdict(list), []
    with open(V3 / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["area"] not in ("북면", "상면", "설악면"):
                continue
            src = V3 / r["dst"].replace("\\", "/")
            if not src.exists():
                continue
            if r["label"] == "defect":
                code = YAJANG_NAME_TO_CODE.get(r.get("name", ""))
                if code:
                    defects[code].append(src)
            else:
                normals.append(src)
    return defects, normals


def scan_s20_aihub(used: set) -> dict:
    """S20/S22 + AIHub에서 test3가 안 쓴 이미지. code_of는 build_testset_bycode와 같은 규칙."""
    import re
    S20 = DATASET / "original" / "rename_data_s20_s22_bbox"
    AIHUB = DATASET / "original" / "aihub_data_bbox" / "image"

    def code_of(folder: str):
        if folder in ("IN", "PJ"):
            return folder
        m = re.match(r"^S22_([A-Z]{2})_", folder)
        if m:
            return m.group(1)
        m = re.match(r"^([A-Z]{2,3})_S20_", folder)
        if m:
            return m.group(1)
        if re.fullmatch(r"[A-Z]{2,3}", folder):
            return folder
        return None

    pool = defaultdict(list)
    for root in (S20, AIHUB):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            code = code_of(d.name)
            if code is None:
                continue
            for f in d.rglob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.name not in used:
                    pool[code].append(f)
    return pool


def _verify_no_leak(rows: list) -> None:
    """만든 검증셋이 test3 학습분과 정말 안 겹치는지 **결과 기록으로 재확인**한다.

    위 `test3_used_files()`는 yaml+시드로 재현한 것이라, 재현이 어긋나면 유출이
    있는데도 없다고 나올 수 있다. 스윕 결과 폴더에 남는 `used_files.csv`는 그
    run이 **실제로 쓴** 파일 목록이므로 이쪽이 근거로 확실하다.

    2026-08-15에 `testset_bycode`를 이 파일과 대조해보니 결함 455장 중 259장이
    test5 학습분이었다 — 그 판으로 잰 YOLO 성적(90%)이 전부 무의미했다.
    같은 일이 반복되지 않게 만들 때마다 자동으로 확인한다.
    """
    used_csv = (RESULTS / "test3_18class_sweep01" / "test3_960_e50" / "used_files.csv")
    if not used_csv.exists():
        print(f"\n[검증 생략] used_files.csv 없음: {used_csv}")
        return
    with open(used_csv, encoding="utf-8-sig") as f:
        # dest_filename은 `yolo_txt_BK__원본명.png`처럼 접두어가 붙는다.
        # 반드시 source_path(원본 절대경로)의 파일명으로 비교해야 한다.
        used = {Path(r["source_path"]).name for r in csv.DictReader(f)}
    mine = {Path(r["src"]).name for r in rows}
    overlap = used & mine
    print(f"\n[유출 검증] test3 used_files.csv {len(used):,}개와 대조 -> "
          f"겹침 {len(overlap)}개", end="")
    if overlap:
        print("  !! 유출 있음")
        for n in list(overlap)[:10]:
            print(f"    {n}")
    else:
        print("  OK")


def main():
    random.seed(0)
    out = FILTER_DATA / "test3_holdout"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)   # 참고용 — 검증은 분류 정확도만 볼 것이므로 라벨은 비워둠

    print("test3 학습에 실제로 쓴 파일 목록 재현 중...")
    used = test3_used_files()
    print(f"  {len(used):,}개 제외 대상\n")

    ya_def, ya_nor = scan_yajang()
    s2_pool = scan_s20_aihub(used)

    codes = ["BK", "CC", "CL", "CM", "DS", "HL", "JD", "JF", "JS",
             "LP", "LS", "PO", "RT", "SD", "SG", "TO"]

    rows = []
    print(f"{'코드':<6}{'야장':>6}{'S20/AIHub':>11}{'합계':>6}")
    for code in codes:
        ya = [f for f in ya_def.get(code, []) if f.name not in used]
        random.shuffle(ya)
        take_ya = ya[:DEFECT_PER_CODE]
        need = DEFECT_PER_CODE - len(take_ya)
        s2 = s2_pool.get(code, [])
        random.shuffle(s2)
        take_s2 = s2[:need] if need > 0 else []
        for src in take_ya:
            dst = out / "images" / f"{code}__yajang__{src.name}"
            shutil.copy2(src, dst)
            rows.append({"code": code, "from": "yajang", "file": dst.name, "src": str(src)})
        for src in take_s2:
            dst = out / "images" / f"{code}__s20aihub__{src.name}"
            shutil.copy2(src, dst)
            rows.append({"code": code, "from": "s20aihub", "file": dst.name, "src": str(src)})
        total = len(take_ya) + len(take_s2)
        flag = "" if total == DEFECT_PER_CODE else "  <- 부족"
        print(f"  {code:<6}{len(take_ya):>6}{len(take_s2):>11}{total:>6}{flag}")

    print(f"\n정상 {NORMAL_TOTAL}장 — PJ {NORMAL_MIX['PJ']:.0%} / IN {NORMAL_MIX['IN']:.0%}, 야장 우선")
    ya_nor_avail = [f for f in ya_nor if f.name not in used]
    random.shuffle(ya_nor_avail)
    for code, ratio in NORMAL_MIX.items():
        want = round(NORMAL_TOTAL * ratio)
        if code == "PJ":
            take_ya = ya_nor_avail[:want]
            ya_nor_avail = ya_nor_avail[want:]
        else:
            take_ya = []
        need = want - len(take_ya)
        s2 = [f for f in s2_pool.get(code, [])] if need > 0 else []
        random.shuffle(s2)
        take_s2 = s2[:need]
        for src in take_ya:
            dst = out / "images" / f"{code}__yajang__{src.name}"
            shutil.copy2(src, dst)
            rows.append({"code": code, "from": "yajang", "file": dst.name, "src": str(src)})
        for src in take_s2:
            dst = out / "images" / f"{code}__s20aihub__{src.name}"
            shutil.copy2(src, dst)
            rows.append({"code": code, "from": "s20aihub", "file": dst.name, "src": str(src)})
        print(f"  {code:<6}{'야장':>6} {len(take_ya):>4}  {'S20/AIHub':>10} {len(take_s2):>4}")

    _verify_no_leak(rows)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["code", "from", "file", "src"])
        w.writeheader()
        w.writerows(rows)

    n_ya = sum(1 for r in rows if r["from"] == "yajang")
    print(f"\n총 {len(rows)}장 · 야장 {n_ya} ({n_ya/len(rows):.0%}) · "
          f"S20/AIHub {len(rows)-n_ya}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

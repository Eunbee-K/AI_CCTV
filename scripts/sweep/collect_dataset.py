"""외장하드 원본(서울시 rename_data + AI Hub aihub_data_bbox)에서 지정한
클래스들만 골라, 클래스별로 정해진 정확한 장수(quota)만큼 YOLO 학습용
임시 데이터셋(images/labels train/val/test + data.yaml)을 만든다.

원본은 폴더 구조만 다르고 라벨은 전부 전역 class id YOLO txt라
(rename_data_s20_s22_bbox는 preconvert로 전량 변환 완료), XML/JSON을
파싱할 필요 없이 remap_global_yolo_lines로 로컬 class_id만 바꿔 읽는다.
라벨 라인의 class id 뒤 토큰은 그대로 보존하므로 bbox(4개 좌표)든
seg 폴리곤(좌표 다수)이든 동일하게 처리된다.

- seoul_txt: 서울시 2020/2022 데이터. `<하위폴더>/images/`, `<하위폴더>/labels/` 구조.
- yolo_txt: AI Hub bbox 데이터(`aihub_data_bbox/image/<코드>/`,
  `aihub_data_bbox/labels/<코드>/`) 구조.
- dir_pair: 이미지/라벨 폴더 경로를 직접 지정. aihub_data_seg처럼 이미지 폴더명
  ("1-3.파손(Broken-Pipe,BK)")과 라벨 폴더명("BK")이 규칙적으로 대응하지 않는
  경우에 사용.

두 가지 모드:

- scan:        실제로 복사하지 않고, 클래스별로 모을 수 있는 이미지 수가
               설정한 quota(train+val+test)를 채울 만큼 되는지 미리 점검한다.
- materialize: 실제로 이미지+라벨을 dest 폴더에 복사하고 data.yaml을 만든다.
               run_sweep.py가 각 run 시작 전에 이 함수를 호출한다.
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import yaml

from common import ensure_dir, load_yaml, now_iso
from label_formats import remap_global_yolo_lines

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _label_index_by_stem(labels_dir: Path) -> dict:
    """labels_dir를 한 번만 훑어서 stem -> .txt 경로 매핑을 만든다.

    이미지 하나하나마다 Path.exists()를 부르면 폴더당 파일 수가 많을 때
    (AI Hub의 IN=9만장, PJ=4.5만장 등) 개별 stat 호출이 누적되어 매우
    느려지므로, 디렉터리 목록을 한 번만 읽어 메모리에서 매칭한다.
    """
    return {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}


def iter_image_label_pairs(images_dir: Path, labels_dir: Path):
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return
    label_index = _label_index_by_stem(labels_dir)
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = label_index.get(img_path.stem)
        if label_path is not None:
            yield img_path, label_path


def build_class_pool(external_root: Path, class_code: str, sources: list) -> list:
    """해당 클래스의 모든 소스를 훑어서 (img_path, kind, label_path, code) 풀을 만든다."""
    pool = []
    for src in sources:
        kind = src["type"]
        if kind == "seoul_txt":
            source_dir = external_root / src["path"]
            pairs = iter_image_label_pairs(source_dir / "images", source_dir / "labels")
            code = None
        elif kind == "yolo_txt":
            base_dir = external_root / src["path"]
            code = src.get("code", class_code)
            pairs = iter_image_label_pairs(base_dir / "image" / code, base_dir / "labels" / code)
        elif kind == "dir_pair":
            pairs = iter_image_label_pairs(external_root / src["images"], external_root / src["labels"])
            code = None
        else:
            raise ValueError(f"알 수 없는 source type: {kind}")
        for img_path, label_path in pairs:
            pool.append((img_path, kind, label_path, code))
    return pool


def deterministic_pick(pool: list, quota: dict, seed, class_code: str) -> dict:
    """pool에서 quota(train/val/test)만큼 결정적으로 뽑는다. 부족하면 있는 만큼
    같은 비율로 나누고 경고를 반환한다."""
    keyed = sorted(pool, key=lambda item: str(item[0]))
    rng = random.Random(f"{seed}:{class_code}")
    rng.shuffle(keyed)

    requested_total = quota["train"] + quota["val"] + quota["test"]
    available = len(keyed)
    warning = None

    if available < requested_total:
        warning = (
            f"[{class_code}] 요청한 {requested_total}장을 채울 수 없음 "
            f"(실제 사용 가능: {available}장) -> 비율 유지한 채 축소"
        )
        ratio = available / requested_total if requested_total else 0
        n_train = round(quota["train"] * ratio)
        n_val = round(quota["val"] * ratio)
        n_test = available - n_train - n_val
    else:
        n_train, n_val, n_test = quota["train"], quota["val"], quota["test"]

    picked = keyed[: n_train + n_val + n_test]
    return {
        "train": picked[:n_train],
        "val": picked[n_train:n_train + n_val],
        "test": picked[n_train + n_val:n_train + n_val + n_test],
        "warning": warning,
    }


def convert_one(item, class_to_id: dict) -> list:
    """(img_path, kind, label_path, code) 하나를 YOLO 라벨 라인 목록으로 변환.

    모든 라벨이 이미 전역 class id YOLO txt이므로 remap_global_yolo_lines로
    로컬 class_id만 바꿔주면 된다.
    """
    _img_path, _kind, label_path, _code = item
    return remap_global_yolo_lines(label_path, class_to_id)


def scan(cfg: dict) -> None:
    ds = cfg["dataset"]
    external_root = Path(ds["external_root"])
    classes = ds["classes"]

    print(f"external_root = {external_root}")
    print(f"target classes = {list(classes.keys())} (총 {len(classes)}개)\n")

    total_requested = {"train": 0, "val": 0, "test": 0}
    total_available = 0

    for class_code, spec in classes.items():
        quota = spec["quota"]
        pool = build_class_pool(external_root, class_code, spec["sources"])
        requested = quota["train"] + quota["val"] + quota["test"]
        status = "OK" if len(pool) >= requested else "!! 부족"

        print(f"[{class_code}] 소스 {len(spec['sources'])}개 -> 사용 가능(폴더 기준) {len(pool)}장 / 요청 {requested}장  {status}")
        for src in spec["sources"]:
            loc = src.get("path") or f"{src.get('images')} + {src.get('labels')}"
            print(f"    - {src['type']}: {loc}" + (f" (code={src['code']})" if src.get("code") else ""))

        total_available += min(len(pool), requested)
        for k in ("train", "val", "test"):
            total_requested[k] += quota[k]

    print(f"\n합계: 요청 train/val/test = {total_requested}")


def materialize(cfg: dict, dest_dir) -> dict:
    ds = cfg["dataset"]
    external_root = Path(ds["external_root"])
    classes_cfg = ds["classes"]
    class_codes = list(classes_cfg.keys())
    class_to_id = {code: i for i, code in enumerate(class_codes)}
    seed = ds.get("split_seed", 42)

    dest_dir = Path(dest_dir)
    for split in ("train", "val", "test"):
        ensure_dir(dest_dir / "images" / split)
        ensure_dir(dest_dir / "labels" / split)

    image_counts = {"train": 0, "val": 0, "test": 0}
    box_counts = {split: {c: 0 for c in class_codes} for split in ("train", "val", "test")}
    warnings = []
    used_files_rows = []

    for class_code, spec in classes_cfg.items():
        pool = build_class_pool(external_root, class_code, spec["sources"])
        picked = deterministic_pick(pool, spec["quota"], seed, class_code)
        if picked["warning"]:
            warnings.append(picked["warning"])

        is_background = spec.get("background", False)

        for split in ("train", "val", "test"):
            for item in picked[split]:
                img_path, kind, _label_path, _code = item
                yolo_lines = convert_one(item, class_to_id)
                if not yolo_lines and not is_background:
                    continue

                safe_prefix = f"{kind}_{class_code}"
                dest_img = dest_dir / "images" / split / f"{safe_prefix}__{img_path.name}"
                dest_lbl = dest_dir / "labels" / split / f"{safe_prefix}__{img_path.stem}.txt"

                shutil.copy2(img_path, dest_img)
                dest_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")

                image_counts[split] += 1
                for line in yolo_lines:
                    cid = int(line.split()[0])
                    box_counts[split][class_codes[cid]] += 1

                used_files_rows.append({
                    "split": split,
                    "class": class_code,
                    "source_kind": kind,
                    "source_path": str(img_path),
                    "dest_filename": dest_img.name,
                })

    used_files_path = dest_dir / "used_files.csv"
    with open(used_files_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "source_kind", "source_path", "dest_filename"])
        writer.writeheader()
        writer.writerows(used_files_rows)

    names_map = {i: c for i, c in enumerate(class_codes)}
    data_yaml_path = dest_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(dest_dir),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names_map,
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    manifest = {
        "created_at": now_iso(),
        "split_seed": seed,
        "classes": class_codes,
        "image_counts": image_counts,
        "box_counts": box_counts,
        "warnings": warnings,
    }
    with open(dest_dir / "materialize_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["scan", "materialize"])
    ap.add_argument("--config", required=True, help="스윕 설정 YAML 경로")
    ap.add_argument("--dest", help="materialize 전용: 데이터셋을 만들 임시 폴더 경로")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if args.mode == "scan":
        scan(cfg)
    else:
        if not args.dest:
            ap.error("materialize 모드는 --dest 가 필요합니다")
        manifest = materialize(cfg, args.dest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

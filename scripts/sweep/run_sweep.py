"""워크스테이션에서 사람 개입 없이 하이퍼파라미터 스윕을 순차 실행한다.

run 하나마다:
  1. 외장하드 원본에서 BK/DS 이미지+라벨만 골라 임시 데이터셋을 만들고 (collect_dataset.materialize)
  2. YOLO 학습을 돌리고 (ultralytics)
  3. results.csv에서 지표를 뽑아 sweep_summary.csv에 한 줄 추가하고
  4. 임시 데이터셋 폴더를 삭제해서 디스크를 비운다 (외장하드 용량 문제 대응)

인터넷이 없는 환경에서 그대로 돌아가도록 설계했다 (외부 API 호출 없음).
중간에 프로세스가 죽거나 워크스테이션이 재부팅돼도, sweep_state.json에
status="done"으로 남은 run은 건너뛰고 이어서 진행한다.
"""
import argparse
import json
import shutil
import traceback
from pathlib import Path

from common import StateStore, ensure_dir, load_yaml, now_iso, setup_logger
import collect_dataset
import report


def default_state_path(cfg: dict) -> Path:
    results_root = Path(cfg["workspace"]["results_root"])
    return results_root.parent / f"{cfg['sweep_name']}.state.json"


def run_one(cfg: dict, run_cfg: dict, log) -> dict:
    sweep_name = cfg["sweep_name"]
    run_name = run_cfg["name"]

    dataset_root = Path(cfg["workspace"]["dataset_root"]) / run_name
    results_root = Path(cfg["workspace"]["results_root"])
    run_result_dir = results_root / sweep_name / run_name

    log.info(f"[{run_name}] 1/3 데이터셋 구성 중 (외장하드 -> {dataset_root})")
    manifest = collect_dataset.materialize(cfg, dataset_root)
    log.info(f"[{run_name}] 데이터셋 완료: {manifest['image_counts']}")

    log.info(f"[{run_name}] 2/3 학습 시작 (imgsz={run_cfg['imgsz']}, epochs={run_cfg['epochs']}, batch={run_cfg.get('batch', 'auto')})")
    from ultralytics import YOLO  # 학습 시점에만 import (스캔/설계 단계에서 torch 로딩 방지)

    model = YOLO(cfg["model"])
    train_metrics = model.train(
        data=str(dataset_root / "data.yaml"),
        imgsz=run_cfg["imgsz"],
        epochs=run_cfg["epochs"],
        batch=run_cfg.get("batch", "auto"),
        device=cfg.get("device", 0),
        cache=cfg.get("cache", True),
        cos_lr=cfg.get("cos_lr", True),
        patience=cfg.get("patience", 50),
        project=str(results_root / sweep_name),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    log.info(f"[{run_name}] 3/3 결과 정리 중")
    metrics = report.parse_ultralytics_results_csv(run_result_dir / "results.csv")

    # 클래스별 mAP 저장 — 다음 실험에서 어떤 클래스를 빼거나 합칠지 판단하는 근거.
    # val에 GT 박스가 있어서 실제로 평가된 클래스만 기록하고, 평가 불가능한
    # 클래스(IN/PJ 같은 배경 클래스)는 null로 남긴다 — ultralytics의 maps 배열은
    # 미평가 클래스에 전체 평균을 채워 넣어서 그대로 쓰면 오해를 부른다.
    # 실패해도 run 자체는 성공으로 처리한다 (ultralytics 버전에 따라 속성이 다를 수 있음).
    try:
        names = train_metrics.names
        box = train_metrics.box
        evaluated = [int(c) for c in getattr(box, "ap_class_index", [])]
        ap50 = box.ap50   # 평가된 클래스 순서(ap_class_index 순)의 AP50
        ap = box.ap       # 같은 순서의 AP50-95
        per_class = {name: None for name in names.values()}
        for i, c in enumerate(evaluated):
            per_class[names[c]] = {
                "map50": round(float(ap50[i]), 4),
                "map50_95": round(float(ap[i]), 4),
            }
        with open(run_result_dir / "per_class_metrics.json", "w", encoding="utf-8") as f:
            json.dump(per_class, f, ensure_ascii=False, indent=2)
        log.info(f"[{run_name}] 클래스별 mAP 저장: per_class_metrics.json")
    except Exception as e:
        log.warning(f"[{run_name}] 클래스별 mAP 저장 실패 (무시하고 진행): {e}")

    manifest_copy = run_result_dir / "materialize_manifest.json"
    shutil.copy2(dataset_root / "materialize_manifest.json", manifest_copy)

    used_files_copy = run_result_dir / "used_files.csv"
    shutil.copy2(dataset_root / "used_files.csv", used_files_copy)

    if not cfg["workspace"].get("keep_last_weight", False):
        last_pt = run_result_dir / "weights" / "last.pt"
        last_pt.unlink(missing_ok=True)

    row = {
        "sweep_name": sweep_name,
        "run_name": run_name,
        "status": "done",
        "imgsz": run_cfg["imgsz"],
        "epochs": run_cfg["epochs"],
        "batch": run_cfg.get("batch", "auto"),
        "model": cfg["model"],
        "train_images": manifest["image_counts"]["train"],
        "val_images": manifest["image_counts"]["val"],
        "test_images": manifest["image_counts"]["test"],
        "finished_at": now_iso(),
        **metrics,
    }
    report.append_run_result(cfg["workspace"]["summary_csv"], row)
    return row


def run_sweep(config_path: str, state_arg: str = None) -> None:
    cfg = load_yaml(config_path)
    sweep_name = cfg["sweep_name"]

    results_root = ensure_dir(cfg["workspace"]["results_root"])
    ensure_dir(cfg["workspace"]["dataset_root"])

    state_path = Path(state_arg) if state_arg else default_state_path(cfg)
    state = StateStore(state_path)

    log = setup_logger(sweep_name, results_root.parent / f"{sweep_name}.log")
    log.info(f"=== 스윕 시작: {sweep_name} ({len(cfg['runs'])}개 run) ===")

    for run_cfg in cfg["runs"]:
        run_name = run_cfg["name"]
        entry = state.get(run_name)

        if entry.get("status") == "done":
            log.info(f"[{run_name}] 이미 완료됨 -> 건너뜀")
            continue

        state.set(run_name, status="running", started_at=now_iso(), error=None)
        dataset_root = Path(cfg["workspace"]["dataset_root"]) / run_name

        try:
            row = run_one(cfg, run_cfg, log)
            state.set(run_name, status="done", finished_at=now_iso(), metrics=row)
            log.info(f"[{run_name}] 완료. best mAP50-95={row.get('best_map50_95')}")
        except Exception as e:
            log.error(f"[{run_name}] 실패: {e}")
            log.error(traceback.format_exc())
            state.set(run_name, status="failed", finished_at=now_iso(), error=str(e))
        finally:
            if dataset_root.exists():
                shutil.rmtree(dataset_root, ignore_errors=True)
                log.info(f"[{run_name}] 임시 데이터셋 삭제 완료 (디스크 확보)")

    done = sum(1 for r in state.all_runs().values() if r.get("status") == "done")
    failed = sum(1 for r in state.all_runs().values() if r.get("status") == "failed")
    log.info(f"=== 스윕 종료: 완료 {done}건 / 실패 {failed}건 / 총 {len(cfg['runs'])}건 ===")

    (results_root / sweep_name / "SWEEP_DONE.flag").parent.mkdir(parents=True, exist_ok=True)
    (results_root / sweep_name / "SWEEP_DONE.flag").write_text(now_iso(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", required=True, nargs="+",
        help="스윕 설정 YAML 경로. 여러 개 주면 앞의 스윕이 전부 끝난 뒤 다음 스윕을 이어서 실행 (예: test3 -> test4_seg)",
    )
    ap.add_argument("--state", help="진행 상태 JSON 경로 (config가 1개일 때만 사용 가능, 생략 시 자동 생성)")
    args = ap.parse_args()

    if args.state and len(args.config) > 1:
        ap.error("--state는 --config가 1개일 때만 지정할 수 있습니다 (여러 개면 각자 자동 생성)")

    for config_path in args.config:
        run_sweep(config_path, args.state)


if __name__ == "__main__":
    main()

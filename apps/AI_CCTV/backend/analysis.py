import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from . import defect_filter, session_store, ws_manager
from .annotate import annotate_frame
from .config import (FILTER_MAX_MISS_ROWS, FILTER_MODE, FILTER_THRESHOLD,
                     FRAME_INTERVAL)
from .frames import extract_frames, seconds_to_mmss
from .ocr import ocr_distance_from_frame, ocr_overlay_metadata, normalize_diameter_text, try_ocr_find_range
from .rows import mark_dist_conflicts
from .state import OCR_TO_META, state
from .yolo_infer import call_yolo, init_yolo
from .yolo_remote import call_yolo_remote


def ensure_yolo_loaded():
    if state.yolo_model is None and state.yolo_load_error is None:
        state.yolo_model, state.yolo_load_error = init_yolo()
    return state.yolo_model, state.yolo_load_error


def start_analysis() -> Optional[str]:
    if state.analyzing:
        return "Analysis already running"
    if not state.video_queue:
        return "No videos to analyze"
    if not state.pipe_condition:
        return "관로 구분(신설/노후)을 선택한 뒤 분석을 실행하세요."

    # 원격(Colab) 추론 서버가 설정돼 있으면 로컬 YOLO 모델은 로드하지 않는다.
    if not state.remote_yolo_url:
        model, err = ensure_yolo_loaded()
        if not model:
            return f"YOLO model not loaded: {err}"

    state.analyzing = True
    threading.Thread(target=_run_batch_thread, daemon=True).start()
    return None


def _fill_report_meta_from_ocr(video_name: str, meta: dict) -> None:
    """영상 자막에서 읽은 값으로 그 관로의 보고서 정보를 채운다.

    맨홀번호·관종 등은 관로마다 다르므로 영상별로 따로 저장한다.
    사용자가 이미 입력해둔 값은 덮어쓰지 않는다 — 검수해서 고친 것을 다시
    OCR 결과로 되돌리면 안 되기 때문이다.
    """
    pipe_meta = state.pipe_meta(video_name)
    filled = []
    for ocr_key, meta_key in OCR_TO_META.items():
        value = str(meta.get(ocr_key) or "").strip()
        if value and not pipe_meta.get(meta_key):
            pipe_meta[meta_key] = value
            filled.append(f"{meta_key}={value}")
    if filled:
        ws_manager.log(f" - 보고서 정보 자동 입력: {', '.join(filled)}")


def _update_travel_distance(video_name: str) -> None:
    """이 관로의 총주행거리·연장을 마지막(가장 먼) 거리 표기로 채운다.

    야장에서 연장과 총주행거리는 완주한 경우 같은 값이고, 미주행거리는 0이다.
    """
    v = state.video_data_map.get(video_name) or {}
    best = 0.0
    for r in v.get("rows", []):
        m = re.search(r'(\d+(?:\.\d+)?)', str(r.get("dist") or ""))
        if m:
            best = max(best, float(m.group(1)))
    if best <= 0:
        return

    pipe_meta = state.pipe_meta(video_name)
    text = f"{best:.2f}m"
    for key in ("총주행거리", "연장"):
        if not pipe_meta.get(key):
            pipe_meta[key] = text
    if not pipe_meta.get("미주행거리"):
        pipe_meta["미주행거리"] = "0.0m"
    ws_manager.log(f" - 총주행거리/연장 = {text} (미주행 0.0m)")


def _run_filter(frames, video_name: str) -> Dict[str, float]:
    """프레임별 결함 확률. 필터가 꺼져 있거나 쓸 수 없으면 빈 dict."""
    if FILTER_MODE == "off" or not frames:
        return {}

    ok, why = defect_filter.availability()
    if not ok:
        # 필터는 보조 장치다. 없다고 분석을 멈추지 않고 알리기만 한다.
        ws_manager.log(f" - Filter unavailable, skipping: {why}", "WARN")
        return {}

    t0 = time.time()
    probs = defect_filter.defect_probs(frames)
    hit = sum(1 for p in probs.values() if p >= FILTER_THRESHOLD)
    ws_manager.log(
        f" - Filter({FILTER_MODE}): {hit}/{len(probs)} frames look defective "
        f"(threshold {FILTER_THRESHOLD}, {time.time() - t0:.1f}s)"
    )
    return probs


def _filter_only_frames(frames, probs: Dict[str, float], yolo_times: set) -> List[tuple]:
    """필터는 결함이라 보는데 YOLO는 아무것도 못 찾은 프레임을 연속 구간으로 묶는다.

    프레임마다 행을 만들면 표가 넘친다. 이어진 구간은 확률이 가장 높은 한 장만
    대표로 남겨, 검토자가 그 지점으로 가서 눈으로 확인할 수 있게 한다.
    """
    flagged = []
    for f in frames:
        try:
            t = int(Path(f).stem)   # extract_frames가 {초:06d}.jpg로 저장한다
        except ValueError:
            continue
        if t in yolo_times:
            continue
        p = probs.get(str(f))
        if p is not None and p >= FILTER_THRESHOLD:
            flagged.append((t, Path(f), p))

    flagged.sort()
    runs: List[List[tuple]] = []
    for item in flagged:
        if runs and item[0] - runs[-1][-1][0] <= FRAME_INTERVAL:
            runs[-1].append(item)
        else:
            runs.append([item])
    return [max(run, key=lambda x: x[2]) for run in runs]


def _apply_parallel_filter(v_data: dict, frames, probs: Dict[str, float], path) -> None:
    """병렬 모드: 두 판단이 엇갈린 곳을 표에 드러낸다.

    - YOLO는 결함이라는데 필터는 정상 → 오탐 의심. 비고에 표시만 하고 지우지는 않는다.
    - 필터는 결함이라는데 YOLO는 못 찾음 → 누락 의심. 행을 새로 만들어 검토 대상에 올린다.
      이건 지금까지 존재 자체를 알 수 없던 경우라, 병렬로 돌리는 가장 큰 이유다.
    """
    suspect_fp = 0
    for r in v_data["rows"]:
        p = r.get("filter_prob")
        if p is not None and p < FILTER_THRESHOLD:
            suspect_fp += 1
            r["note"] = (r["note"] + " " if r["note"] else "") + "오탐 의심(필터)"

    yolo_times = {r["time"] for r in v_data["rows"]}
    missed = _filter_only_frames(frames, probs, yolo_times)
    capped = missed[:FILTER_MAX_MISS_ROWS]

    for t_sec, frame_fp, p in capped:
        v_data["rows"].append({
            "time": t_sec,
            "dist": ocr_distance_from_frame(frame_fp) if frame_fp.exists() else "",
            "defects": [],          # 필터는 종류를 모른다. 사람이 채운다.
            "note": "확인필요(필터만 감지)",
            "frame_path": frame_fp,
            "frame_annot_path": None,
            "boxes": [],
            "boxes_norm": [],
            "fp": False,
            "grade": "중",
            "direction": "",
            "filter_prob": p,
            # 수동 행과 같은 취급 — 거리 그룹에 접혀 들어가면 찾을 수 없다.
            "manual": True,
            "filter_only": True,
        })

    ws_manager.log(
        f" - Filter cross-check: {suspect_fp} YOLO detections look like false positives, "
        f"{len(missed)} frames flagged that YOLO missed"
        + (f" (showing {len(capped)})" if len(capped) < len(missed) else "")
    )


def _run_batch_thread():
    try:
        errors = []
        model_stats = {"yolo": 0}
        total = len(state.video_queue)

        ws_manager.log(">>> Analysis Started <<<")

        for idx, path in enumerate(state.video_queue):
            ws_manager.log(f"Analyzing ({idx + 1}/{total}): {path.name}")
            ws_manager.progress(path.name, idx + 1, total, "start")

            v_data = state.video_data_map[path.name]
            v_data["rows"] = []

            ws_manager.progress(path.name, idx + 1, total, "ocr_range")
            s, e = try_ocr_find_range(path)
            if s >= e:
                cap_tmp = cv2.VideoCapture(str(path))
                total_frames = cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                fps_tmp = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
                cap_tmp.release()
                e = int(total_frames / fps_tmp) if fps_tmp > 0 else 0
                s = 0

            ws_manager.log(f" - Range: {seconds_to_mmss(s)} ~ {seconds_to_mmss(e)}")

            ws_manager.progress(path.name, idx + 1, total, "extract_frames")
            frames = extract_frames(path, state.frames_root, s, e, FRAME_INTERVAL)
            ws_manager.log(f" - Frames extracted: {len(frames)}")

            # Stage-1 필터. parallel이면 전 프레임을 재고 YOLO 결과와 대조하고,
            # series면 통과한 프레임만 YOLO에 넘긴다.
            probs = _run_filter(frames, path.name)
            yolo_frames = frames
            if probs and FILTER_MODE == "series":
                yolo_frames = [f for f in frames if probs.get(str(f), 1.0) >= FILTER_THRESHOLD]
                ws_manager.log(
                    f" - Filter(series): {len(yolo_frames)}/{len(frames)} frames pass "
                    f"→ {len(frames) - len(yolo_frames)} skipped"
                )

            ws_manager.progress(path.name, idx + 1, total, "yolo")
            if state.remote_yolo_url:
                ws_manager.log(f" - Running YOLO inference on Colab ({state.remote_yolo_url})...")
                merged_rows, yolo_err = call_yolo_remote(yolo_frames, state.remote_yolo_url)
            else:
                ws_manager.log(" - Running YOLO inference (local CPU)...")
                merged_rows, yolo_err = call_yolo(state.yolo_model, yolo_frames)
            if yolo_err:
                ws_manager.log(f"   ! YOLO Error: {yolo_err}", "ERROR")
                errors.append(yolo_err)

            model_stats["yolo"] += len(merged_rows)
            ws_manager.log(f" - YOLO found {len(merged_rows)} issue frames.")

            ws_manager.progress(path.name, idx + 1, total, "ocr_meta")
            meta = ocr_overlay_metadata(frames)
            pid = path.stem
            dia = normalize_diameter_text(meta.get("diameter_text", ""))
            site = meta.get("site_name", "")

            v_data["pipe_id"] = pid
            v_data["dia"] = dia
            ws_manager.log(f" - OCR meta: site='{site}', pipe='{pid}', dia='{dia}'")

            if site and not state.site_name:
                state.site_name = site

            _fill_report_meta_from_ocr(path.name, meta)

            for item in merged_rows:
                t_sec = item.get("time_s", 0)
                frame_fp = state.frames_root / path.stem / f"{t_sec:06d}.jpg"

                dist_ai = item.get("distance_text", "")
                dist_ocr = ""
                if frame_fp.exists():
                    dist_ocr = ocr_distance_from_frame(frame_fp)

                dist_final = dist_ocr if dist_ocr else dist_ai

                # bbox+라벨+신뢰도를 그린 프레임(보고서용)과
                # UI 오버레이용 정규화 좌표를 함께 만든다
                boxes = item.get("boxes", [])
                annot_fp, norm_boxes = (None, [])
                if boxes and frame_fp.exists():
                    annot_fp, norm_boxes = annotate_frame(frame_fp, boxes)

                v_data["rows"].append({
                    "time": t_sec,
                    "dist": dist_final,
                    "defects": item.get("defects", []),
                    "note": "",
                    "frame_path": frame_fp,
                    "frame_annot_path": annot_fp,
                    "boxes": boxes,
                    "boxes_norm": norm_boxes,
                    "fp": False,
                    "grade": "중",   # 야장 캡션용 등급. 검수하면서 소/중/대로 고친다.
                    "direction": item.get("direction", ""),
                    # 필터가 같은 프레임을 어떻게 봤는가. 없으면 판정 안 함(None).
                    "filter_prob": probs.get(str(frame_fp)),
                })

            if probs and FILTER_MODE == "parallel":
                _apply_parallel_filter(v_data, frames, probs, path)

            mark_dist_conflicts(v_data["rows"])
            _update_travel_distance(path.name)
            ws_manager.result_update(path.name)
            # 영상 하나 끝날 때마다 저장 — 중간에 죽어도 여기까지는 남는다
            session_store.save()
            ws_manager.log(f" - Done. Found {len(v_data['rows'])} issues.")

        session_store.save()
        state.analyzing = False
        total_rows = sum(len(v["rows"]) for v in state.video_data_map.values())
        ws_manager.log(f">>> Analysis Finished. Total {total_rows} issues found. <<<")
        ws_manager.batch_done(model_stats, errors)

    except Exception as e:
        state.analyzing = False
        ws_manager.log(f"CRITICAL ERROR: {e}", "ERROR")
        ws_manager.batch_done({"yolo": 0}, [str(e)])

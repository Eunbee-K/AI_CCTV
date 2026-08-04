import re
import threading
from typing import Optional

import cv2

from . import session_store, ws_manager
from .annotate import annotate_frame
from .config import FRAME_INTERVAL
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

            ws_manager.progress(path.name, idx + 1, total, "yolo")
            if state.remote_yolo_url:
                ws_manager.log(f" - Running YOLO inference on Colab ({state.remote_yolo_url})...")
                merged_rows, yolo_err = call_yolo_remote(frames, state.remote_yolo_url)
            else:
                ws_manager.log(" - Running YOLO inference (local CPU)...")
                merged_rows, yolo_err = call_yolo(state.yolo_model, frames)
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
                })

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

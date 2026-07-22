import threading
from typing import Optional

import cv2

from . import ws_manager
from .annotate import annotate_frame
from .config import FRAME_INTERVAL
from .frames import extract_frames, seconds_to_mmss
from .ocr import ocr_distance_from_frame, ocr_overlay_metadata, normalize_diameter_text, try_ocr_find_range
from .rows import mark_dist_conflicts
from .state import state
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

    # 원격(Colab) 추론 서버가 설정돼 있으면 로컬 YOLO 모델은 로드하지 않는다.
    if not state.remote_yolo_url:
        model, err = ensure_yolo_loaded()
        if not model:
            return f"YOLO model not loaded: {err}"

    state.analyzing = True
    threading.Thread(target=_run_batch_thread, daemon=True).start()
    return None


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
                    "direction": item.get("direction", ""),
                })

            mark_dist_conflicts(v_data["rows"])
            ws_manager.result_update(path.name)
            ws_manager.log(f" - Done. Found {len(v_data['rows'])} issues.")

        state.analyzing = False
        total_rows = sum(len(v["rows"]) for v in state.video_data_map.values())
        ws_manager.log(f">>> Analysis Finished. Total {total_rows} issues found. <<<")
        ws_manager.batch_done(model_stats, errors)

    except Exception as e:
        state.analyzing = False
        ws_manager.log(f"CRITICAL ERROR: {e}", "ERROR")
        ws_manager.batch_done({"yolo": 0}, [str(e)])

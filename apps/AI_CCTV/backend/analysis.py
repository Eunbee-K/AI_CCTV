import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import (defect_classifier, defect_filter, llm_infer, session_store,
               ws_manager)
from .annotate import annotate_frame
from .config import (CLASSIFIER_ENABLED,
                     FILTER_MAX_MISS_ROWS, FILTER_MODE, FILTER_THRESHOLD,
                     OUTSIDE_DIST_M, OUTSIDE_SCAN_MAX,
                     FILTER_TOP_RATIO,
                     FRAME_INTERVAL, YOLO_DROP_BELOW)
from .frames import extract_frames, seconds_to_mmss
from .ocr import (ocr_distance_from_frame, ocr_overlay_metadata,
                  normalize_diameter_text, video_duration_s)
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


def _run_filter(frames, video_name: str) -> Tuple[Dict[str, float],
                                                  Dict[str, Tuple[str, float]]]:
    """프레임별 결함 확률과 이름. 쓸 수 없으면 빈 dict 둘.

    **분류기 하나가 필터와 이름을 겸한다.** 24클래스 중 정상 계열이 아닐 확률이
    곧 결함 확률이다. 테스트셋에서 이진 전용 모델과 같은 성적이 나왔고
    (AUC 0.9906 vs 0.9920, 잡음 ±0.008 안), 관 밖 오탐이 39% -> 0%로 사라졌다.

    분류기가 없으면 옛 이진 필터(filter.onnx)로 물러선다 — 모델 파일을 아직
    안 옮긴 PC에서도 분석이 멈추지 않아야 한다.
    """
    if FILTER_MODE == "off" or not frames:
        return {}, {}

    # 이 모델은 노후관로(우수관) 자료가 중심이다. 신설 전용 모델(NEW_v1)이
    # AUC 0.75로 미달이라 보류된 사이 신설에 아무 보조 장치가 없던 적이 있어,
    # 지금은 관로구분과 무관하게 돌린다 — 얼마나 맞는지는 눈으로 확인한다.
    if CLASSIFIER_ENABLED:
        ok, why = defect_classifier.availability()
        if ok:
            t0 = time.time()
            probs, names = defect_classifier.analyze(frames)
            ws_manager.log(
                f" - 분류기({FILTER_MODE}): {len(probs)}/{len(frames)}프레임 채점 · "
                f"이름 {len(names)}개 (상위 {FILTER_TOP_RATIO:.0%} 통과, "
                f"{time.time() - t0:.1f}s)"
            )
            return probs, names
        ws_manager.log(f" - 분류기를 못 씀({why}) — 옛 이진 필터로 돌아감", "WARN")

    ok, why = defect_filter.availability()
    if not ok:
        # 필터는 보조 장치다. 없다고 분석을 멈추지 않고 알리기만 한다.
        ws_manager.log(f" - Filter unavailable, skipping: {why}", "WARN")
        return {}, {}

    t0 = time.time()
    probs = defect_filter.defect_probs(frames)
    # 판정 기준이 모드마다 다르다 — lead/series는 순위, parallel은 임계값.
    # 로그에 실제로 쓰는 기준을 적어야 나중에 숫자를 해석할 수 있다.
    if FILTER_MODE in ("lead", "series"):
        detail = f"top {FILTER_TOP_RATIO:.0%} will pass"
    else:
        hit = sum(1 for p in probs.values() if p >= FILTER_THRESHOLD)
        detail = f"{hit} over threshold {FILTER_THRESHOLD}"
    ws_manager.log(
        f" - Filter({FILTER_MODE}): scored {len(probs)}/{len(frames)} frames "
        f"({detail}, {time.time() - t0:.1f}s)"
    )
    return probs, {}


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


def _classifier_active() -> bool:
    """지금 점수를 내는 것이 분류기인가(옛 이진 필터가 아니라)."""
    return CLASSIFIER_ENABLED and defect_classifier.availability()[0]


def _outside_frames(frames) -> set:
    """관 밖(맨홀·지상 전경) 구간의 프레임. 자막 거리가 0에 가까운 앞부분이다.

    **필터는 관 밖을 결함이라고 부른다.** 학습 데이터에 관 밖 정상이 한 장도 없어서다
    (`build_old_v2.py`가 AIHub OUT_*을 일부러 뺐다 — v1이 "관 밖 = 정상"만 배워
    망가진 것을 피하려던 것인데, 아예 안 보여주니 판단할 근거가 없어졌다).

    테스트셋 실측: 관 밖 40장을 **100% 결함으로 판정**한다(중앙값 0.976).
    관 안쪽은 정확하다 — IN 0.0158 / PJ 0.0187.

    조사 영상은 늘 관 밖에서 시작하므로, 이 구간이 상위 순위를 차지해 진짜 결함을
    밀어낸다. 모델을 고치기 전까지 거리로 잘라낸다.

    카메라는 한 번 들어가면 다시 안 나오므로 앞에서부터 훑다가 거리가 잡히면 멈춘다
    — 전 프레임을 OCR하면 프레임당 1~2초씩 든다.
    """
    outside, checked = set(), 0
    for f in sorted(frames, key=_frame_sec):
        if checked >= OUTSIDE_SCAN_MAX:
            break
        checked += 1
        m = re.search(r"(\d+(?:\.\d+)?)", ocr_distance_from_frame(Path(f)) or "")
        if m and float(m.group(1)) >= OUTSIDE_DIST_M:
            break                      # 관 안으로 들어왔다
        outside.add(str(f))
    if outside:
        secs = sorted(_frame_sec(f) for f in outside)
        ws_manager.log(
            f" - 관 밖으로 판단해 제외: {len(outside)}프레임 "
            f"({seconds_to_mmss(secs[0])}~{seconds_to_mmss(secs[-1])})"
        )
    return outside


def _filter_runs(frames, probs: Dict[str, float]) -> List[List[Path]]:
    """필터가 "결함"이라 본 구간들. 각 구간은 이어진 프레임 묶음이다.

    **절대 임계값이 아니라 영상 안에서의 순위로 자른다.** 학습에 쓴 야장은 결함
    비율이 45%인데 실제 영상은 1.5%라, 같은 임계값이 전혀 다르게 동작한다
    (0.032로 자르면 실영상 프레임의 99.4%가 통과한다). 순위는 분포가 달라도
    유지되므로 영상 조건에 자동으로 맞춰진다.

    하나의 결함은 보통 여러 프레임에 걸쳐 보인다. 프레임마다 행을 만들면 같은
    결함이 열 줄로 늘어나므로, 이어진 것끼리 한 구간으로 묶어 한 행만 만든다.
    """
    # 관 밖 잘라내기는 **옛 이진 필터(v3)를 쓸 때만** 필요하다. v3는 관 밖을
    # 100% 결함이라 부르지만(중앙값 0.976) 지금 분류기는 관 밖 오탐이 0%다
    # (테스트셋 OUT_MH/OUT_CAR 0%). 프레임당 OCR이 1~2초라 필요 없으면 안 하는
    # 편이 훨씬 빠르다.
    if not _classifier_active():
        outside = _outside_frames(frames)
        frames = [f for f in frames if str(f) not in outside] or list(frames)

    ranked = sorted(frames, key=lambda f: -probs.get(str(f), 0.0))
    keep_n = max(1, round(len(frames) * FILTER_TOP_RATIO))
    kept = sorted(ranked[:keep_n], key=lambda f: _frame_sec(f))

    runs: List[List[Path]] = []
    for f in kept:
        if runs and _frame_sec(f) - _frame_sec(runs[-1][-1]) <= FRAME_INTERVAL:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def _frame_sec(f) -> int:
    """extract_frames가 {초:06d}.jpg로 저장한다. 아니면 정렬만 되게 -1."""
    stem = Path(f).stem
    return int(stem) if stem.isdigit() else -1


def _run_llm(frames) -> Dict[int, List[str]]:
    """LLM 판독. 꺼져 있거나 키가 없으면 빈 dict — 분석은 그대로 진행된다."""
    ok, why = llm_infer.availability()
    if not ok:
        ws_manager.log(f" - LLM: {why}")
        return {}

    t0 = time.time()
    ws_manager.log(f" - LLM 판독 시작 ({why}) — 프레임 {len(frames)}장")
    by_time, errors = llm_infer.analyze(frames)
    for e in errors[:3]:
        ws_manager.log(f"   ! {e}", "ERROR")
    if len(errors) > 3:
        ws_manager.log(f"   ! 외 {len(errors) - 3}건", "ERROR")
    ws_manager.log(
        f" - LLM: {len(by_time)}개 프레임에서 결함 판독 ({time.time() - t0:.0f}s)"
    )
    return by_time


def _build_lead_rows(v_data: dict, frames, probs: Dict[str, float],
                     merged_rows: List[dict], path,
                     llm_by_time: Optional[Dict[int, List[str]]] = None,
                     cls_by_time: Optional[Dict[int, Tuple[str, float]]] = None) -> None:
    """**필터가 잡은 구간이 곧 결함 리스트.** YOLO는 이름만 붙인다.

    행을 만드는 주체가 뒤바뀐 것이 핵심이다. 지금까지는 YOLO가 박스를 친 프레임만
    행이 됐고 필터는 옆에서 훈수만 뒀는데, 여기서는 필터가 고른 구간이 먼저 행이
    되고 YOLO 검출은 그 행에 이름·박스로 얹힌다. YOLO가 아무것도 못 찾은 구간도
    행으로 남아 검수자가 직접 보고 판단한다.

    구간 대표 프레임은 **YOLO가 가장 자신 있게 검출한 프레임**을 쓴다. 박스가 보여야
    검수자가 판단할 수 있기 때문이다. 검출이 없으면 필터 확률이 가장 높은 프레임.

    LLM 판독(llm_by_time)이 켜져 있으면 이름을 붙이는 자리에 나란히 선다. YOLO가
    못 붙인 구간을 채우거나, 다르게 본 이름을 후보로 얹는다. **박스는 YOLO만
    만든다** — LLM은 위치를 모르기 때문이다.
    """
    runs = _filter_runs(frames, probs)
    if FILTER_MAX_MISS_ROWS > 0:
        runs = runs[:FILTER_MAX_MISS_ROWS]

    # YOLO 검출을 시각으로 찾을 수 있게 해둔다
    yolo_at = {int(item.get("time_s", 0)): item for item in merged_rows}
    llm_by_time = llm_by_time or {}
    cls_by_time = cls_by_time or {}

    # **분류기가 확실히 정상이라 본 프레임의 YOLO 검출은 버린다.**
    # YOLO는 관 밖(맨홀·지상 전경)을 결함이라 부른다 — 테스트셋 실측
    # OUT_MH 89% / OUT_CAR 82% / OUT_INVERT 82%. 분류기는 같은 사진을
    # 0%로 정확히 정상이라 답한다(점수 0.024~0.043).
    # 옛 v3 필터 때는 OCR로 거리를 읽어 앞부분을 잘라냈는데(_outside_frames),
    # 분류기 점수로 같은 일을 프레임당 0초에 할 수 있다.
    if _classifier_active() and yolo_at:
        sec_prob = {_frame_sec(f): p for f, p in probs.items()}
        dropped = [t for t in yolo_at
                   if sec_prob.get(t, 1.0) < YOLO_DROP_BELOW]
        for t in dropped:
            del yolo_at[t]
        if dropped:
            ws_manager.log(
                f" - 분류기가 정상이라 본 곳의 YOLO 검출 {len(dropped)}개 무시 "
                f"(관 밖 오탐 방지): "
                + ", ".join(seconds_to_mmss(t) for t in sorted(dropped)[:8])
                + (f" … 외 {len(dropped) - 8}곳" if len(dropped) > 8 else "")
            )
    used: set = set()
    llm_used: set = set()

    def best_conf(item):
        # 박스의 신뢰도 키는 "confidence"다("conf"가 아니다 — yolo_infer.py 참고).
        return max((b.get("confidence") or 0 for b in item.get("boxes", [])), default=0)

    for run in runs:
        secs = [_frame_sec(f) for f in run]
        hits = [(t, yolo_at[t]) for t in secs if t in yolo_at]

        if not hits and secs:
            # **구간 바로 옆의 검출도 끌어온다.** 필터와 YOLO가 같은 결함을 한 프레임
            # 어긋나게 보는 일이 흔하다. JB1-2-0413의 2:31 결함이 그랬다 — 필터는
            # 02:33 한 장만 골랐는데 YOLO는 02:32와 02:34를 잡아, 딱 사이로 빠져
            # "이름 미부여"가 됐다. 한 프레임 폭만 넓혀 같은 결함으로 본다.
            lo, hi = secs[0] - FRAME_INTERVAL, secs[-1] + FRAME_INTERVAL
            hits = [(t, yolo_at[t]) for t in yolo_at if lo <= t <= hi and t not in used]

        if hits:
            # 신뢰도가 가장 높은 검출을 가진 프레임을 대표로 — 박스가 보여야 한다
            t_sec, item = max(hits, key=lambda kv: best_conf(kv[1]))
            used.update(t for t, _ in hits)
        else:
            t_sec = max(secs, key=lambda t: probs.get(
                str(state.frames_root / path.stem / f"{t:06d}.jpg"), 0.0))
            item = {}

        frame_fp = state.frames_root / path.stem / f"{t_sec:06d}.jpg"
        boxes = item.get("boxes", [])
        annot_fp, norm_boxes = (None, [])
        if boxes and frame_fp.exists():
            annot_fp, norm_boxes = annotate_frame(frame_fp, boxes)

        dist_ocr = ocr_distance_from_frame(frame_fp) if frame_fp.exists() else ""
        span = ""
        if len(run) > 1:
            span = f"{seconds_to_mmss(secs[0])}~{seconds_to_mmss(secs[-1])}"

        # LLM이 이 구간에서 본 이름. 구간 전체를 훑어 합친다 — LLM은 프레임을
        # 몇 장씩 묶어 보므로 대표 프레임 한 장만 보면 놓친다.
        llm_names: List[str] = []
        for t in secs:
            for c in llm_by_time.get(t, []):
                if c not in llm_names:
                    llm_names.append(c)
            if t in llm_by_time:
                llm_used.add(t)

        # 분류기가 이 구간에서 본 이름. 구간 전체를 훑어 가장 확신한 것을 고른다.
        cls_names: List[str] = []
        if cls_by_time:
            cand = [cls_by_time[t] for t in secs if t in cls_by_time]
            if cand:
                cls_names = [max(cand, key=lambda kv: kv[1])[0]]

        yolo_names = list(item.get("defects", []))
        # YOLO가 이름을 못 붙였으면 다른 판독자 것을 그대로 쓴다. 붙였으면 YOLO를
        # 앞에 두고 나머지가 추가로 본 것만 뒤에 붙인다 — 박스가 있는 쪽이 근거가
        # 분명하다. 분류기를 LLM보다 앞에 두는 것은 회사망에서도 늘 돌기 때문이다.
        extra = [c for c in cls_names + llm_names if c not in yolo_names]
        seen: set = set()
        extra = [c for c in extra if not (c in seen or seen.add(c))]
        defects = yolo_names + extra
        cls_only = [c for c in cls_names if c not in yolo_names]
        llm_only = [c for c in llm_names if c not in yolo_names and c not in cls_names]

        note = ""
        if not defects:
            # 아무도 이름을 못 붙인 구간. 결함항목은 검수자가 고른다.
            note = "확인필요(이름 미부여)"
        elif not yolo_names:
            note = "분류기 판독" if cls_only else "LLM 판독"
        else:
            add = ([f"분류기: {', '.join(cls_only)}"] if cls_only else []) + \
                  ([f"LLM: {', '.join(llm_only)}"] if llm_only else [])
            if add:
                note = "추가 — " + " / ".join(add)
        if span:
            note = (note + " " if note else "") + f"구간 {span}"

        v_data["rows"].append({
            "time": t_sec,
            "dist": dist_ocr or item.get("distance_text", ""),
            "defects": defects,
            "note": note,
            "frame_path": frame_fp,
            "frame_annot_path": annot_fp,
            "boxes": boxes,
            "boxes_norm": norm_boxes,
            "fp": False,
            "grade": "중",
            "direction": item.get("direction", ""),
            "filter_prob": probs.get(str(frame_fp)),
            # 이름이 없는 행은 거리 그룹에 접히면 찾을 수 없다. 수동 행과 같은 취급.
            "manual": not defects,
            "filter_only": not defects,
        })

    rows = v_data["rows"]
    by_yolo = sum(1 for r in rows if r["boxes"])
    # 박스가 없는데 이름이 있으면 분류기나 LLM이 붙인 것이다
    by_other = sum(1 for r in rows if r["defects"] and not r["boxes"])
    ws_manager.log(
        f" - Filter(lead): {len(runs)} defect runs → {len(rows)} rows "
        f"({by_yolo} named by YOLO, {by_other} by 분류기/LLM only, "
        f"{len(rows) - by_yolo - by_other} need review)"
    )

    # 필터가 안 고른 곳에서 YOLO/LLM이 찾은 것. 표에는 넣지 않지만 조용히 사라지면
    # 안 되므로 남긴다 — 이 숫자가 크면 FILTER_TOP_RATIO를 올려야 한다는 뜻이다.
    for tag, orphan in (("YOLO", sorted(t for t in yolo_at if t not in used)),
                        ("LLM", sorted(t for t in llm_by_time if t not in llm_used))):
        if orphan:
            ws_manager.log(
                f"   ! {tag} detected at {len(orphan)} frames the filter did not pick: "
                + ", ".join(seconds_to_mmss(t) for t in orphan[:12])
                + (f" … 외 {len(orphan) - 12}곳" if len(orphan) > 12 else ""),
                "WARN",
            )


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

            # 영상 전체(0초 ~ 끝)를 본다. 관 밖 구간은 여기서 잘라내지 않고
            # 거리 판독(OCR)과 분류기의 관 외부 인식이 뒤에서 걸러낸다.
            s, e = 0, video_duration_s(path)
            ws_manager.log(f" - Range: {seconds_to_mmss(s)} ~ {seconds_to_mmss(e)} (전체)")

            ws_manager.progress(path.name, idx + 1, total, "extract_frames")
            frames = extract_frames(path, state.frames_root, s, e, FRAME_INTERVAL)
            ws_manager.log(f" - Frames extracted: {len(frames)}")

            # Stage-1 필터. parallel이면 전 프레임을 재고 YOLO 결과와 대조하고,
            # series면 통과한 프레임만 YOLO에 넘긴다.
            probs, cls_names = _run_filter(frames, path.name)
            yolo_frames = frames
            if probs and FILTER_MODE == "series":
                # **절대 임계값이 아니라 영상 안에서의 순위로 자른다.** 학습·검증에 쓴
                # 야장 데이터는 결함 비율이 45%인데 실제 영상은 1.5%라, 같은 임계값이
                # 전혀 다르게 동작한다(0.032로 자르니 프레임의 0.6%밖에 안 걸러졌다).
                # 순위는 분포가 달라도 유지되므로 영상 조건에 자동으로 맞춰진다.
                #
                # 판정을 못 받은 프레임(읽기 실패 등)은 확률을 1.0으로 둬 통과시킨다.
                # 필터가 조용히 버리는 것보다 YOLO가 한 번 더 보는 편이 안전하다.
                ranked = sorted(frames, key=lambda f: -probs.get(str(f), 1.0))
                keep_n = max(1, round(len(frames) * FILTER_TOP_RATIO))
                yolo_frames = sorted(ranked[:keep_n], key=lambda f: str(f))
                dropped = ranked[keep_n:]
                ws_manager.log(
                    f" - Filter(series): {len(yolo_frames)}/{len(frames)} frames pass "
                    f"(top {FILTER_TOP_RATIO:.0%}) → {len(dropped)} skipped"
                )
                # 직렬은 되돌릴 수 없다. 버린 구간을 로그에 남겨 눈으로 되짚을 수 있게 한다.
                if dropped:
                    secs = sorted(int(Path(f).stem) for f in dropped if Path(f).stem.isdigit())
                    ws_manager.log(
                        f"   skipped at: {', '.join(seconds_to_mmss(x) for x in secs[:12])}"
                        + (f" … 외 {len(secs) - 12}곳" if len(secs) > 12 else "")
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
            # **관로번호는 자막에서 읽는다.** 예전에는 파일명(path.stem)을 그대로
            # 썼는데, 파일명은 현장에서 임의로 붙인 일련번호인 경우가 많아
            # 실제 관로번호와 다르다(`5982.mp4` → 자막은 `SM1-121-002`).
            # 자막을 못 읽었을 때만 파일명으로 물러선다 — 관로번호 칸이 비면
            # 조사표가 어느 관로 것인지 알 수 없게 되기 때문이다.
            pid = meta.get("pipe_id", "") or path.stem
            dia = normalize_diameter_text(meta.get("diameter_text", ""))
            site = meta.get("site_name", "")

            v_data["pipe_id"] = pid
            src = "자막" if meta.get("pipe_id") else "파일명(자막 판독 실패)"
            v_data["dia"] = dia
            ws_manager.log(
                f" - OCR meta: site='{site}', pipe='{pid}'({src}), dia='{dia}'")

            if site and not state.site_name:
                state.site_name = site

            _fill_report_meta_from_ocr(path.name, meta)

            # lead: 필터가 고른 구간이 행이 되고, YOLO 검출은 거기에 이름으로 얹힌다.
            # 그 외 모드: YOLO가 박스를 친 프레임이 행이 된다(기존 방식).
            if probs and FILTER_MODE == "lead":
                # 이름은 필터와 같은 호출에서 이미 나왔다 — 다시 돌리지 않는다
                cls_by_time = {_frame_sec(f): v for f, v in cls_names.items()}
                ws_manager.progress(path.name, idx + 1, total, "llm")
                llm_by_time = _run_llm(frames)
                _build_lead_rows(v_data, frames, probs, merged_rows, path,
                                 llm_by_time, cls_by_time)
                merged_rows = []

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

"""LLM(Gemini + GPT) 판독 — 세 번째 의견.

ver_2.2 앱(`E:/CCTV/ver_2.2_AICCTV`)에서 쓰던 구조를 옮겨온 것이다. 그쪽은 LLM이
분석의 전부였지만 여기서는 **필터가 고른 구간에 이름을 붙이는 두 번째 판독자**로
쓴다. 프레임을 골라내지도, 버리지도 않는다.

  필터  결함이 있는 구간을 고른다  (행을 만든다)
  YOLO  그 구간에 이름을 붙인다
  LLM   같은 구간에 이름을 붙인다  (YOLO가 못 붙였거나 다르게 본 것을 채운다)

**회사망에서는 못 쓴다.** 외부 API를 호출하므로 망분리 환경에서는 켜지지 않는다.
그래서 기본값이 off이고, 앱에서 껐다 켰다 할 수 있게 되어 있다(state.llm_enabled).
꺼져 있으면 이 모듈은 아무 일도 하지 않고 분석은 그대로 진행된다.

키는 환경변수(OPENAI_API_KEY / GOOGLE_API_KEY)나 앱 화면에서 넣는다. 화면에서 넣은
키는 메모리에만 두고 파일로 저장하지 않는다 — 결과 세션 파일에 섞여 들어가면
그대로 공유되기 때문이다.
"""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .config import (LLM_CHUNK_SIZE, LLM_EXAMPLES_DIR, LLM_EXAMPLES_PER_LABEL,
                     LLM_GEMINI_MODEL, LLM_GPT_MODEL, LLM_MAX_WORKERS,
                     LLM_TIMEOUT_S, EXTRACT_MAX_SIDE)

# ver_2.2가 쓰던 8종. 우리 코드표(31종)보다 거칠지만 LLM이 실제로 구분해내는 단위다.
# 옮기면서 코드표에 맞춰 매핑까지 해둔다 — 표에는 코드로 들어가야 하기 때문이다.
LLM_LABEL_TO_CODE = {
    "파손": "BK",
    "이탈": "JS",          # 이음부-이탈
    "손상": "JF",          # 이음부-손상
    "침하": "SG",
    "영구장애물": "PO",
    "균열": "CL",          # LLM은 원주/길이/복합을 구분하지 못한다. 가장 흔한 길이로.
    "가지관 돌출": "LP",   # 연결관-돌출
    "기타": "ETC",
}
_EN_TO_KO = {
    "Breakage": "파손", "Separation": "이탈", "Damage": "손상",
    "Subsidence": "침하", "Obstacle": "영구장애물", "Crack": "균열",
    "Intruding Pipe": "가지관 돌출", "Other": "기타",
}

SYSTEM_PROMPT_EN = (
    "You are an expert in sewage pipe CCTV analysis.\n"
    "Analyze the provided frames based on the text definitions below.\n"
    "Priority: Text Definitions > General Knowledge.\n\n"
    "--- Defect Definitions ---\n"
    "1. **Breakage (파손)**: Clearly broken pipe or holes.\n"
    "2. **Separation (이탈)**: Widened or misaligned joints.\n"
    "3. **Damage (손상)**: Rough surface, gouges, but not broken.\n"
    "4. **Subsidence (침하)**: Water ponding >10% of diameter.\n"
    "5. **Obstacle (영구장애물)**: Fixed objects (roots, soil, rocks).\n"
    "6. **Crack (균열)**: Fine surface cracks.\n"
    "7. **Intruding Pipe (가지관 돌출)**: Connecting pipe sticking out.\n"
    "8. **Other (기타)**: Ambiguous defects.\n\n"
    "--- Instructions ---\n"
    "0. Reference images may be provided, each tagged `Ex: <label>`. Images tagged\n"
    "   `Ex: 정상` are NORMAL pipe — do NOT report defects for frames that look\n"
    "   like these. Match the reference images over your general intuition.\n"
    "1. Analyze frames sequentially.\n"
    "2. Output **ONLY ONE JSON LIST**.\n"
    "3. Sensitivity: MEDIUM. Report clear defects.\n"
    "4. Labels must be in KOREAN: "
    + json.dumps(list(LLM_LABEL_TO_CODE), ensure_ascii=False) + ".\n"
    "5. If no defect, return \"defects\": [].\n\n"
    "--- JSON Structure ---\n"
    "[{\"time_s\": <int>, \"defects\": [\"<STR>\"], \"distance_text\": \"<str>\"}, ...]"
)


# ───────── 공통 ─────────

def _api_keys() -> Tuple[str, str]:
    """(openai, google). 화면에서 넣은 키가 환경변수보다 우선한다."""
    from .state import state
    import os
    openai_key = (state.llm_keys.get("openai") or os.getenv("OPENAI_API_KEY")
                  or os.getenv("GPT_API_KEY") or "").strip()
    google_key = (state.llm_keys.get("google") or os.getenv("GOOGLE_API_KEY")
                  or os.getenv("GEMINI_API_KEY") or "").strip()
    return openai_key, google_key


def availability() -> Tuple[bool, str]:
    """지금 LLM 판독을 쓸 수 있는가. (가능 여부, 이유)"""
    from .state import state
    if not state.llm_enabled:
        return False, "LLM 판독 꺼짐"

    openai_key, google_key = _api_keys()
    if not openai_key and not google_key:
        return False, "API 키가 없습니다 (OPENAI_API_KEY / GOOGLE_API_KEY)"

    have = []
    if google_key and _genai() is not None:
        have.append("Gemini")
    if openai_key and _openai() is not None:
        have.append("GPT")
    if not have:
        return False, "google-generativeai / openai 패키지가 설치돼 있지 않습니다"

    ex = load_examples()
    detail = " + ".join(have)
    # 예시가 실제로 붙었는지 화면에서 확인할 수 있어야 한다. 폴더 경로만 맞춰두고
    # 안 들어간 채로 도는 것이 제일 알아채기 어렵다.
    detail += f" · 예시 {len(ex)}장" if ex else " · 예시 없음"
    return True, detail


def _genai():
    try:
        import google.generativeai as genai
        return genai
    except Exception:
        return None


def _openai():
    try:
        from openai import OpenAI
        return OpenAI
    except Exception:
        return None


def _repair_json(text: str):
    """LLM이 끝을 잘라먹거나 쉼표를 흘리는 경우가 잦아 한 번 다듬어 본다."""
    try:
        return json.loads(text)
    except Exception:
        pass
    s = re.sub(r",\s*([}\]])", r"\1", text.strip())
    # 닫히지 않은 괄호를 채운다
    s += "}" * max(0, s.count("{") - s.count("}"))
    s += "]" * max(0, s.count("[") - s.count("]"))
    try:
        return json.loads(s)
    except Exception:
        return []


_EXAMPLES: Optional[List[Tuple[str, bytes]]] = None


def load_examples() -> List[Tuple[str, bytes]]:
    """프롬프트에 같이 보낼 예시 사진 [(라벨, jpeg바이트), ...].

    ver_2.2는 GPT에만, 라벨당 1장만, 파일명 부분일치로 골라 보냈다. 여기서는
    `labels.jsonl`(있으면)을 정답으로 읽고 **두 모델 모두에게** 보낸다.

    **정상 사진도 함께 보낸다.** LLM은 결함을 과하게 잡는 편이라(실측에서 88프레임
    중 69장을 결함이라 봤다) "이런 건 정상"을 보여주는 쪽이 실제로 필요하다.

    한 번 읽어 캐시한다 — 매 청크마다 다시 인코딩하면 그만큼 느려진다.
    """
    global _EXAMPLES
    if _EXAMPLES is not None:
        return _EXAMPLES

    _EXAMPLES = []
    d = LLM_EXAMPLES_DIR
    if not d.exists():
        return _EXAMPLES

    by_label: Dict[str, List[Path]] = {}
    manifest = d / "labels.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fp = d / str(obj.get("image", ""))
            if not fp.exists():
                continue
            for lab in obj.get("labels", []):
                by_label.setdefault(str(lab).strip(), []).append(fp)
    else:
        # 파일명 규칙: {날짜}_{라벨}[_라벨2](번호).jpg
        for fp in sorted(d.iterdir()):
            if fp.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = re.match(r"\d+_(.+?)\(\d+\)", fp.stem)
            if not m:
                continue
            for lab in m.group(1).split("_"):
                by_label.setdefault(lab.strip(), []).append(fp)

    for lab, files in by_label.items():
        # 우리가 이름을 아는 결함 + 정상만 보낸다. 모르는 라벨은 혼란만 준다.
        if lab != "정상" and lab not in LLM_LABEL_TO_CODE:
            continue
        for fp in files[:LLM_EXAMPLES_PER_LABEL]:
            data = _img_b64(fp)
            if data:
                _EXAMPLES.append((lab, data))
    return _EXAMPLES


def _img_b64(fp: Path) -> Optional[bytes]:
    """API에 보낼 JPEG 바이트. 원본 그대로 보내면 느리고 비싸다."""
    try:
        with Image.open(fp) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > EXTRACT_MAX_SIDE:
                r = EXTRACT_MAX_SIDE / max(w, h)
                im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
    except Exception:
        return None


def _normalize(items, model_tag: str) -> List[dict]:
    """LLM 응답을 우리 코드로 바꾼다. 모르는 라벨은 버린다."""
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            t = int(item.get("time_s"))
        except (TypeError, ValueError):
            continue
        raw = item.get("defects") or []
        if not isinstance(raw, list):
            raw = [str(raw)]
        codes = []
        for d in raw:
            ko = _EN_TO_KO.get(str(d).strip(), str(d).strip())
            code = LLM_LABEL_TO_CODE.get(ko)
            if code and code not in codes:
                codes.append(code)
        if codes:
            out.append({"time_s": t, "defects": codes, "model": model_tag,
                        "distance_text": str(item.get("distance_text") or "")})
    return out


# ───────── 모델별 호출 ─────────

def _call_gemini(frames: List[Path]) -> Tuple[List[dict], Optional[str]]:
    genai = _genai()
    _, key = _api_keys()
    if genai is None or not key:
        return [], None          # 안 쓰는 것은 오류가 아니다
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            LLM_GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT_EN,
            generation_config={"temperature": 0.0,
                               "response_mime_type": "application/json",
                               "max_output_tokens": 8192},
        )
        parts = []
        examples = load_examples()
        if examples:
            parts.append("--- Visual Examples (Reference) ---")
            for lab, data in examples:
                parts.append({"mime_type": "image/jpeg", "data": data})
                parts.append(f"Ex: {lab}")
        parts.append("--- Analyze Frames based on Text Definitions ---")
        for fp in frames:
            d = _img_b64(fp)
            if not d:
                continue
            parts.append(f"time_s: {int(fp.stem)}")
            parts.append({"mime_type": "image/jpeg", "data": d})

        resp = model.generate_content(parts, request_options={"timeout": LLM_TIMEOUT_S})
        txt = getattr(resp, "text", "") or ""
        if not txt and getattr(resp, "candidates", None):
            cand = resp.candidates[0]
            txt = "\n".join(p.text or "" for p in getattr(cand.content, "parts", [])
                            if getattr(p, "text", None))
        if not txt:
            return [], "Gemini: 응답 없음"
        s, e = txt.find("["), txt.rfind("]")
        if s == -1:
            return [], "Gemini: JSON 없음"
        return _normalize(_repair_json(txt[s:e + 1]), "gemini"), None
    except Exception as ex:
        return [], f"Gemini: {ex}"


def _call_gpt(frames: List[Path]) -> Tuple[List[dict], Optional[str]]:
    OpenAI = _openai()
    key, _ = _api_keys()
    if OpenAI is None or not key:
        return [], None
    try:
        client = OpenAI(api_key=key)
        content = []
        examples = load_examples()
        if examples:
            content.append({"type": "text", "text": "--- Visual Examples (Reference) ---"})
            for lab, data in examples:
                b64 = base64.b64encode(data).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                              "detail": "low"}})
                content.append({"type": "text", "text": f"Ex: {lab}"})
        content.append({"type": "text", "text": "--- Analyze Frames ---"})
        for fp in frames:
            d = _img_b64(fp)
            if not d:
                continue
            b64 = base64.b64encode(d).decode()
            content.append({"type": "text", "text": f"time_s: {int(fp.stem)}"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                          "detail": "low"}})

        resp = client.chat.completions.create(
            model=LLM_GPT_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_EN},
                      {"role": "user", "content": content}],
            max_completion_tokens=4096,
            response_format={"type": "json_object"},
            timeout=LLM_TIMEOUT_S,
        )
        data = _repair_json(resp.choices[0].message.content or "")
        # response_format=json_object라 리스트가 dict로 한 번 감싸여 온다
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        return _normalize(data, "gpt"), None
    except Exception as ex:
        return [], f"GPT: {ex}"


# ───────── 바깥에서 쓰는 것 ─────────

def analyze(frames: List[Path]) -> Tuple[Dict[int, List[str]], List[str]]:
    """프레임들을 판독해 {초: [결함코드...]}와 오류 목록을 돌려준다.

    두 모델을 같은 프레임에 동시에 돌리고 결과를 합친다(합집합). ver_2.2는
    GPT를 우선하고 Gemini를 보조로 썼는데, 여기서는 이름을 채우는 게 목적이라
    둘 중 하나만 봤어도 후보로 올린다. 최종 판단은 검수자가 한다.
    """
    ok, why = availability()
    if not ok:
        return {}, [why]

    chunks = [frames[i:i + LLM_CHUNK_SIZE] for i in range(0, len(frames), LLM_CHUNK_SIZE)]
    by_time: Dict[int, List[str]] = {}
    errors: List[str] = []

    # **청크를 한 번에 다 던진다.** 청크마다 결과를 기다리면 왕복 시간이 그대로
    # 쌓인다 — 88프레임(13청크)에 456초가 걸렸다. 호출은 대부분 대기 시간이므로
    # 동시에 띄우는 편이 훨씬 빠르다. 너무 많이 띄우면 429가 나므로 워커로 조인다.
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as exc:
        futures = []
        for chunk in chunks:
            futures.append(exc.submit(_call_gemini, chunk))
            futures.append(exc.submit(_call_gpt, chunk))
        for fut in futures:
            try:
                items, err = fut.result()
            except Exception as ex:
                items, err = [], str(ex)
            if err:
                errors.append(err)
            for item in items:
                got = by_time.setdefault(item["time_s"], [])
                for c in item["defects"]:
                    if c not in got:
                        got.append(c)

    return by_time, errors

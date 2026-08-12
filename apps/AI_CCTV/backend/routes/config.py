from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from fastapi import HTTPException

from .. import llm_infer, session_store
from ..config import DEFECT_CODE_KO
from ..state import (PIPE_META_FIELDS, PROJECT_META_FIELDS, REPORT_META_FIELDS,
                     REPORT_META_SPEC, state)
from ..yolo_remote import normalize_remote_url, probe_remote

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/defect_codes")
def get_defect_codes():
    """결함 코드 목록. 결과표의 [결함항목] 콤보박스가 이걸로 채워진다.

    표기는 "CC(균열-원주)" 형태로 보여주고 저장은 코드만 한다.
    """
    return {"codes": [{"code": c, "ko": ko} for c, ko in DEFECT_CODE_KO.items()]}


class RemoteUrlBody(BaseModel):
    url: str


class ReportMetaBody(BaseModel):
    values: dict
    video: Optional[str] = None


def _resolve_video(name: Optional[str]) -> Optional[str]:
    """대상 관로(영상). 지정이 없으면 첫 번째 영상을 쓴다."""
    if name and name in state.video_data_map:
        return name
    return next(iter(state.video_data_map), None)


@router.get("/report_meta")
def get_report_meta(video: Optional[str] = None):
    """야장 상단 표의 입력값 + 각 항목을 어떻게 그릴지(spec).

    맨홀번호·관종·거리 등은 관로마다 달라서 영상별로 따로 관리한다(scope=pipe).
    발주처·조사자처럼 현장 전체에 공통인 것은 한 벌만 둔다(scope=project).
    사업명·관로번호는 다른 화면에서 관리하는 값이라 확인용으로만 보여준다.
    """
    target = _resolve_video(video)
    pipe_id = (state.video_data_map.get(target) or {}).get("pipe_id", "") if target else ""
    pipe_values = state.pipe_meta(target) if target else dict(PIPE_META_FIELDS)

    scope = {k: "project" for k in PROJECT_META_FIELDS}
    scope.update({k: "pipe" for k in PIPE_META_FIELDS})

    return {
        "video": target,
        "videos": list(state.video_data_map.keys()),
        "pipe_id": pipe_id,
        "fields": ["사업명", "관로번호", *PROJECT_META_FIELDS.keys(), *PIPE_META_FIELDS.keys()],
        "values": {**state.project_meta, **pipe_values,
                   "사업명": state.site_name, "관로번호": pipe_id},
        "spec": REPORT_META_SPEC,
        "scope": scope,
    }


@router.post("/report_meta")
def set_report_meta(body: ReportMetaBody):
    # 읽기전용 항목(사업명·관로번호)은 화면에도 뜨므로 같이 올라온다 — 그냥 무시한다.
    readonly = {k for k, s in REPORT_META_SPEC.items() if s.get("type") == "readonly"}
    unknown = [k for k in body.values if k not in REPORT_META_FIELDS and k not in readonly]
    if unknown:
        raise HTTPException(400, f"알 수 없는 항목: {', '.join(unknown)}")

    target = _resolve_video(body.video)
    pipe_values = state.pipe_meta(target) if target else None

    for k, v in body.values.items():
        text = str(v or "").strip()
        if k in PROJECT_META_FIELDS:
            state.project_meta[k] = text
        elif k in PIPE_META_FIELDS and pipe_values is not None:
            pipe_values[k] = text

    session_store.save()
    return {"status": "ok", "video": target,
            "values": state.meta_for(target) if target else state.project_meta}


class LlmBody(BaseModel):
    enabled: bool
    openai_key: Optional[str] = None
    google_key: Optional[str] = None


@router.get("/llm")
def get_llm():
    """LLM 판독 상태. 키는 값을 돌려주지 않고 들어있는지만 알린다."""
    ok, detail = llm_infer.availability()
    return {
        "enabled": state.llm_enabled,
        "ready": ok,
        "detail": detail,
        "has_openai": bool(state.llm_keys.get("openai")),
        "has_google": bool(state.llm_keys.get("google")),
    }


@router.post("/llm")
def set_llm(body: LlmBody):
    """켜고 끄기 + 키 입력.

    키는 메모리에만 둔다(state.llm_keys). 세션 파일에 저장하면 결과를 주고받을 때
    키가 딸려 나가기 때문이다. 앱을 다시 켜면 환경변수에서만 읽는다.
    빈 문자열을 보내면 그 키를 지운다 — None이면 그대로 둔다.
    """
    state.llm_enabled = bool(body.enabled)
    if body.openai_key is not None:
        state.llm_keys["openai"] = body.openai_key.strip()
    if body.google_key is not None:
        state.llm_keys["google"] = body.google_key.strip()

    ok, detail = llm_infer.availability()
    return {"status": "ok", "enabled": state.llm_enabled, "ready": ok, "detail": detail,
            "has_openai": bool(state.llm_keys.get("openai")),
            "has_google": bool(state.llm_keys.get("google"))}


@router.get("/remote_yolo_url")
def get_remote_url():
    return {"url": state.remote_yolo_url}


@router.post("/remote_yolo_url")
def set_remote_url(body: RemoteUrlBody):
    """주소를 저장하면서 실제로 서버가 응답하는지까지 확인한다.

    저장만 하고 '연결됨'으로 표시하면, 주소가 틀렸어도 분석을 돌릴 때가 되어서야
    실패를 알게 된다. 연결 시점에 확인해서 원인을 바로 알려준다.
    """
    url = normalize_remote_url(body.url)
    state.remote_yolo_url = url
    session_store.save()
    if not url:
        return {"status": "ok", "url": "", "reachable": False, "detail": "연결 해제 (로컬 추론 사용)"}

    reachable, detail = probe_remote(url)
    return {"status": "ok", "url": url, "reachable": reachable, "detail": detail}

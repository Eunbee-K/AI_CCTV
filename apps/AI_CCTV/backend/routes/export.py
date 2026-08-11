import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..excel_export import export_excel
from ..pipeasset_pdf import export_pipeasset_pdf
from ..state import state

router = APIRouter(prefix="/api/export", tags=["export"])

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class ExportBody(BaseModel):
    path: str
    video: Optional[str] = None   # 주면 그 관로만, 없으면 전체


def _safe(part: str) -> str:
    """파일명에 쓸 수 없는 문자를 정리한다."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (part or "").strip()).strip(" .")


def _report_filename(prefix: str = "CCTV조사표", ext: str = "xlsx",
                     only_video: Optional[str] = None) -> str:
    """현장명_관로번호_날짜.확장자. 값이 없는 부분은 건너뛴다.

    한 관로만 뽑을 때는 그 관로의 현장명·번호를 쓰고, 전체면 '전체'를 붙여
    파일이 섞이지 않게 한다.
    """
    if only_video:
        site = state.site_name_of(only_video)
        pipe_id = (state.video_data_map.get(only_video) or {}).get("pipe_id", "")
        parts = [_safe(p) for p in (site, pipe_id) if _safe(p)]
    else:
        pipe_id = ""
        if state.video_data_map:
            pipe_id = next(iter(state.video_data_map.values())).get("pipe_id", "")
        parts = [_safe(p) for p in (state.site_name, pipe_id) if _safe(p)]
        if len(state.video_data_map) > 1:
            parts.append(f"전체{len(state.video_data_map)}건")
    parts.append(datetime.now().strftime("%Y%m%d"))
    return "_".join([prefix, *parts]) + f".{ext}"


def _require_video(only_video: Optional[str]) -> None:
    if only_video and only_video not in state.video_data_map:
        raise HTTPException(404, f"그 관로를 찾을 수 없습니다: {only_video}")


@router.post("/excel")
def export_excel_route(body: ExportBody):
    """지정한 경로에 저장 (데스크톱 exe 전용 — 파일 대화상자로 고른 로컬 경로).

    웹에서는 이 경로가 '서버 PC의 경로'라 의미가 없으므로 /excel/download를 쓴다.
    """
    _require_video(body.video)
    err = export_excel(body.path, only_video=body.video)
    if err:
        raise HTTPException(500, err)
    return {"status": "ok", "path": body.path}


@router.get("/excel/download")
def download_excel(video: Optional[str] = None):
    """보고서를 만들어 브라우저로 내려보낸다.

    video를 주면 그 관로만, 없으면 분석한 전체를 한 파일로 담는다.
    저장 위치는 브라우저(사용자 PC)가 정한다 — 서버는 어디에 저장할지 알 수 없고,
    알 필요도 없다. 파일은 임시 폴더에 만들었다가 응답으로 흘려보낸다.
    """
    if not state.video_data_map:
        raise HTTPException(400, "내보낼 결과가 없습니다. 먼저 영상을 분석하세요.")
    _require_video(video)

    out = state.temp_dir / "report.xlsx"
    err = export_excel(str(out), only_video=video)
    if err:
        raise HTTPException(500, f"보고서 생성 실패: {err}")

    return FileResponse(str(out), media_type=XLSX_MIME,
                        filename=_report_filename(only_video=video))


@router.get("/pdf/download")
def download_pipeasset_pdf(video: Optional[str] = None):
    """파이프에셋 야장(하수관거 현황 조사 보고서) 형식 PDF를 내려보낸다."""
    if not state.video_data_map:
        raise HTTPException(400, "내보낼 결과가 없습니다. 먼저 영상을 분석하세요.")
    _require_video(video)

    out = state.temp_dir / "report_pipeasset.pdf"
    err = export_pipeasset_pdf(str(out), only_video=video)
    if err:
        raise HTTPException(500, err)

    return FileResponse(str(out), media_type="application/pdf",
                        filename=_report_filename("CCTV야장", "pdf", only_video=video))

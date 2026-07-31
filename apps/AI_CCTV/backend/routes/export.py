import re
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..excel_export import export_excel
from ..state import state

router = APIRouter(prefix="/api/export", tags=["export"])

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class ExportBody(BaseModel):
    path: str


def _safe(part: str) -> str:
    """파일명에 쓸 수 없는 문자를 정리한다."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (part or "").strip()).strip(" .")


def _report_filename() -> str:
    """현장명_관로번호_날짜.xlsx. 값이 없는 부분은 건너뛴다."""
    pipe_id = ""
    if state.video_data_map:
        pipe_id = next(iter(state.video_data_map.values())).get("pipe_id", "")
    parts = [_safe(p) for p in (state.site_name, pipe_id) if _safe(p)]
    parts.append(datetime.now().strftime("%Y%m%d"))
    return "_".join(["CCTV조사표", *parts]) + ".xlsx"


@router.post("/excel")
def export_excel_route(body: ExportBody):
    """지정한 경로에 저장 (데스크톱 exe 전용 — 파일 대화상자로 고른 로컬 경로).

    웹에서는 이 경로가 '서버 PC의 경로'라 의미가 없으므로 /excel/download를 쓴다.
    """
    err = export_excel(body.path)
    if err:
        raise HTTPException(500, err)
    return {"status": "ok", "path": body.path}


@router.get("/excel/download")
def download_excel():
    """보고서를 만들어 브라우저로 내려보낸다.

    저장 위치는 브라우저(사용자 PC)가 정한다 — 서버는 어디에 저장할지 알 수 없고,
    알 필요도 없다. 파일은 임시 폴더에 만들었다가 응답으로 흘려보낸다.
    """
    if not state.video_data_map:
        raise HTTPException(400, "내보낼 결과가 없습니다. 먼저 영상을 분석하세요.")

    out = state.temp_dir / "report.xlsx"
    err = export_excel(str(out))
    if err:
        raise HTTPException(500, f"보고서 생성 실패: {err}")

    return FileResponse(str(out), media_type=XLSX_MIME, filename=_report_filename())

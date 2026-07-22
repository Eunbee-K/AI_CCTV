from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..excel_export import export_excel

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportBody(BaseModel):
    path: str


@router.post("/excel")
def export_excel_route(body: ExportBody):
    err = export_excel(body.path)
    if err:
        raise HTTPException(500, err)
    return {"status": "ok", "path": body.path}

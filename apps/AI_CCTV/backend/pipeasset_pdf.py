"""파이프에셋 야장(하수관거 현황 조사 보고서) 형식 PDF 생성.

실물 야장(`2.3b CCTV조사 야장`, 가평군 노후하수관로 정밀조사용역)을 기준으로 만들었다.
A4 세로 한 장의 구성:

    ┌─────────────────────────────────────┐
    │        하수관거 현황 조사 보고서       │  제목
    ├─────────────────────────────────────┤
    │  사업명 / 보고서번호 / 관로 / 맨홀 …   │  메타데이터 표 (약 20항목)
    ├──────┬──────────────────────────────┤
    │ 관로  │  사진   사진                  │  좌: 관로 모식도(거리 눈금)
    │ 모식도│  사진   사진                  │  우: 사진 2열 × 3행 = 6장
    │      │  사진   사진                  │     각 사진 아래 캡션
    ├──────┴──────────────────────────────┤
    │ 보고서번호:… / 관로번호:… - (n)  조사자 │  꼬리말
    └─────────────────────────────────────┘

한 관로가 여러 페이지에 걸치고, 관로가 바뀌면 페이지 번호가 (1)부터 다시 시작한다.
엑셀 출력(excel_export.py)은 그대로 두고 이쪽을 선택지로 추가한 것이다.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import fitz

from .config import pipeasset_caption
from .rows import build_results_view
from .state import state

# ───────── 지면 상수 (pt, A4 595×842) ─────────
PAGE_W, PAGE_H = 595.0, 842.0
MARGIN_X = 36.0
TITLE_Y = 46.0
TABLE_TOP = 78.0
ROW_H = 15.0                      # 메타데이터 표 한 줄 높이
PHOTOS_PER_PAGE = 6
PHOTO_COLS = 2

RAIL_X = 78.0                     # 관로 모식도 세로선 x
PHOTO_LEFT = 150.0                # 사진 영역 시작 x
PHOTO_GAP_X = 12.0
PHOTO_GAP_Y = 40.0                # 캡션 자리를 포함한 세로 간격
CAPTION_H = 14.0

LINE = (0.0, 0.0, 0.0)
GRAY_FILL = (0.85, 0.85, 0.85)
BLUE = (0.15, 0.15, 0.75)
ORANGE = (0.85, 0.35, 0.15)
PIPE_FILL = (0.80, 0.88, 0.96)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _find_font() -> Optional[str]:
    """한글을 그릴 수 있는 TTF를 찾는다. 없으면 None(→ 생성 거부)."""
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


class _Pen:
    """폰트 등록과 텍스트/사각형 그리기를 묶은 얇은 헬퍼."""

    def __init__(self, doc: fitz.Document, font_path: str):
        self.doc = doc
        self.font_path = font_path
        self.font = fitz.Font(fontfile=font_path)

    def new_page(self) -> fitz.Page:
        page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_font(fontname="KR", fontfile=self.font_path)
        return page

    def text(self, page, x, y, s, size=7.5, color=LINE, align="left", width=None):
        """y는 글자 baseline이 아니라 '박스 세로 중앙'으로 다룬다(표 셀에 맞추기 쉽게)."""
        s = str(s or "")
        if not s:
            return
        tw = self.font.text_length(s, fontsize=size)
        if align == "center" and width:
            x = x + (width - tw) / 2
        elif align == "right" and width:
            x = x + width - tw - 2
        page.insert_text((x, y + size * 0.36), s, fontname="KR", fontsize=size, color=color)

    def cell(self, page, x, y, w, h, s="", size=7.5, fill=None, align="center", color=LINE):
        rect = fitz.Rect(x, y, x + w, y + h)
        page.draw_rect(rect, color=LINE, fill=fill, width=0.5)
        if s:
            pad = 3 if align == "left" else 0
            self.text(page, x + pad, y + h / 2 - size * 0.5, s, size=size,
                      color=color, align=align, width=w - pad * 2)
        return rect


# ───────── 메타데이터 표 ─────────

def _draw_meta_table(pen: _Pen, page, pipe_id: str, dia: str, m: dict) -> float:
    """야장 상단 표를 그리고 표 아래 y를 반환한다. m은 이 관로의 메타(공통+관로별)."""
    x0, x1 = MARGIN_X, PAGE_W - MARGIN_X
    W = x1 - x0
    y = TABLE_TOP

    def L(w):   # 라벨 셀(회색)
        return dict(fill=GRAY_FILL, size=7.5)

    # 1행: 사업명 | 값 | 보고서번호 | 값
    pen.cell(page, x0, y, W * .11, ROW_H, "사업명", **L(0))
    pen.cell(page, x0 + W * .11, y, W * .55, ROW_H, state.site_name, align="left")
    pen.cell(page, x0 + W * .66, y, W * .16, ROW_H, "보고서번호", **L(0))
    pen.cell(page, x0 + W * .82, y, W * .18, ROW_H, m.get("보고서번호", ""), align="left")
    y += ROW_H

    pen.cell(page, x0, y, W * .11, ROW_H, "발주처", **L(0))
    pen.cell(page, x0 + W * .11, y, W * .55, ROW_H, m.get("발주처", ""), align="left")
    pen.cell(page, x0 + W * .66, y, W * .16, ROW_H, "사업기간", **L(0))
    pen.cell(page, x0 + W * .82, y, W * .18, ROW_H, m.get("사업기간", ""), align="left")
    y += ROW_H

    pen.cell(page, x0, y, W * .11, ROW_H, "처리구역", **L(0))
    pen.cell(page, x0 + W * .11, y, W * .22, ROW_H, m.get("처리구역", ""), align="left")
    pen.cell(page, x0 + W * .33, y, W * .12, ROW_H, "배수구역", **L(0))
    pen.cell(page, x0 + W * .45, y, W * .21, ROW_H, m.get("배수구역", ""), align="left")
    pen.cell(page, x0 + W * .66, y, W * .16, ROW_H, "배수분구", **L(0))
    pen.cell(page, x0 + W * .82, y, W * .18, ROW_H, m.get("배수분구", ""), align="left")
    y += ROW_H

    pen.cell(page, x0, y, W * .11, ROW_H, "조사위치", **L(0))
    pen.cell(page, x0 + W * .11, y, W * .55, ROW_H, m.get("조사위치", ""), align="left")
    pen.cell(page, x0 + W * .66, y, W * .16, ROW_H, "시공자", **L(0))
    pen.cell(page, x0 + W * .82, y, W * .18, ROW_H, m.get("시공자", ""), align="left")
    y += ROW_H

    pen.cell(page, x0, y, W * .11, ROW_H, "조사목적", **L(0))
    pen.cell(page, x0 + W * .22, y, W * .11, ROW_H, "조사일자", **L(0))
    pen.cell(page, x0 + W * .11, y, W * .11, ROW_H, m.get("조사목적", ""), align="left")
    pen.cell(page, x0 + W * .33, y, W * .33, ROW_H, m.get("조사일자", ""), align="left")
    pen.cell(page, x0 + W * .66, y, W * .16, ROW_H, "조사자", **L(0))
    pen.cell(page, x0 + W * .82, y, W * .18, ROW_H, m.get("조사자", ""), align="left")
    y += ROW_H

    # 관로 제원
    heads = [("관로번호", .17), ("구분", .10), ("관종", .16), ("규격", .11),
             ("배수방식", .11), ("연장", .12), ("총주행거리", .12), ("미주행거리", .11)]
    cx = x0
    for name, frac in heads:
        pen.cell(page, cx, y, W * frac, ROW_H, name, **L(0))
        cx += W * frac
    y += ROW_H
    vals = [pipe_id, m.get("구분", ""), m.get("관종", ""), dia,
            m.get("배수방식", ""), m.get("연장", ""),
            m.get("총주행거리", ""), m.get("미주행거리", "")]
    cx = x0
    for (_, frac), v in zip(heads, vals):
        pen.cell(page, cx, y, W * frac, ROW_H, v)
        cx += W * frac
    y += ROW_H

    # 맨홀부
    mh = [("맨홀부", .11), ("맨홀번호", .18), ("맨홀종류", .13), ("맨홀재질", .13),
          ("맨홀깊이", .12), ("맨홀인버트", .13), ("맨홀크기", .10), ("사다리모양", .10)]
    cx = x0
    for name, frac in mh:
        pen.cell(page, cx, y, W * frac, ROW_H, name, **L(0))
        cx += W * frac
    y += ROW_H
    for side in ("상류맨홀", "하류맨홀"):
        row = [side, m.get(f"{side}번호", ""), m.get(f"{side}종류", ""),
               m.get(f"{side}재질", ""), m.get(f"{side}깊이", ""), "", "", ""]
        cx = x0
        for (_, frac), v in zip(mh, row):
            pen.cell(page, cx, y, W * frac, ROW_H, v,
                     fill=GRAY_FILL if v == side else None)
            cx += W * frac
        y += ROW_H

    # 좌표 — 자막에서 읽은 위경도가 있으면 쓰고, 없으면 야장 관례대로 '미측정'
    lat, lon = m.get("위도", ""), m.get("경도", "")
    coord = f"위도:{lat or '미측정'}/경도:{lon or '미측정'}"
    for left_lbl, right_lbl in (("상류맨홀좌표", "하류맨홀좌표"),):
        pen.cell(page, x0, y, W * .16, ROW_H, left_lbl, **L(0))
        pen.cell(page, x0 + W * .16, y, W * .34, ROW_H, coord, align="left", size=7)
        pen.cell(page, x0 + W * .50, y, W * .16, ROW_H, right_lbl, **L(0))
        pen.cell(page, x0 + W * .66, y, W * .34, ROW_H, coord, align="left", size=7)
        y += ROW_H

    grades = [("조사지점맨홀구조적상태등급", "맨홀구조적상태등급",
               "조사지점맨홀운영적상태등급", "맨홀운영적상태등급"),
              ("하수관로(암거)구조적상태등급", "관로구조적상태등급",
               "하수관로(암거)운영적상태등급", "관로운영적상태등급")]
    for l1, k1, l2, k2 in grades:
        pen.cell(page, x0, y, W * .28, ROW_H, l1, **L(0))
        pen.cell(page, x0 + W * .28, y, W * .22, ROW_H, m.get(k1, ""), align="left")
        pen.cell(page, x0 + W * .50, y, W * .28, ROW_H, l2, **L(0))
        pen.cell(page, x0 + W * .78, y, W * .22, ROW_H, m.get(k2, ""), align="left")
        y += ROW_H

    # 미주행
    pen.cell(page, x0, y, W * .12, ROW_H, "미주행방향", **L(0))
    pen.cell(page, x0 + W * .12, y, W * .13, ROW_H, "발생지점", **L(0))
    pen.cell(page, x0 + W * .25, y, W * .25, ROW_H, "미주행사유", **L(0))
    pen.cell(page, x0 + W * .50, y, W * .50, ROW_H, "미주행사유에 대한 설명 / 비고", **L(0))
    y += ROW_H
    # 완주 여부 + 주행방향은 실물 야장에서 이 칸에 "완주 역주행"처럼 적혀 있다
    note = m.get("미주행사유", "") or (f"완주 {m.get('주행방향', '')}".strip()
                                       if m.get("주행방향") else "")
    for d in ("상류->하류", "하류->상류"):
        pen.cell(page, x0, y, W * .12, ROW_H, d, size=7)
        pen.cell(page, x0 + W * .12, y, W * .13, ROW_H, "")
        pen.cell(page, x0 + W * .25, y, W * .25, ROW_H, "")
        pen.cell(page, x0 + W * .50, y, W * .50, ROW_H,
                 note if d == "하류->상류" else "", align="left")
        y += ROW_H

    return y


# ───────── 관로 모식도 ─────────

def _draw_pipe_rail(pen: _Pen, page, top: float, bottom: float,
                    anchors: List[Tuple[float, float, str]], total_text: str):
    """왼쪽 세로 관로 그림 + 거리 눈금.

    anchors = [(사진 왼쪽 x, 사진 중앙 y, 거리표기)] — 사진마다 하나씩 눈금을 찍고
    선으로 잇는다. 눈금은 관로 길이를 따라 위에서 아래로 순서대로 배치한다.
    """
    w = 9.0
    page.draw_rect(fitz.Rect(RAIL_X - w / 2, top, RAIL_X + w / 2, bottom),
                   color=LINE, fill=PIPE_FILL, width=0.7)

    # 위(상류)·아래(하류) 맨홀
    for y in (top, bottom):
        page.draw_rect(fitz.Rect(RAIL_X - 8, y - 3.5, RAIL_X + 8, y + 3.5),
                       color=LINE, fill=(1, 1, 1), width=0.7)

    if total_text:
        cy = (top + bottom) / 2
        pen.text(page, MARGIN_X - 2, cy - 9, "전체거리:", size=7, color=BLUE)
        pen.text(page, MARGIN_X - 2, cy + 1, total_text, size=7, color=BLUE)

    if not anchors:
        return

    # 눈금 y는 관로를 따라 균등 배치 (사진 위치와 1:1 대응)
    span = bottom - top - 24
    for i, (px, py, dist_txt, detour_y) in enumerate(anchors):
        my = top + 12 + span * (i / max(1, len(anchors) - 1))
        hub = fitz.Point(RAIL_X + 24, my)
        page.draw_line(fitz.Point(RAIL_X + w / 2, my), hub, color=ORANGE, width=0.6)

        if detour_y is None:
            page.draw_line(hub, fitz.Point(px - 3, py), color=ORANGE, width=0.6)
        else:
            # 사진 행 사이 여백까지 내려간 뒤 가로질러 올라간다 (사진을 안 가림)
            turn1 = fitz.Point(hub.x, detour_y)
            turn2 = fitz.Point(px - 3, detour_y)
            page.draw_line(hub, turn1, color=ORANGE, width=0.6)
            page.draw_line(turn1, turn2, color=ORANGE, width=0.6)
            page.draw_line(turn2, fitz.Point(px - 3, py), color=ORANGE, width=0.6)

        page.draw_circle(hub, 2.6, color=LINE, width=0.5)
        page.draw_rect(fitz.Rect(px - 5, py - 2, px - 1, py + 2), color=None, fill=ORANGE)
        if dist_txt:
            pen.text(page, RAIL_X + 29, my - 4, dist_txt, size=6.8, color=BLUE)


# ───────── 사진 격자 ─────────

def _draw_photos(pen: _Pen, page, chunk: List[dict], top: float, bottom: float) -> List[Tuple[float, float, str]]:
    """사진 6장(2열×3행)을 그리고, 모식도와 이을 (사진 왼쪽 중앙 x, y, 거리문구) 목록을 돌려준다."""
    avail_w = PAGE_W - MARGIN_X - PHOTO_LEFT
    cell_w = (avail_w - PHOTO_GAP_X) / PHOTO_COLS
    rows = (PHOTOS_PER_PAGE + PHOTO_COLS - 1) // PHOTO_COLS
    cell_h = (bottom - top - PHOTO_GAP_Y * (rows - 1)) / rows
    img_h = cell_h - CAPTION_H

    anchors = []
    for i, row in enumerate(chunk):
        r, c = divmod(i, PHOTO_COLS)
        x = PHOTO_LEFT + c * (cell_w + PHOTO_GAP_X)
        y = top + r * (cell_h + PHOTO_GAP_Y)
        rect = fitz.Rect(x, y, x + cell_w, y + img_h)

        fp = row.get("_image")
        if fp and Path(fp).exists():
            try:
                # 조사 사진은 증빙이라 비율을 바꾸면 안 된다. 남는 여백은 회색으로 채운다.
                page.draw_rect(rect, color=None, fill=(0.93, 0.93, 0.93))
                page.insert_image(rect, filename=str(fp), keep_proportion=True)
                page.draw_rect(rect, color=LINE, width=0.5)
            except Exception:
                page.draw_rect(rect, color=LINE, width=0.5)
        else:
            page.draw_rect(rect, color=LINE, fill=(0.93, 0.93, 0.93), width=0.5)
            pen.text(page, x, y + img_h / 2, "(이미지 없음)", size=7,
                     align="center", width=cell_w, color=(0.5, 0.5, 0.5))

        pen.text(page, x, y + img_h + 3, row["_caption"], size=7,
                 align="center", width=cell_w, color=BLUE)

        # 오른쪽 열은 선이 왼쪽 사진을 가로지르지 않도록 행 사이 여백으로 우회시킨다.
        detour_y = None if c == 0 else y + cell_h + PHOTO_GAP_Y * 0.4
        anchors.append((x, y + img_h / 2, row.get("_dist_text", ""), detour_y))
    return anchors


# ───────── 본체 ─────────

def _rows_for_pdf(video_name: str) -> List[dict]:
    """야장에 실을 행을 시간순으로 뽑고, 사진 경로·캡션·거리표기를 붙인다."""
    out = []
    for item in build_results_view():
        if item.get("filename") != video_name:
            continue
        children = item["children"] if item["type"] == "group" else ([item] if item["type"] == "row" else [])
        out.extend(children)

    v_data = state.video_data_map.get(video_name, {})
    prepared = []
    for r in sorted(out, key=lambda x: x["time_s"]):
        src = next((d for d in v_data.get("rows", []) if d["time"] == r["time_s"]), None)
        img = None
        if src:
            img = src.get("frame_annot_path") or src.get("frame_path")
        label = (r.get("defects") or ["ETC"])[0]
        dist = (r.get("dist") or "").strip()
        prepared.append({
            **r,
            "_image": img,
            "_caption": pipeasset_caption(label, r.get("grade", "중"), r["time_s"]),
            "_dist_text": f"{dist}m" if dist and not dist.endswith("m") else (dist or ""),
        })
    return prepared


def export_pipeasset_pdf(path: str) -> Optional[str]:
    """야장 PDF를 path에 저장. 실패 시 에러 메시지, 성공 시 None."""
    font_path = _find_font()
    if not font_path:
        return ("PDF에 쓸 한글 폰트를 찾지 못했습니다. "
                "맑은 고딕(malgun.ttf) 또는 나눔고딕이 설치돼 있어야 합니다.")
    if not state.video_data_map:
        return "내보낼 결과가 없습니다. 먼저 영상을 분석하세요."

    doc = fitz.open()
    pen = _Pen(doc, font_path)

    for video_name, v_data in state.video_data_map.items():
        rows = _rows_for_pdf(video_name)
        if not rows:
            continue
        # 맨홀번호·관종·거리는 관로마다 다르다 — 이 영상의 값을 쓴다
        m = state.meta_for(video_name)
        pipe_id = v_data.get("pipe_id", "")
        dia = v_data.get("dia", "")

        chunks = [rows[i:i + PHOTOS_PER_PAGE] for i in range(0, len(rows), PHOTOS_PER_PAGE)]
        for page_no, chunk in enumerate(chunks, 1):
            page = pen.new_page()

            pen.text(page, 0, TITLE_Y, "하수관거 현황 조사 보고서", size=17,
                     align="center", width=PAGE_W)

            body_top = _draw_meta_table(pen, page, pipe_id, dia, m) + 22
            body_bottom = PAGE_H - 60

            marks = _draw_photos(pen, page, chunk, body_top, body_bottom)
            _draw_pipe_rail(pen, page, body_top, body_bottom, marks, m.get("연장", ""))

            pen.text(page, MARGIN_X, PAGE_H - 34,
                     f"보고서번호:{m.get('보고서번호', '')} / 관로번호:{pipe_id} - ({page_no})", size=7.5)
            pen.text(page, PAGE_W - MARGIN_X - 150, PAGE_H - 34, m.get("조사자", ""),
                     size=7.5, align="right", width=150)

    if doc.page_count == 0:
        doc.close()
        return "내보낼 결함 행이 없습니다."

    try:
        doc.save(path, garbage=3, deflate=True)
        return None
    except Exception as e:
        return str(e)
    finally:
        doc.close()

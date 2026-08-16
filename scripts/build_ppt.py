"""2차 심사 발표 PPT를 만든다 (14장, 16:9).

구성은 `docs/reports/2026-08-16-발표슬라이드-구성안.md` 그대로다.
디자인은 참고 PPT([환경시설분과] 하수관로 CCTV(260203)_처장님.pptx)에서
색·폰트를 따왔다 — 맑은 고딕, 강조색 #017BB1.

슬라이드는 **말할 거리를 담되 읽히지 않게** 만든다. 7분에 14장이면 장당 30초라
글을 채우면 발표자가 읽다가 끝난다. 숫자와 그림 위주로 두고, 자세한 설명은
발표자 노트에 넣는다.
"""
import io
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

W, H = Inches(13.333), Inches(7.5)
BLUE = RGBColor(0x01, 0x7B, 0xB1)
NAVY = RGBColor(0x30, 0x57, 0xB9)
DARK = RGBColor(0x2B, 0x2B, 0x2B)
GRAY = RGBColor(0x5A, 0x5A, 0x5A)
LIGHT = RGBColor(0xEA, 0xEA, 0xEA)
GREEN = RGBColor(0x2E, 0x9E, 0x5B)
RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "맑은 고딕"

SP = Path(__file__).parent
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def tb(slide, x, y, w, h, text, size=18, bold=False, color=DARK,
       align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line=1.25):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, ln in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = FONT
    return box


def rect(slide, x, y, w, h, fill=BLUE, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def header(slide, kicker, title):
    """상단 머리말 — 파트 표시 + 제목. 모든 본문 슬라이드에 같은 자리."""
    rect(slide, Inches(0), Inches(0), W, Inches(0.10), fill=BLUE)
    if kicker:
        tb(slide, Inches(0.62), Inches(0.32), Inches(9), Inches(0.34),
           kicker, size=13, bold=True, color=BLUE)
    tb(slide, Inches(0.6), Inches(0.62), Inches(12), Inches(0.72),
       title, size=28, bold=True, color=DARK)


def note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def pic(slide, name, x, y, w=None, h=None):
    f = SP / name
    if not f.exists():
        print("  ! 이미지 없음:", name); return None
    return slide.shapes.add_picture(str(f), x, y, width=w, height=h)


# ── 1. 표지 ──────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
rect(s, Inches(0), Inches(0), W, H, fill=RGBColor(0x0E, 0x2B, 0x3E))
rect(s, Inches(0), Inches(3.02), W, Inches(0.06), fill=BLUE)
tb(s, Inches(1.0), Inches(1.85), Inches(11.3), Inches(1.1),
   "AI 기반 하수관로 CCTV 영상 자동분석 시스템", size=40, bold=True,
   color=WHITE, align=PP_ALIGN.CENTER)
tb(s, Inches(1.0), Inches(3.25), Inches(11.3), Inches(0.6),
   "감독자가 40시간 보던 영상을, 이제 확인만 하면 됩니다",
   size=18, color=RGBColor(0x9F, 0xC9, 0xDE), align=PP_ALIGN.CENTER)
tb(s, Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.0),
   "수도권동부환경본부 환경시설관리처 시설사업5부\n수어사이트 스쿼드(Sewer Sight Squad)",
   size=15, color=RGBColor(0xC8, 0xD6, 0xDE), align=PP_ALIGN.CENTER)
note(s, "인사와 팀 소개. 10초 안에 넘어간다.")

# ── 2. 문제 정의 ─────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅰ. 개요", "지금은 사람이 영상을 처음부터 끝까지 봅니다")
cards = [("40시간", "현장 1개소(30km)\n판독 소요시간 / 1인", RED),
         ("숙련도 편차", "검토자에 따라\n누락·오분류가 갈린다", GRAY),
         ("수기 작성", "관경·거리·관로번호를\n손으로 옮겨 적는다", GRAY)]
for i, (big, sub, col) in enumerate(cards):
    x = Inches(0.8 + i * 4.1)
    rect(s, x, Inches(2.1), Inches(3.7), Inches(2.5), fill=RGBColor(0xF5, 0xF7, 0xF9))
    tb(s, x, Inches(2.45), Inches(3.7), Inches(0.9), big, size=30, bold=True,
       color=col, align=PP_ALIGN.CENTER)
    tb(s, x, Inches(3.45), Inches(3.7), Inches(1.0), sub, size=14,
       color=GRAY, align=PP_ALIGN.CENTER)
rect(s, Inches(0.8), Inches(5.25), Inches(11.73), Inches(0.95), fill=BLUE)
tb(s, Inches(0.8), Inches(5.4), Inches(11.73), Inches(0.7),
   "영상을 다 볼 필요 없이, 봐야 할 곳만 보게 하자", size=22, bold=True,
   color=WHITE, align=PP_ALIGN.CENTER)
note(s, "30초. 세 가지 문제를 빠르게 짚고 마지막 한 줄로 목적을 못박는다.")

# ── 3. 추진 경과 ─────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅰ. 개요", "작은 검증에서 시작해 본격 구축까지")
rows = [("1단계", "~2026 상반기", "생성형 AI 직접 판독", "신설관로에서 가능성 확인"),
        ("2단계", "2026 하반기", "공공데이터 51만장으로 딥러닝 학습",
         "노후관로·위치까지 확장"),
        ("3단계", "현재", "판독기 3중 구조로 통합", "감독자는 확인만")]
for i, (stage, when, what, got) in enumerate(rows):
    y = Inches(2.05 + i * 1.42)
    col = BLUE if i < 2 else GREEN
    rect(s, Inches(0.8), y, Inches(1.5), Inches(1.15), fill=col)
    tb(s, Inches(0.8), y + Inches(0.16), Inches(1.5), Inches(0.4), stage,
       size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    tb(s, Inches(0.8), y + Inches(0.62), Inches(1.5), Inches(0.35), when,
       size=11, color=RGBColor(0xDC, 0xEA, 0xF2), align=PP_ALIGN.CENTER)
    tb(s, Inches(2.55), y + Inches(0.06), Inches(5.5), Inches(0.6), what,
       size=17, bold=True, color=DARK)
    tb(s, Inches(2.55), y + Inches(0.62), Inches(9.6), Inches(0.5),
       "→ " + got, size=14, color=GRAY)
note(s, '30초. "작년엔 생성형 AI였는데 왜 YOLO냐"에 미리 답하는 슬라이드. '
        '"작게 검증하고 크게 키웠다"는 서사로 말할 것.')

# ── 4. 왜 로컬인가 ───────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅱ. 시연", "지금 보실 화면은 이 노트북 한 대에서 돕니다")
items = [("비용", "GPU 서버를 쓰면 더 빠르지만 월 비용이 발생합니다.\n"
                  "시범 단계에서는 로컬 CPU만으로 동작하도록 만들었습니다."),
         ("보안", "관로 위치·GPS가 담긴 영상이 외부로 나가지 않습니다.\n"
                  "망분리 환경에서도 그대로 쓸 수 있습니다."),
         ("확장", "서버가 준비되면 그쪽 GPU로 넘기는 경로도\n설계에 들어 있습니다.")]
for i, (k, v) in enumerate(items):
    y = Inches(2.15 + i * 1.35)
    rect(s, Inches(0.85), y, Inches(1.35), Inches(0.95), fill=BLUE)
    tb(s, Inches(0.85), y + Inches(0.22), Inches(1.35), Inches(0.5), k,
       size=17, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    tb(s, Inches(2.45), y + Inches(0.02), Inches(10.2), Inches(1.0), v,
       size=15, color=DARK)
tb(s, Inches(0.85), Inches(6.25), Inches(11.6), Inches(0.5),
   "프레임당 약 0.44초 — 6분 영상이 1~2분이면 끝납니다", size=15,
   bold=True, color=GREEN)
note(s, '20초. 변명이 아니라 "설계 선택"으로 말할 것. '
        '이걸 미리 말해두면 "왜 느리냐"는 질문을 막는다.')

# ── 5. 시연 영상 ─────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅱ. 시연", "이 다섯 단계가 전부입니다")
pic(s, "fig_flow.png", Inches(0.5), Inches(1.9), w=Inches(12.3))
rect(s, Inches(3.5), Inches(4.3), Inches(6.3), Inches(1.9),
     fill=RGBColor(0xF5, 0xF7, 0xF9))
tb(s, Inches(3.5), Inches(4.75), Inches(6.3), Inches(1.0),
   "▶  시연 영상 삽입 자리\n(30~60초 · 사전 녹화본)", size=20, bold=True,
   color=GRAY, align=PP_ALIGN.CENTER)
note(s, "60초. 말은 최소로. 영상이 다 말해준다. "
        "40시간이 몇 분으로 줄어드는 것이 눈에 보이게 하는 것이 목적.\n"
        "※ 발표 전 이 자리에 녹화 영상을 삽입할 것.")

# ── 6. 전체 구조 ─────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅲ. 어떻게 동작하나", "방금 보신 것을 단계별로 다시 보겠습니다")
pic(s, "fig_flow.png", Inches(0.5), Inches(2.0), w=Inches(12.3))
tb(s, Inches(0.6), Inches(4.5), Inches(12.1), Inches(0.6),
   "①~⑤ 전 과정이 감독자 1인 조작만으로 끝납니다", size=19, bold=True,
   color=BLUE, align=PP_ALIGN.CENTER)
note(s, "40초. 여기서부터 '되짚기' 파트로 전환. 각 단계에 기술 설명을 얹어 간다.")

# ── 7. 판독 3중 구조 (핵심) ──────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅲ. 어떻게 동작하나", "성격이 다른 판독기 셋이 같은 화면을 봅니다")
pic(s, "fig_arch.png", Inches(1.35), Inches(1.65), w=Inches(10.6))
note(s, "60초. 이 발표의 핵심 슬라이드. 세 가지를 말한다.\n"
        "1) 셋이 각자 독립적으로 본다 — 하나가 놓쳐도 다른 쪽이 잡는다\n"
        "2) 결과는 합집합 — 놓치는 것을 줄이는 쪽으로 설계했다. "
        "검수자가 지우는 편이 못 찾는 것보다 낫다\n"
        "3) LLM은 껐다 켤 수 있다 — 외부 API라 망분리 환경 고려")

# ── 8. CLS ───────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅲ. 어떻게 동작하나", "CLS — 볼 곳을 고르고 이름을 답니다")
left = [("학습 데이터", "24클래스 11,495장\n결함 종류당 400장으로 균등화"),
        ("모델", "EfficientNet-B0 · 384px\nColab T4"),
        ("핵심 설계", "정상의 20%를 '관 밖(맨홀·지상)'으로")]
for i, (k, v) in enumerate(left):
    y = Inches(2.0 + i * 1.35)
    tb(s, Inches(0.8), y, Inches(2.2), Inches(0.5), k, size=14, bold=True, color=BLUE)
    tb(s, Inches(0.8), y + Inches(0.42), Inches(5.4), Inches(0.9), v, size=15, color=DARK)
pic(s, "fig_outside.png", Inches(6.6), Inches(1.85), w=Inches(6.1))
note(s, "40초. 관 밖 오탐 39%→0%가 이 설계의 결과. "
        "'맨홀이나 지상 장면을 결함으로 잘못 잡던 문제를 없앴다'로 말하면 쉽다.")

# ── 9. DET ───────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅲ. 어떻게 동작하나", "DET — 결함의 위치를 찾아 표시합니다")
items = [("학습 데이터", "18클래스 133,961장\n공공데이터포털 51만장에서 선별"),
         ("모델", "YOLO11-L · 960px · 50 epoch\n학습 약 29시간"),
         ("결과", "mAP50 0.703")]
for i, (k, v) in enumerate(items):
    y = Inches(2.05 + i * 1.4)
    rect(s, Inches(0.85), y, Inches(1.9), Inches(1.0), fill=RGBColor(0xF0, 0xF4, 0xF7))
    tb(s, Inches(0.85), y + Inches(0.28), Inches(1.9), Inches(0.5), k,
       size=14, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    tb(s, Inches(3.0), y + Inches(0.1), Inches(9.5), Inches(0.9), v,
       size=16, color=DARK)
rect(s, Inches(0.85), Inches(6.2), Inches(11.6), Inches(0.75), fill=RGBColor(0xF5, 0xF7, 0xF9))
tb(s, Inches(0.85), Inches(6.33), Inches(11.6), Inches(0.5),
   "공공데이터포털 하수관로 이미지 51만장 활용", size=15, bold=True,
   color=GREEN, align=PP_ALIGN.CENTER)
note(s, "40초. 가점 항목(공공데이터 활용)을 여기서 한 줄로 연결한다.")

# ── 10. 성능 ─────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅲ. 어떻게 동작하나", "둘을 합치면 놓치는 것이 크게 줄어듭니다")
pic(s, "fig_perf.png", Inches(0.9), Inches(1.75), w=Inches(6.6))
box_items = [("겹치는 것은 12%뿐", "두 판독기가 거의 다른 결함을 맞힙니다", GREEN),
             ("구간은 놓치지 않습니다", "CLS 기준 23종 전부 탐지율 80% 이상\n"
              "(14종은 100%)", BLUE)]
for i, (t, d, c) in enumerate(box_items):
    y = Inches(2.2 + i * 1.85)
    rect(s, Inches(7.9), y, Inches(4.75), Inches(1.5), fill=RGBColor(0xF5, 0xF7, 0xF9))
    tb(s, Inches(8.15), y + Inches(0.18), Inches(4.3), Inches(0.5), t,
       size=17, bold=True, color=c)
    tb(s, Inches(8.15), y + Inches(0.72), Inches(4.3), Inches(0.7), d,
       size=13, color=GRAY)
tb(s, Inches(7.9), Inches(6.1), Inches(4.9), Inches(0.6),
   "※ 검증 데이터셋 기준 실측", size=11, color=GRAY)
note(s, "50초. 핵심 메시지는 '겹치는 게 12%뿐'.\n"
        "※ 정직하게: 이 값은 잘라놓은 사진 기준이고 연속 실영상 검증은 다음 단계다. "
        "질문 받으면 그렇게 답한다.")

# ── 11. 결과물 ───────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅲ. 어떻게 동작하나", "손으로 옮겨 적던 항목이 자동으로 채워집니다")
tb(s, Inches(0.85), Inches(1.75), Inches(5.6), Inches(0.5),
   "영상 자막에서 읽어오는 10항목", size=17, bold=True, color=BLUE)
ocr = ["관로번호", "관경", "거리", "현장명", "상류·하류 맨홀번호",
       "조사일자", "관종", "배수방식", "주행방향", "위경도"]
for i, it in enumerate(ocr):
    x = Inches(0.9 + (i % 2) * 2.75)
    y = Inches(2.35 + (i // 2) * 0.52)
    tb(s, x, y, Inches(2.7), Inches(0.4), "· " + it, size=14, color=DARK)
rect(s, Inches(6.9), Inches(1.75), Inches(5.7), Inches(4.6),
     fill=RGBColor(0xF5, 0xF7, 0xF9))
tb(s, Inches(6.9), Inches(3.6), Inches(5.7), Inches(1.0),
   "야장 PDF 출력물\n스크린샷 자리", size=18, bold=True, color=GRAY,
   align=PP_ALIGN.CENTER)
tb(s, Inches(0.85), Inches(5.5), Inches(5.7), Inches(0.9),
   "실물 조사 서식과 같은 A4 양식으로\n바로 출력됩니다", size=15, color=DARK)
note(s, "40초. 야장 PDF를 크게 보여준다. "
        "'자막 → 야장 칸' 화살표를 넣으면 이해가 빠르다.\n"
        "※ 발표 전 야장 PDF 스크린샷을 오른쪽에 넣을 것.")

# ── 12. 업무 효율화 ──────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅳ. 결론", "판독 시간을 90% 이상 줄입니다")
pic(s, "fig_eff.png", Inches(0.9), Inches(1.8), w=Inches(6.3))
tbl = [("감독자 역할", "전담 판독", "AI 결과 최종 승인"),
       ("보고서 작성", "수기 작성", "자동 생성"),
       ("판독 일관성", "숙련도에 따라 상이", "표준 기준 일괄 적용")]
tb(s, Inches(7.7), Inches(1.95), Inches(5.0), Inches(0.4), "구분", size=12,
   bold=True, color=GRAY)
tb(s, Inches(9.35), Inches(1.95), Inches(1.6), Inches(0.4), "현행", size=12,
   bold=True, color=GRAY)
tb(s, Inches(11.1), Inches(1.95), Inches(1.7), Inches(0.4), "도입 후", size=12,
   bold=True, color=GREEN)
for i, (k, a, b) in enumerate(tbl):
    y = Inches(2.5 + i * 1.05)
    tb(s, Inches(7.7), y, Inches(1.7), Inches(0.5), k, size=13, bold=True, color=DARK)
    tb(s, Inches(9.35), y, Inches(1.7), Inches(0.5), a, size=13, color=GRAY)
    tb(s, Inches(11.1), y, Inches(1.9), Inches(0.5), b, size=13, bold=True, color=GREEN)
note(s, "40초. 팀의 대표 슬라이드. 40시간 → 4시간을 강조.")

# ── 13. 확산 계획 ────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, "Ⅴ. 향후계획", "시범 적용에서 전국 확산까지")
plan = [("①", "2026", "개발 · 시범", "수도권동부환경본부 관할\n관로 현장 시범 적용", BLUE),
        ("②", "2027", "고도화 · 확산", "사용자 피드백 반영\n본사 및 전국 지역본부 배포", NAVY),
        ("③", "2028~", "업무 확대", "노후하수관로 진단\n상태등급 자동평가 연계", GREEN)]
for i, (num, yr, title, desc, col) in enumerate(plan):
    x = Inches(0.85 + i * 4.05)
    rect(s, x, Inches(2.0), Inches(3.7), Inches(3.5), fill=RGBColor(0xF5, 0xF7, 0xF9))
    rect(s, x, Inches(2.0), Inches(3.7), Inches(0.95), fill=col)
    tb(s, x, Inches(2.16), Inches(3.7), Inches(0.55), f"{num}  {yr}", size=19,
       bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    tb(s, x, Inches(3.2), Inches(3.7), Inches(0.5), title, size=17, bold=True,
       color=DARK, align=PP_ALIGN.CENTER)
    tb(s, x + Inches(0.3), Inches(3.95), Inches(3.1), Inches(1.3), desc,
       size=13, color=GRAY, align=PP_ALIGN.CENTER)
note(s, "30초. 확산성 항목(20점) 보강용. 빠르게 넘어간다.")

# ── 14. 마무리 ───────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
rect(s, Inches(0), Inches(0), W, H, fill=RGBColor(0x0E, 0x2B, 0x3E))
tb(s, Inches(1.0), Inches(2.7), Inches(11.3), Inches(1.2),
   "감독자가 40시간 보던 영상을,\n이제 확인만 하면 됩니다", size=34, bold=True,
   color=WHITE, align=PP_ALIGN.CENTER, line=1.35)
rect(s, Inches(5.6), Inches(4.6), Inches(2.1), Inches(0.05), fill=BLUE)
tb(s, Inches(1.0), Inches(5.1), Inches(11.3), Inches(0.6),
   "감사합니다", size=20, color=RGBColor(0x9F, 0xC9, 0xDE), align=PP_ALIGN.CENTER)
note(s, "20초. 질의응답으로 넘어간다.")

out = Path(r"E:\AI_CCTV\docs\reports\2026-08-16-2차심사-발표자료.pptx")
prs.save(out)
print(f"\n저장: {out}")
print(f"슬라이드 {len(prs.slides.__iter__.__self__._sldIdLst)}장")

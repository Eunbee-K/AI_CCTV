import os
import sys
from pathlib import Path


def resource_path(rel: str) -> Path:
    """Dev 실행과 PyInstaller(onedir/onefile) 실행 모두에서 리소스 경로를 통일해서 반환."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / rel


APP_ROOT = Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    """분석 결과·프레임·업로드 영상을 두는 고정 폴더.

    tempfile.mkdtemp()는 실행할 때마다 새로 만들어져서 서버를 껐다 켜면 결과가
    통째로 사라진다. OS가 청소하지 않는 사용자 데이터 폴더에 둬야 재시작 후에도
    살아남는다. AI_CCTV_DATA_DIR 환경변수로 옮길 수 있다.
    """
    env = os.getenv("AI_CCTV_DATA_DIR")
    if env:
        return Path(env)
    base = os.getenv("LOCALAPPDATA") or os.getenv("XDG_DATA_HOME")
    if base:
        return Path(base) / "AI_CCTV"
    return Path.home() / ".ai_cctv"


DATA_DIR = _default_data_dir()
SESSION_FILE = DATA_DIR / "session.json"

# ───────── YOLO 설정 ─────────
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", str(resource_path("assets/best.pt"))))
# Colab 원격 추론 서버 기본 주소 (ngrok 고정 도메인). 빈 문자열이면 로컬 CPU 추론.
REMOTE_YOLO_URL = os.getenv("REMOTE_YOLO_URL", "https://financial-elitism-ebook.ngrok-free.dev")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.10"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))
# 로컬 추론에서 한 번에 모델에 넘기는 프레임 수. 리스트를 통째로 넘기면
# ultralytics가 전부 한 배치로 올려 수 GB를 요구한다(112장 → 3.7GB 할당 실패).
# 원격 경로의 REMOTE_CHUNK_SIZE와 같은 취지.
YOLO_CHUNK_SIZE = int(os.getenv("YOLO_CHUNK_SIZE", "8"))
# 결함이 아닌 맥락 클래스. 모델은 학습 안정을 위해 이들을 클래스로 갖지만
# 조사표에는 올리지 않는다. PJ(정상 이음부)·IN(관 내부)은 정상이고,
# OUT_*(맨홀·인버트·자동차)은 관 밖 장면이다.
YOLO_IGNORE_CLASSES = {
    x.strip().lower()
    for x in os.getenv(
        "YOLO_IGNORE_CLASSES",
        "normal,none,background,ok,pj,in,out_mh,out_invert,out_car",
    ).split(",")
    if x.strip()
}

# ───────── Stage-1 정상/결함 필터 ─────────
# YOLO 앞단(또는 옆)에서 프레임에 결함이 있는지만 판정하는 분류기.
#
#   off       쓰지 않는다
#   lead      **필터가 잡은 구간이 곧 결함 리스트다(기본).** YOLO는 그 구간에
#             이름만 붙인다. 결함을 찾는 일은 필터가, 이름 붙이는 일은 YOLO가
#             한다는 역할 분담을 그대로 옮긴 것이다.
#   series    점수 상위 FILTER_TOP_RATIO만 YOLO에 넘긴다. 나머지는 버린다.
#             행은 YOLO가 만든다 — lead와 다르다.
#   parallel  모든 프레임을 양쪽에 넣고 두 판단을 대조한다. 아무것도 버리지 않는
#             대신 YOLO 오탐/누락 의심을 표에 표시한다.
#
# **2026-08-12 기준 off — 실영상에서 결함을 가려내지 못한다.**
#
# 배포 모델 OLD_v3는 야장 val에서 AUC 0.9503이고 그 값은 배포본(ONNX)에서도 정확히
# 재현된다. 그런데 실제 연속 영상에 돌리면 결함과 그 외가 갈라지지 않는다.
#
#   야장 val 2,580장    결함 중앙 0.9412 / 정상 중앙 0.0370   AUC 0.9503
#   SM2 실영상 1,014장  결함 중앙 0.9420 / 그외 중앙 0.9380   AUC 0.5205
#   JB1-2-0413 128장    조사표 결함 2건이 50위 · 74위 (무작위 기댓값 64위)
#
# v1(AIHub 78,388장)이 현장에서 0.516이었을 때와 **같은 모양**이다. 그때는 정상이
# 전부 AIHub라 도메인이 달랐고, 지금은 학습 데이터에 "조사원이 기록한 순간"만 있고
# 그 사이를 흐르는 평범한 구간이 아예 없다. 실영상은 대부분이 그 구간이다.
#
# 8/11에 "상위 20%로 잘라도 결함 유실 0건"이라고 본 것은 착시였다. 정답 구간이 전체
# 프레임의 24.6%나 돼서(조사표 거리 ±1m를 정답으로 잡았는데 카메라가 느리다),
# **무작위로 20%를 골라도 유실이 0.4건**이다. 필터의 0건은 무작위와 구별되지 않는다.
#
# 되살리려면 학습 데이터에 연속 영상의 평범한 구간을 정상으로 넣고, 조사표가 아니라
# 사람이 프레임 단위로 찍은 정답으로 다시 재야 한다. 그 전에는 켜지 않는다.
#
# **이 모델은 노후관로(우수관) 전용이다.** 관로구분이 '신설'이면 자동으로 건너뛴다
# (analysis.py). 신설용 필터는 데이터 부족으로 보류 — 현장 1,404장으로는 AUC 0.75.
# 2026-08-12 오후: lead. 위 실측대로 필터의 순위 품질은 아직 검증되지 않았지만,
# lead는 프레임을 버리지 않고 "무엇을 볼지"를 정할 뿐이라 틀려도 되돌릴 수 있다.
# 필터가 놓친 곳에서 YOLO가 뭔가 찾으면 로그에 남기므로 조용히 사라지지 않는다.
FILTER_MODE = os.getenv("FILTER_MODE", "lead").strip().lower()
FILTER_MODEL_PATH = Path(
    os.getenv("FILTER_MODEL_PATH", str(resource_path("assets/filter.onnx")))
)
# series에서 남길 비율. **절대 임계값은 쓰지 않는다** — 임계값은 점수 분포에 붙어
# 있는 값인데 야장은 결함 비율 45%, 실영상은 1.5%라 옮겨가지 않는다(야장에서
# "결함 99% 재현"이던 0.032가 실영상에서는 프레임의 0.6%밖에 못 걸렀다).
# 순위는 분포가 달라도 유지되므로 그나마 낫지만, 위에 적었듯 지금 모델로는
# 순위 자체가 무작위다. 이 값은 필터를 되살렸을 때 쓸 자리만 잡아둔 것이다.
#
# **lead 모드에서는 이 값이 곧 검토 리스트의 길이를 정한다.** 상위 N%를 뽑아
# 이어진 구간끼리 묶으면 그 구간 수가 표의 행 수가 된다. SM2 8개 관로 실측:
#
#   상위 10% -> 69행    상위 15% -> 91행
#   상위 20% -> 105행   상위 30% -> 112행     (실제 조사표 결함은 13건)
#
# 20%면 관로당 7~26행이다. 길면 줄이고, 놓치는 게 있으면 늘린다.
FILTER_TOP_RATIO = float(os.getenv("FILTER_TOP_RATIO", "0.20"))
# parallel에서 "결함 의심" 표시에만 쓰는 절대 임계값. 프레임을 버리지 않는다.
# 야장 val 2,580장에서 결함 재현율 99% 지점. 실영상에서는 의미가 없다(위 참고).
# lead/series는 이 값을 쓰지 않는다 — 순위로 자른다.
FILTER_THRESHOLD = float(os.getenv("FILTER_THRESHOLD", "0.032"))
FILTER_BATCH_SIZE = int(os.getenv("FILTER_BATCH_SIZE", "16"))
# 필터가 만드는 행의 상한(영상당). **0이면 무제한** — lead 모드는 필터가 잡은 것을
# 전부 보여주는 게 목적이므로 기본이 무제한이다. 필터가 오작동해 표가 수백 줄로
# 불어나면 여기에 숫자를 넣어 막는다.
FILTER_MAX_MISS_ROWS = int(os.getenv("FILTER_MAX_MISS_ROWS", "0"))

# ───────── LLM 판독 (세 번째 의견) ─────────
# 필터가 고른 구간에 YOLO와 나란히 이름을 붙이는 두 번째 판독자.
# ver_2.2 앱에서 쓰던 Gemini+GPT 구조를 옮겨왔다(backend/llm_infer.py).
#
# **회사망에서는 못 쓴다** — 외부 API를 호출하므로 망분리 환경에서는 켜지지 않는다.
# 그래서 기본이 꺼짐이고, 앱 화면에서 껐다 켰다 한다(state.llm_enabled).
# LLM_ENABLED=1 로 시작하면 처음부터 켜진 채로 뜬다.
LLM_ENABLED_DEFAULT = os.getenv("LLM_ENABLED", "").strip().lower() in ("1", "true", "on")
LLM_GEMINI_MODEL = os.getenv("LLM_GEMINI_MODEL", "gemini-3-flash-preview")
LLM_GPT_MODEL = os.getenv("LLM_GPT_MODEL", "gpt-5")
# 한 번에 보내는 프레임 수. 늘리면 호출이 줄지만 응답이 길어져 잘려 나가기 쉽다.
LLM_CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", "7"))
# 프롬프트에 같이 보낼 예시 사진 폴더(ver.2.0 학습데이터). `labels.jsonl`이 있으면
# 그걸 정답으로 읽고, 없으면 파일명 `{날짜}_{라벨}(번호).jpg`에서 라벨을 뽑는다.
# 폴더가 없으면 글로만 설명하고 넘어간다 — 예시가 없다고 판독을 멈추지는 않는다.
#
# 원본은 이동식 드라이브에 있어 PC마다 문자가 바뀐다(E:, F: ...). 그래서 이 앱이
# 놓인 드라이브를 먼저 보고, 없으면 ver_2.2가 쓰던 바탕화면 경로를 본다.
def _find_examples_dir() -> Path:
    env = os.getenv("LLM_EXAMPLES_DIR", "").strip()
    if env:
        return Path(env)
    rel = Path("CCTV/ver.2.0_학습데이터")
    candidates = [
        Path(__file__).resolve().anchor / rel,          # 이 앱이 놓인 드라이브
        Path.home() / "Desktop" / rel,                  # ver_2.2가 쓰던 자리
    ]
    return next((c for c in candidates if c.exists()), candidates[0])


LLM_EXAMPLES_DIR = _find_examples_dir()
# 라벨당 몇 장을 보낼지. ver_2.2는 1장이었는데, 늘리면 판독이 안정되는 대신
# 호출이 무거워진다(라벨 9종 × N장이 매 청크마다 따라붙는다).
LLM_EXAMPLES_PER_LABEL = int(os.getenv("LLM_EXAMPLES_PER_LABEL", "2"))
# 동시에 띄우는 API 호출 수. 호출은 대부분 대기 시간이라 늘리면 그만큼 빨라지는데,
# 너무 늘리면 429(rate limit)가 난다. 88프레임 기준 2일 때 456초였다.
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "6"))
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "600"))

# 필요하면 여기에 클래스별 한글 표시명을 추가/수정. 목록에 없는 클래스는
# best.pt에 저장된 클래스명이 그대로 표시된다.
YOLO_CLASS_MAP = {
    # test5는 변형 클래스를 DEFORM으로 학습했다. 코드표(DEFECT_CODE_KO)와
    # 조사표는 DF를 쓰므로 여기서 맞춰준다. 이걸 안 하면 표에 'DEFORM'이
    # 날것으로 찍히고 한글명도 안 붙는다.
    "DEFORM": "DF",
    "breakage": "파손",
    "separation": "이탈",
    "damage": "손상",
    "subsidence": "침하",
    "obstacle": "연결구장애물",
    "crack": "균열",
    "intruding_pipe": "가지관 돌출",
    "intruding pipe": "가지관 돌출",
    "other": "기타",
}

# ───────── 결함 코드 ↔ 한글명 ─────────
# 하수관로 상태등급 매뉴얼 기준 25종(0~24) + 데이터셋 보조 클래스(25~30).
# 출처: docs/메타데이터(총괄).xlsx, scripts/sweep/label_formats.py::GLOBAL_CLASS_ID
DEFECT_CODE_KO = {
    "CC": "균열-원주",
    "CL": "균열-길이",
    "CM": "균열-복합",
    "SD": "표면손상",
    "BC": "좌굴",
    "LD": "라이닝 결함",
    "DF": "변형",
    "BK": "파손",
    "CX": "붕괴",
    "PO": "영구장애물",
    "HL": "천공",
    "LP": "연결관-돌출",
    "LS": "연결관-접합부이상",
    "JS": "이음부-이탈",
    "JF": "이음부-손상",
    "JD": "이음부-단차",
    "NS": "역경사",
    "SG": "침하",
    "DE": "내피생성",
    "DS": "토사퇴적",
    "DG": "폐유부착",
    "TO": "임시장애물",
    "RT": "뿌리침입",
    "IF": "침입수",
    "PB": "막힘",
    # 매뉴얼 25종에는 없지만 데이터셋에 존재하는 클래스
    "PJ": "이음부(정상)",
    "ETC": "기타",
    # test5/test6은 영구장애물(PO)과 임시장애물(TO)을 하나로 묶어 학습했다.
    # 둘을 나누면 정확도가 떨어져서 합친 것이라, 검수자가 표에서 구분해 준다.
    "OBST": "장애물",
    "IN": "내부(정상)",
    "OUT_MH": "외부-맨홀",
    "OUT_INVERT": "외부-인버트",
    "OUT_CAR": "외부-자동차",
}

# 모델이 코드 대신 한글을 내보내는 경우(현재 best.pt + YOLO_CLASS_MAP)도 있어서 역방향도 둔다
_DEFECT_KO_TO_CODE = {ko: code for code, ko in DEFECT_CODE_KO.items()}


# 파이프에셋 야장(하수관거 현황 조사 보고서) 캡션 표기.
# 야장은 "(구분)결함명(등급) [시각]" 형식이고, 결함명이 우리 코드표와 미묘하게 다르다
# (하이픈 없음: 균열-원주 → 균열원주). 구분은 실제 관로 결함이면 "관로",
# 정상 구조물이나 조사 진행 상태면 "상태"를 쓴다.
# 출처: 2.3b CCTV조사 야장 (가평군 노후하수관로 정밀조사용역) 실물 PDF
DEFECT_PIPEASSET = {
    "CC": ("관로", "균열원주"),
    "CL": ("관로", "균열길이"),
    "CM": ("관로", "균열복합"),
    "SD": ("관로", "표면손상"),
    "BC": ("관로", "좌굴"),
    "LD": ("관로", "라이닝결함"),
    "DF": ("관로", "변형"),
    "BK": ("관로", "파손"),
    "CX": ("관로", "붕괴"),
    "PO": ("관로", "영구장애물"),
    "HL": ("관로", "천공"),
    "LP": ("관로", "연결관돌출"),
    "LS": ("관로", "연결관접합부"),
    "JS": ("관로", "이음부이탈"),
    "JF": ("관로", "이음부손상"),
    "JD": ("관로", "이음부단차"),
    "NS": ("관로", "역경사"),
    "SG": ("관로", "침하"),
    "DE": ("관로", "내피생성"),
    "DS": ("관로", "토사퇴적"),
    "DG": ("관로", "폐유부착"),
    "TO": ("관로", "임시장애물"),
    "RT": ("관로", "뿌리침입"),
    "IF": ("관로", "침입수"),
    "PB": ("관로", "막힘"),
    # 결함이 아니라 구조물·맥락 → 야장에서는 "상태"로 분류된다
    "PJ": ("상태", "이음부(접합부)존재"),
    "IN": ("상태", "관로내부"),
    "ETC": ("상태", "기타"),
    "OUT_MH": ("상태", "맨홀"),
    "OUT_INVERT": ("상태", "인버트"),
    "OUT_CAR": ("상태", "외부"),
}

DEFECT_GRADES = ("소", "중", "대")


def pipeasset_caption(label: str, grade: str, time_s: int) -> str:
    """야장 사진 캡션 한 줄. 예: "(관로)토사퇴적(중) [00:04:39]"

    매핑에 없는 라벨은 구분을 "관로"로 두고 라벨을 그대로 쓴다.
    """
    code = str(label or "").strip().upper()
    cat, name = DEFECT_PIPEASSET.get(code, ("관로", str(label or "").strip()))
    g = grade if grade in DEFECT_GRADES else "중"
    t = max(0, int(time_s or 0))
    return f"({cat}){name}({g}) [{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}]"


def defect_korean(label: str) -> str:
    """결함 라벨의 한글명을 반환. 모르는 라벨이면 빈 문자열.

    모델에 따라 라벨이 코드('BK')로도, 한글('파손')로도 올 수 있어 양쪽을 받는다.
    """
    raw = str(label or "").strip()
    if not raw:
        return ""
    ko = DEFECT_CODE_KO.get(raw.upper())
    if ko:
        return ko
    return raw if raw in _DEFECT_KO_TO_CODE else ""

# ───────── 프레임 추출 설정 ─────────
FRAME_INTERVAL = 2
EXTRACT_MAX_SIDE = 768
EXTRACT_JPEG_QUALITY = 80
USE_OCR = True

# 영상 앞 30초를 훑어 '조사시작' 자막을 찾는 기능. 영상당 OCR을 최대 15번 더 돌려
# 느린 데다, 못 찾으면 어차피 0초부터 보므로 기본으로 꺼둔다.
# 다시 켜려면 OCR_FIND_START=1 로 실행.
OCR_FIND_START = os.getenv("OCR_FIND_START", "0") == "1"

# ───────── 서버 설정 ─────────
HOST = "127.0.0.1"

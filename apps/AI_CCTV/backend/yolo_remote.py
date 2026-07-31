from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .config import YOLO_CONF, YOLO_IGNORE_CLASSES, YOLO_IMGSZ
from .overlay_filter import is_overlay_text_box
from .yolo_infer import _normalize_yolo_class_name, _yolo_display_label


def _image_size(fp: Path):
    """프레임 크기(w, h)를 픽셀 디코딩 없이 헤더에서만 읽는다. 실패 시 (0, 0)."""
    try:
        from PIL import Image
        with Image.open(fp) as im:
            return im.size  # (w, h)
    except Exception:
        return (0, 0)

REMOTE_TIMEOUT_S = 180
# 한 요청에 보내는 프레임 수. 전체(수백 장)를 한 번에 보내면 Colab 쪽에서
# 리스트 전체가 GPU 한 배치로 올라가 CUDA OOM(500 에러)이 난다.
REMOTE_CHUNK_SIZE = 24

# ngrok 무료 도메인은 이 헤더가 없으면 실제 응답 대신 "방문 확인" HTML 경고
# 페이지를 돌려준다. 그러면 resp.json()이 깨지므로 항상 붙여서 보낸다.
REMOTE_HEADERS = {"ngrok-skip-browser-warning": "true"}


def normalize_remote_url(raw: str) -> str:
    """사용자가 붙여넣은 주소를 정리한다(스킴 없으면 https, 끝 슬래시 제거)."""
    u = (raw or "").strip().rstrip("/")
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u


# ngrok이 터널 자체를 못 찾거나 뒤쪽 서버가 죽었을 때 돌려주는 에러코드.
# 이 경우 HTTP 404가 오지만 "우리 서버의 404"가 아니므로 구분해야 한다.
_NGROK_ERRORS = {
    "ERR_NGROK_3200": "ngrok 터널이 없습니다. Colab에서 셀을 다시 실행해 새 주소를 받으세요.",
    "ERR_NGROK_3004": "ngrok 터널이 오프라인입니다. Colab 세션이 끊겼는지 확인하세요.",
    "ERR_NGROK_6024": "ngrok 터널이 오프라인입니다. Colab 세션이 끊겼는지 확인하세요.",
    "ERR_NGROK_8012": "ngrok은 살아있지만 Colab 안의 서버가 응답하지 않습니다. 추론 서버 셀을 확인하세요.",
}


def probe_remote(remote_url: str) -> Tuple[bool, str]:
    """[연결] 버튼용. 서버가 실제로 응답하는지 확인하고 (성공여부, 메시지)를 준다.

    주소만 저장해두면 분석을 돌릴 때가 되어서야 실패를 알게 되므로,
    연결 시점에 미리 확인해서 원인을 알려준다.

    확인 순서는 `/predict` → `/health` → `/`. `/predict`는 POST 전용이라
    GET하면 405가 오는데, 그게 "이 주소에 추론 엔드포인트가 있다"는 가장 확실한 증거다.
    """
    base = remote_url.rstrip("/")

    for path in ("/predict", "/health", "/"):
        try:
            resp = requests.get(base + path, headers=REMOTE_HEADERS, timeout=10)
        except requests.exceptions.SSLError as e:
            return False, f"SSL 오류: {e}"
        except requests.exceptions.ConnectionError:
            return False, "서버에 연결할 수 없습니다. Colab 셀이 실행 중인지, 주소가 맞는지 확인하세요."
        except requests.exceptions.Timeout:
            return False, "응답 시간 초과(10초). Colab 세션이 살아있는지 확인하세요."
        except Exception as e:
            return False, f"연결 실패: {e}"

        ngrok_err = resp.headers.get("Ngrok-Error-Code", "")
        if ngrok_err:
            return False, _NGROK_ERRORS.get(ngrok_err, f"ngrok 오류({ngrok_err}). 주소를 다시 확인하세요.")

        if path == "/predict":
            # /predict의 응답은 결정적이다. 405 = POST 전용 라우트 있음(정상),
            # 404 = 서버는 살아있어도 추론 엔드포인트가 없는 엉뚱한 주소.
            if resp.status_code == 405:
                return True, "추론 엔드포인트(/predict) 확인됨"
            if resp.status_code == 404:
                return False, ("서버는 응답하지만 /predict 엔드포인트가 없습니다. "
                               "Colab 추론 서버 주소가 맞는지 확인하세요.")

        if resp.status_code == 404:
            continue  # /health 가 없을 뿐일 수 있다 → 다음 경로 확인
        if resp.status_code >= 500:
            return False, f"서버 오류 HTTP {resp.status_code}"
        if "text/html" in resp.headers.get("content-type", "") and "ngrok" in resp.text.lower():
            return False, "ngrok 경고 페이지가 반환됩니다. 주소가 올바른지 확인하세요."
        return True, f"연결 확인 (HTTP {resp.status_code})"

    return False, "서버 응답을 확인하지 못했습니다."


def _post_chunk(url: str, chunk: List[Path]) -> dict:
    opened = [open(fp, "rb") for fp in chunk]
    try:
        files = [
            ("files", (fp.name, fobj, "image/jpeg"))
            for fp, fobj in zip(chunk, opened)
        ]
        resp = requests.post(
            url,
            files=files,
            data={"conf": YOLO_CONF, "imgsz": YOLO_IMGSZ},
            headers=REMOTE_HEADERS,
            timeout=REMOTE_TIMEOUT_S,
        )
        if resp.status_code != 200:
            # 서버가 에러 내용을 JSON으로 보내주면 그걸 그대로 보여준다
            try:
                detail = resp.json().get("error", "")
            except Exception:
                detail = (resp.text or "")[:200]
            raise RuntimeError(f"HTTP {resp.status_code}: {detail or resp.reason}")
        try:
            return resp.json()
        except ValueError:
            # JSON이 아니면 대개 ngrok 경고/로그인 페이지가 온 것이다
            raise RuntimeError(
                f"서버가 JSON이 아닌 응답을 보냈습니다(content-type="
                f"{resp.headers.get('content-type', '?')}). 주소가 /predict 를 제공하는지 확인하세요."
            )
    finally:
        for fobj in opened:
            fobj.close()


def call_yolo_remote(frames: List[Path], remote_url: str) -> Tuple[list, Optional[str]]:
    """Colab 등에서 띄운 /predict 서버로 프레임들을 나눠 보내고 결과를 받아온다.
    반환 형태는 yolo_infer.call_yolo()와 동일하게 맞춰서, analysis.py에서
    로컬/원격 추론을 그대로 바꿔 쓸 수 있도록 한다.
    청크 하나가 실패해도 나머지 청크 결과는 살리고, 에러 메시지에 실패 내역을 담는다.
    """
    if not frames:
        return [], None

    url = remote_url.rstrip("/") + "/predict"
    by_name = {}
    errors = []

    for i in range(0, len(frames), REMOTE_CHUNK_SIZE):
        chunk = frames[i:i + REMOTE_CHUNK_SIZE]
        try:
            data = _post_chunk(url, chunk)
        except Exception as e:
            errors.append(f"프레임 {i + 1}~{i + len(chunk)}: {e}")
            continue
        for r in data.get("results", []):
            by_name[r.get("filename")] = r

    err = None
    if errors:
        err = f"Colab 서버 호출 일부 실패 ({len(errors)}개 청크): " + " | ".join(errors[:3])

    items = []
    for fp in frames:
        r = by_name.get(fp.name)
        if not r:
            continue

        img_w, img_h = _image_size(fp)
        defects = set()
        boxes_out = []
        for box in r.get("boxes", []):
            raw_name = str(box.get("class_name", ""))
            if _normalize_yolo_class_name(raw_name) in YOLO_IGNORE_CLASSES:
                continue
            # 화면 자막(거리/현장명/위경도 등) 위치의 오탐 제외
            if is_overlay_text_box(box.get("xyxy"), img_w, img_h):
                continue
            label = _yolo_display_label(raw_name)
            defects.add(label)
            boxes_out.append({**box, "label": label})

        if defects:
            items.append({
                "time_s": int(fp.stem),
                "defects": sorted(defects),
                "distance_text": "",
                "diameter_text": "",
                "pipe_id": "",
                "site_name": "",
                "direction": "",
                "model": "yolo-remote",
                "boxes": boxes_out,
            })

    return items, err

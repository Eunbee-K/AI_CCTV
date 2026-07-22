"""전역 결함 class id 매핑과, 이미 전역 id로 저장된 YOLO txt 라벨을 이번
실험의 로컬 class_id로 리매핑하는 유틸.

원본 XML(S20)/LabelMe JSON(S22) 라벨은 전부 preconvert 과정을 거쳐 전역 class id
YOLO txt로 변환 완료된 상태라(rename_data_s20_s22_bbox, aihub_data_bbox 전부),
더 이상 이 프로젝트에서 XML/JSON을 직접 파싱할 필요가 없다.
"""
from pathlib import Path
from typing import List

# docs/메타데이터(original 데이터셋)_최종.xlsx "클래스 코드" 시트의 전역 번호.
# aihub_data_bbox와 preconvert된 rename_data_s20_s22_bbox의 YOLO txt 라벨이
# 전부 이 번호를 그대로 쓰고 있어서, 실험별 로컬 class_id로 리매핑할 때 쓴다.
GLOBAL_CLASS_ID = {
    "CC": 0, "CL": 1, "CM": 2, "SD": 3, "BC": 4, "LD": 5, "DF": 6, "BK": 7,
    "CX": 8, "PO": 9, "HL": 10, "LP": 11, "LS": 12, "JS": 13, "JF": 14,
    "JD": 15, "NS": 16, "SG": 17, "DE": 18, "DS": 19, "DG": 20, "TO": 21,
    "RT": 22, "IF": 23, "PB": 24, "PJ": 25, "ETC": 26, "IN": 27,
    "OUT_MH": 28, "OUT_INVERT": 29, "OUT_CAR": 30,
}
GLOBAL_ID_TO_CODE = {v: k for k, v in GLOBAL_CLASS_ID.items()}


def remap_global_yolo_lines(txt_path: Path, class_to_id: dict) -> List[str]:
    """이미 전역 class id(GLOBAL_CLASS_ID)로 저장된 YOLO txt를 읽어, 이번
    실험에서 쓰는 클래스만 로컬 id로 바꿔서 반환한다.

    aihub_data_bbox 원본이나 rename_data_s20_s22_bbox의 (preconvert 완료된)
    .txt 라벨 모두 이 함수로 읽는다. 좌표는 이미 정규화되어 있으므로 그대로
    쓰고 맨 앞 class id만 바꾼다. 한 이미지 안에 여러 클래스가 섞여 있어도
    (전역 id 기준으로) 각각 올바르게 처리한다.
    """
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if not parts:
                continue
            code = GLOBAL_ID_TO_CODE.get(int(parts[0]))
            if code in class_to_id:
                lines.append(" ".join([str(class_to_id[code])] + parts[1:]))
    return lines

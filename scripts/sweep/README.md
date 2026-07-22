# 하이퍼파라미터 스윕 자동화

워크스테이션(오프라인 리눅스)에서 사람 개입 없이 여러 학습 조건을 순차 실행하고,
인터넷이 연결될 때마다 결과 요약표를 이메일로 자동 발송하는 스크립트 모음.

## 구성

| 파일 | 역할 |
|---|---|
| `label_formats.py` | 전역 결함 class id 매핑 + 전역 id YOLO txt를 실험별 로컬 class_id로 리매핑 |
| `collect_dataset.py` | 외장하드 원본에서 지정 클래스만 골라 클래스별 정확한 수량(quota)으로 임시 데이터셋 생성 (`scan` / `materialize`) |
| `run_sweep.py` | run마다 데이터셋 생성 → 학습 → 결과 기록 → 데이터셋 삭제를 반복 |
| `report.py` | `results.csv` 파싱, 누적 CSV/엑셀 요약표 생성 |
| `mailer.py` | 모든 run이 끝난 뒤부터 인터넷 연결을 주기적으로 확인, 연결되면 요약표 이메일 발송 |
| `common.py` | 설정/상태 파일, 로깅 공용 유틸 |

## 원본 데이터 구조 (중요)

`E:/AI_CCTV_DATASET/original` 아래에 성격이 다른 두 소스가 섞여 있다. 원본이
XML(S20)/LabelMe JSON(S22)이었던 서울시 데이터는 전량 전역 class id YOLO txt로
변환(preconvert) + 원본 삭제까지 끝난 상태라, 지금은 둘 다 순수 YOLO txt 소스다.

- **`rename_data_s20_s22_bbox`** (서울시 2020/2022): `<결함폴더>/images/`, `<결함폴더>/labels/`
  구조. (`type: seoul_txt`)
- **`aihub_data_bbox`** (AI Hub): `image/<코드>/`, `labels/<코드>/` 구조. 라벨이
  이미 YOLO txt이고, 클래스 번호도 이미 `docs/메타데이터(original 데이터셋)_최종.xlsx`
  "클래스 코드" 시트의 전역 번호를 그대로 쓰고 있다 (`label_formats.GLOBAL_CLASS_ID`).
  폴더당 클래스가 섞이지 않아 상대적으로 깨끗하다. (`type: yolo_txt`)
- **`aihub_data_seg`** (AI Hub, 세그멘테이션): 라벨은 전역 class id YOLO seg
  폴리곤 txt(`labels/<코드>/`)인데 이미지 폴더명이
  `image/1-3.파손(Broken-Pipe,BK)/`처럼 코드와 규칙적으로 대응하지 않아서,
  이미지/라벨 폴더를 직접 지정하는 `type: dir_pair`
  (`{type: dir_pair, images: "...", labels: "..."}`)로 쓴다. 라벨 리매핑은
  class id 뒤 토큰(폴리곤 좌표)을 그대로 보존하므로 bbox와 같은 코드로 처리된다.
  seg 학습은 `model`만 `yolo11l-seg.pt`로 바꾸면 된다 (`test4_seg.example.yaml` 참고).
  서울시 seg 원본(`rename_data_s20_s22_seg`)은 아직 XML/JSON 미변환이라 스윕에서
  사용하지 않는다.

`docs/메타데이터(총괄).xlsx`의 "클래스 코드" 시트가 전체 31개 코드(구조결함18 +
운영결함7[PB는 데이터 0장이라 사실상 30개] + 문맥용 PJ/ETC/IN/OUT_*)의 기준이고,
"test3" 같은 시트가 실제로 어떤 실험에 클래스별로 몇 장씩(train/val/test) 쓸지
정해둔 스펙이다. 스윕 설정(`configs/sweeps/*.yaml`)은 이 스펙을 그대로 옮긴 것.

**IN/PJ는 "정상(배경)" 클래스다.** AI Hub 원본 라벨엔 자기 자신을 가리키는 bbox가
있지만 실제로는 결함이 아니라 "정상 상태"를 나타내는 문맥용 이미지라서, 원본
라벨은 `aihub_data_bbox/labels_original_backup/<코드>/`로 옮겨 보존하고 실제
학습에 쓰이는 `labels/<코드>/`의 `.txt`는 빈 파일로 이미 교체해 두었다. 스윕
설정에서 이 클래스는 `background: true`를 켜야 `collect_dataset.py materialize`가
박스 없는 이미지를 스킵하지 않고 빈 라벨(배경/네거티브 샘플)째로 포함시킨다.

```yaml
IN:
  background: true
  quota: {train: 12000, val: 1500, test: 1500}
  sources:
    - {type: yolo_txt, path: "aihub_data_bbox", code: IN}
```

## 사전 준비 (워크스테이션, 최초 1회)

인터넷이 연결되어 있을 때 미리 해둬야 하는 것들 (오프라인 상태에서는 아래가 안 됨):

1. `pip install ultralytics pyyaml openpyxl` (ultralytics는 이미 설치돼 있을 것)
2. `yolo11l.pt` 등 학습에 쓸 사전학습 가중치를 워크스테이션 로컬 경로에 미리 복사
   (`E:/AI_CCTV_RESULTS/pretrained_models/yolo11l.pt` 같은 걸 USB 등으로 옮겨두기)
3. 외장하드를 워크스테이션에 마운트하고 `rename_data_s20_s22_bbox`와
   `aihub_data_bbox`가 같이 있는 상위 경로(`external_root`) 확인

## 1단계 — 설정 파일 만들기

`test3_18class.example.yaml`(bbox 18클래스) / `test4_seg.example.yaml`(seg 8클래스+배경)
을 복사해서 쓴다.

```bash
cp configs/sweeps/test3_18class.example.yaml configs/sweeps/test3_18class.yaml
cp configs/sweeps/test4_seg.example.yaml configs/sweeps/test4_seg.yaml
cp scripts/sweep/mail_config.example.yaml scripts/sweep/mail_config.yaml
```

`test3_18class.yaml`에서 최소한 아래는 실제 환경에 맞게 고칠 것:

- `dataset.external_root` — 외장하드 실제 마운트 경로 (`rename_data_s20_s22_bbox`,
  `aihub_data_bbox`의 공통 상위 폴더)
- `workspace.dataset_root` / `results_root` / `summary_csv` — 워크스테이션 로컬 디스크 경로
- `model` — 로컬에 미리 받아둔 가중치 경로
- `runs` — 돌리고 싶은 imgsz/epochs/batch 조합

`mail_config.yaml`에는 Gmail 앱 비밀번호를 넣는다 (일반 로그인 비밀번호 아님).
Google 계정 → 보안 → 2단계 인증 → 앱 비밀번호에서 발급.
**이 파일은 `.gitignore`에 등록되어 있어 git에 올라가지 않는다.**

## 2단계 — 라벨 매핑 검증 (`scan`, 반드시 먼저 실행)

```bash
cd scripts/sweep
python collect_dataset.py scan --config ../../configs/sweeps/test3_18class.yaml
```

**폴더 기준 사용 가능 수량**이 quota(train+val+test 요청량)를 채울 만큼 되는지
클래스마다 확인해준다.

## 3단계 — 스윕 실행 (백그라운드, 계속 진행)

SSH 접속이 끊겨도 계속 돌도록 `nohup` 또는 `tmux`로 실행한다.
`--config`에 여러 개를 주면 앞 스윕이 전부 끝난 뒤 다음 스윕이 자동으로 이어진다.

```bash
cd scripts/sweep
# test3(bbox) 끝나면 test4_seg(세그멘테이션)가 자동으로 이어서 실행됨
nohup python run_sweep.py \
  --config ../../configs/sweeps/test3_18class.yaml ../../configs/sweeps/test4_seg.yaml \
  > /dev/null 2>&1 &
```

여러 스윕을 이어 돌릴 때 mailer는 **마지막 config**를 바라보게 하면 된다 —
두 config의 `summary_csv`가 같은 경로면 요약표 하나에 두 실험이 함께 쌓여서,
마지막 스윕이 끝난 뒤 메일 한 통으로 전체 결과가 온다.

- 진행 로그: `<results_root의 부모 경로>/<sweep_name>.log`
- 진행 상태: `<results_root의 부모 경로>/<sweep_name>.state.json` (run별 pending/running/done/failed)
- run 하나가 실패해도 스윕 전체가 멈추지 않고 다음 run으로 넘어간다.
- 워크스테이션이 재부팅되거나 프로세스가 죽었다 다시 실행해도, 이미 `done`인
  run은 건너뛰고 이어서 진행한다 (`run_sweep.py`를 그대로 다시 실행하면 됨).
- run마다 임시 데이터셋(`workspace.dataset_root/<run_name>`)은 학습이 끝나면
  성공/실패 상관없이 자동 삭제된다. test3 규모(train만 약 13만 4천장)는 디스크를
  꽤 쓰니 `workspace.dataset_root`가 여유 있는 디스크를 가리키는지 확인할 것.
- 데이터셋은 삭제돼도, 그 run에 실제로 어떤 원본 파일을 썼는지는
  `<results_root>/<sweep_name>/<run_name>/used_files.csv`에 영구히 남는다
  (컬럼: `split, class, source_kind, source_path, dest_filename` — 외장하드
  원본 경로 기준). `materialize_manifest.json`(장수/박스 수 집계)도 같은
  폴더에 함께 복사되어 남는다.

## 4단계 — 결과 이메일 발송 (인터넷 연결될 때만)

`run_sweep.py`와 동시에(또는 그 전에) 띄워둬도 된다 — 모든 run이 끝나기 전까지는
인터넷 연결을 확인하지 않고 그냥 대기만 하다가, 스윕 전체가 끝난 뒤부터 주기적으로
확인을 시작한다.

```bash
cd scripts/sweep
nohup python mailer.py \
  --config ../../configs/sweeps/test3_18class.yaml \
  --mail-config mail_config.yaml > /dev/null 2>&1 &
```

- 모든 run 완료(`SWEEP_DONE.flag` 생성) 전까지는 인터넷 확인 없이 대기.
- 완료된 뒤부터 3시간(기본값, `mail_config.yaml`의 `check_interval_sec`)마다 인터넷
  연결 여부를 확인하고, 연결돼 있으면 전체 요약표(`sweep_summary.csv` → 가능하면
  `.xlsx`)를 `mail_config.yaml`의 `recipient_email`로 보낸다.
- `--interval <초>`로 확인 주기를 CLI에서 덮어쓸 수 있다.
- 지금 막 워크스테이션을 인터넷에 연결했고 바로 결과를 받고 싶다면 (스윕이 아직
  안 끝났어도 대기 없이 즉시 한 번 확인):
  ```bash
  python mailer.py --config ... --mail-config mail_config.yaml --once
  ```
- 발송 로그: `<sweep_name>.mailer.log`, 마지막 발송 시점: `<sweep_name>.mail_state.json`

## 다음 테스트 오더를 줄 때 (형식)

`docs/메타데이터(총괄).xlsx`에 새 시트(예: `test4`)로 클래스별 수량을 먼저 정리한
뒤, 아래 형식으로 알려주면 `configs/sweeps/*.yaml`에 그대로 반영한다.

```
실험명: test4
데이터 형식: bbox
대상 클래스 및 목표 수량 (train / val / test):
- BK: 12000 / 1500 / 1500
- ...
(선택) 하이퍼파라미터 스윕: imgsz/epochs/batch 조합
(선택) test3 대비 달라지는 점만 말해도 됨 (예: "test3에 CX, BC만 추가해줘")
```

클래스 코드가 `rename_data_s20_s22_bbox`/`aihub_data_bbox` 중 어디에 있는지는
`docs/메타데이터(original 데이터셋)_최종.xlsx`의 "상세 현황" 시트에서 확인 가능
(코드가 처음 등장하는 것이면 `collect_dataset.py scan`으로 라벨 매핑부터 검증할 것).

## 참고 / 알아두면 좋은 점

- 스윕에서 바꾸는 게 하이퍼파라미터뿐이고 클래스 구성은 고정이라면, run마다 매번
  데이터셋을 새로 복사하는 대신 한 번만 만들어 재사용하도록 바꿀 수도 있다
  (디스크 여유가 생기면 `run_sweep.py`의 materialize 호출을 루프 바깥으로 빼면 됨).
  지금은 디스크 용량 문제 때문에 run마다 복사/삭제하도록 되어 있다.
- `experiments/detect_v2_bk_ds/metadata.yaml` 같은 기존 실험 기록 형식에 맞추려면,
  스윕이 끝난 뒤 `sweep_summary.csv`를 보고 대표 run들을 골라 수기로 정리하면 된다
  (자동 생성은 하지 않음 — 기존 리포지토리 관례상 실험 요약은 사람이 검토 후 기록).

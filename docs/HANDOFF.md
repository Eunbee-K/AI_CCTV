# 다른 PC에서 이어서 하기

**마지막 갱신** 2026-08-12 · **브랜치** `feat/stage1-classifier` (main 미병합)

외장하드만 들고 옮겨 다니는 것을 전제로 정리한 문서다. 이 드라이브 밖(예전 PC의
`C:\Users\...`)에 있던 것은 전부 드라이브 안으로 옮겨뒀다.

---

## 0. 먼저 확인

**드라이브 문자가 바뀐다.** 원래 PC에서는 `E:`였다. 새 PC에 꽂고 실제 문자를 먼저 본 뒤,
아래 경로의 `E:`를 그 문자로 바꿔 읽으면 된다. **코드는 안 고쳐도 된다** — 스크립트는
`scripts/paths.py`가, 앱은 `backend/config.py`가 저장소 위치에서 드라이브를 되짚는다.

**저장소가 드라이브에 두 벌 있다. 헷갈리지 말 것.**

| 경로 | 상태 |
|---|---|
| `<드라이브>:\AI_CCTV` | **← 이게 작업본.** `feat/stage1-classifier`, 최신 |
| `<드라이브>:\AI_CCTV_PROJECT` | 옛 사본. `main`의 오래된 커밋. 쓰지 말 것 |

```
cd /d E:\AI_CCTV
git status          # feat/stage1-classifier · clean 이어야 한다
git log --oneline -5
```

---

## 1. 지금 앱이 어떻게 도는가

```
프레임 1초 간격 추출
        ↓
   Stage-1 필터  ──→  결함 있는 구간을 고른다 (표의 행이 된다)
        ↓
      YOLO       ──→  그 구간에 이름 + 박스를 붙인다
        ↓
      LLM(선택)  ──→  같은 구간에 이름을 덧붙인다. 박스는 없다
```

**필터가 결함을 찾고, YOLO/LLM은 이름만 붙인다.** 이게 2026-08-12에 잡은 구조다
(`FILTER_MODE=lead`).

| 자리 | 파일 | 실체 |
|---|---|---|
| 필터 | `apps/AI_CCTV/assets/filter.onnx` | OLD_v3 (EfficientNet-B0) |
| YOLO | `apps/AI_CCTV/assets/best.pt` | **test5** 17종 · mAP50 0.698 |
| 예전 YOLO | `apps/AI_CCTV/assets/best_test6.pt` | test6 12종 · 되돌릴 때 씀 |
| LLM | `backend/llm_infer.py` | Gemini + GPT · **기본 꺼짐** |

`.pt`/`.onnx`는 `.gitignore` 대상이라 **git에 없다.** 드라이브 안에는 있으니 그대로 쓰면 된다.
새로 받는 경우엔 `AI_CCTV_RESULTS`에서 다시 복사해야 한다.

### 신설 / 노후 구분

**지금은 갈라지지 않는다.** 둘 다 같은 필터·같은 YOLO·같은 설정으로 돈다.
`state.pipe_condition`은 "분석 전에 뭐라도 골랐는지"만 확인하고 그 값으로 분기하는
코드는 한 군데도 없다. 신설 전용 필터(NEW_v1, AUC 0.7466)는 미달이라 배포하지 않았다.

---

## 2. 켜고 끄는 것들 — 환경변수

전부 기본값으로 두면 위 구조대로 돈다. 바꿀 때만 쓴다.

| 변수 | 기본 | 뜻 |
|---|---|---|
| `FILTER_MODE` | `lead` | `off` / `lead` / `series`(버림) / `parallel`(표시만) |
| `FILTER_TOP_RATIO` | `0.20` | 상위 몇 %를 결함 구간으로 볼지. 표 길이를 정한다 |
| `FRAME_INTERVAL` | `1` | 프레임 간격(초). 2로 올리면 빨라지지만 결함을 건너뛴다 |
| `YOLO_MODEL_PATH` | `assets/best.pt` | test6으로 되돌리려면 `assets/best_test6.pt` |
| `YOLO_CONF` | `0.10` | 표가 지저분하면 올린다 |
| `LLM_ENABLED` | (꺼짐) | `1`이면 켠 채로 시작 |
| `LLM_EXAMPLES_DIR` | `<드라이브>:\CCTV\ver.2.0_학습데이터` | LLM에 같이 보낼 예시 사진 |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | — | **새 PC에서 다시 넣어야 한다** (아래 참고) |

### API 키

키는 **디스크에 저장하지 않는다.** 앱 화면(`LLM 판독` → `사용`)에서 넣으면 서버 메모리에만
있고 앱을 끄면 사라진다. 매번 넣기 싫으면 환경변수로 박아두면 된다.

원래 PC에는 환경변수로 들어 있었다. 새 PC에는 없다. 키 원본은
`<드라이브>:\CCTV\api_keys.jsonl`과 `api.txt`에 있다(ver_2.2가 저장해둔 것).

---

## 3. 드라이브 어디에 무엇이 있는가

```
<드라이브>:\
├─ AI_CCTV\                       ← 작업 저장소 (git)
│   ├─ apps\AI_CCTV\assets\       ← best.pt · filter.onnx (git 미포함)
│   └─ .handoff\claude-history\   ← Claude 대화기록 + 기억 (아래 5번)
├─ AI_CCTV_DATASET\
│   ├─ video\상면 CCTV\           ← **8/12에 C: 바탕화면에서 옮김.** 현장 검증용
│   │   ├─ 20260128...\           ← SM1·SM2 관로별 폴더 (영상 + 한글 조사표)
│   │   └─ 20260131...\
│   ├─ video\JB1-2-0413 27M.mp4   ← 노후 검증 영상 (결함 2건, 둘 다 변형)
│   ├─ fieldset_v2_defects\       ← 한글 조사표에서 뽑은 결함 거리 (defects.csv)
│   ├─ clsdata_old_v3\            ← 필터 학습 데이터 (배포본 OLD_v3)
│   ├─ clsdata_new_v1\            ← 신설용 시도 (보류)
│   └─ filter_eval_SM2*.csv       ← 실측 캐시. 다시 재려면 이걸 쓴다
├─ AI_CCTV_RESULTS\
│   ├─ filter\runs\               ← 필터 학습 결과 (OLD_v1~v3, NEW_v1)
│   ├─ test5_17class_sweep01\     ← 지금 배포된 YOLO
│   └─ test6_12class_sweep01\     ← 이전 YOLO
└─ CCTV\                          ← ver_2.2 옛 앱
    ├─ ver_2.2_AICCTV\            ← LLM 구조 원본 (여기서 뜯어왔다)
    ├─ ver.2.0_학습데이터\         ← LLM 예시 사진 236장 + labels.jsonl
    └─ api_keys.jsonl             ← API 키 원본
```

---

## 4. 새 PC에서 처음 할 일

```bash
cd /d E:\AI_CCTV\apps\AI_CCTV
pip install -r requirements.txt          # LLM 안 쓰면 마지막 두 줄은 빼도 된다

# 로컬 CPU로 띄우기 (Colab 없이)
python serve_web_cpu.py --port 8000       # 브라우저 → 로그인 admin/admin1234
```

돌아가는지 빠르게 확인하려면:

```bash
python -c "import sys; sys.path.insert(0,'.'); from backend import config as c, defect_filter; \
print(c.YOLO_MODEL_PATH.exists(), defect_filter.availability(), c.FILTER_MODE, c.FRAME_INTERVAL)"
# True (True, '') lead 1  가 나와야 한다
```

---

## 5. 여기까지 온 맥락 (Claude 대화기록)

`<드라이브>:\AI_CCTV\.handoff\claude-history\` 에 통째로 복사해뒀다.

새 PC에서 이어서 대화하려면 그 폴더를 새 PC의
`C:\Users\<사용자>\.claude\projects\<드라이브문자>--AI-CCTV\` 로 복사하면 된다.
폴더 이름은 작업 경로에서 만들어지므로, 드라이브가 `F:`면 `f--AI-CCTV`가 된다.

| 파일 | 내용 |
|---|---|
| `4c4f6a02-….jsonl` | **8/10~8/12 세션.** 필터 재구축 → 실영상 검증 → lead 구조 → LLM |
| `9422a5b1-….jsonl` | 8/9~8/10 세션 |
| `a43c47a8-….jsonl` | 8/8 세션 |
| `memory\` | 세션 간 기억 (작업 3줄기 구분, UI 검증 방식 등) |

문서로 먼저 볼 것: [`docs/reports/2026-08-12-Stage1-필터-재구축.md`](reports/2026-08-12-Stage1-필터-재구축.md)

---

## 6. 이어서 할 것

**바로 다음 (막힌 것부터)**

1. **필터 실영상 성능을 제대로 재야 한다.** ~~SM2 실영상 AUC 0.5205~~ 는
   **2026-08-15에 폐기했다** — 검증셋을 잘못 만들었다. 정답을 프레임 단위로 찍지
   않고 조사표(야장) 거리 기록에서 기계적으로 펼쳐서, 정답 "결함" 구간이 전체의
   24.6%(한 관로는 100%)까지 부풀었다. **0.52는 필터가 아니라 그 라벨을 잰 값이다.**
   → 연속 영상 몇 개에 **사람이 프레임 단위로 정답을 찍어** 다시 잰다. 재학습은
   그 결과를 보고 정한다. 현재 근거는 야장 val(0.9503)뿐이고 실영상은 미측정이다.

2. **변형(DF)을 아무도 못 잡는다.** YOLO test5에 DEFORM 클래스가 있는데 실영상에서
   한 번도 발화하지 않았다(conf 0.02까지 내려도). 필터도 이 유형이 제일 약하다.
   JB1-2-0413의 결함 2건이 둘 다 변형인데 이름이 `이음부-단차`로 붙는다.

3. **거리 기준 프레임 추출.** 지금은 1초 간격이라 카메라가 멈춘 구간에서 같은 장면이
   47% 중복된다. 거리(예: 0.2m)마다 한 장씩 뽑으면 중복이 사라지고 사각지대도 준다.

**중기**

4. **신설 전용 경로.** 필터·YOLO·상위 컷 비율 세 군데를 `state.pipe_condition`으로
   가르면 된다. 지금은 근거가 없어 공용이다. 신설 현장 조사표가 확보되면 시작.
5. **v6 YOLO** — fieldset 신설 데이터로 파인튜닝 (원래 8/10 예정, 지연 중)
6. `feat/stage1-classifier` → `main` 병합 여부 결정

**검증용으로 쓸 것**

| 영상 | 관로 | 정답 |
|---|---|---|
| `AI_CCTV_DATASET/video/상면 CCTV/20260128...` | SM2 8개 (노후) | 한글 조사표 결함 15건 |
| `AI_CCTV_DATASET/video/상면 CCTV/20260128...` | SM1 (신설·재시공) | 폴더별 한글 조사표 |
| `AI_CCTV_DATASET/video/JB1-2-0413 27M.mp4` | 노후 | 1:47(10.1m)·2:31(13.5m) 변형 |

**실측할 때 반드시 무작위 대조군을 같이 볼 것.** 8/11에 대조군 없이 "결함 유실 0건"을
보고 필터가 잘 되는 줄 알았는데, 무작위로 골라도 0.4건이었다.
`scripts/classifier/eval_series_vs_parallel.py`가 이제 대조군과 AUC를 같이 찍는다.

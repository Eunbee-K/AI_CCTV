"""워크스테이션이 인터넷에 연결되는 순간을 기다렸다가, 새로 끝난 스윕 결과를
CSV/엑셀 요약표로 이메일 발송한다.

동작 순서:
1. run_sweep.py의 모든 run이 끝날 때까지는(SWEEP_DONE.flag 생성 전) 인터넷
   확인 자체를 하지 않고 조용히 대기만 한다 (학습 도중에는 아무 일도 안 함).
2. 모든 run이 끝나면 그때부터 주기적으로(기본 3시간마다) 인터넷 연결을
   확인하고, 연결되고 + 새로 완료된 run이 있으면 메일을 보낸다.

인터넷이 막 연결됐을 때 위 대기를 기다리지 않고 바로 확인하고 싶으면
--once 로 1회만 실행한다 (이 경우 학습 완료 여부와 무관하게 즉시 확인).
"""
import argparse
import csv
import json
import smtplib
import socket
import ssl
import time
from email.message import EmailMessage
from pathlib import Path

from common import atomic_write_json, load_yaml, now_iso, setup_logger
import report


def internet_available(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def count_result_rows(summary_csv_path: Path) -> int:
    if not summary_csv_path.exists():
        return 0
    with open(summary_csv_path, "r", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # 헤더 제외


def get_new_run_names(summary_csv_path: Path, since_count: int) -> list:
    """summary_csv에서 이전 발송 시점(since_count번째 행) 이후 새로 추가된 run_name들."""
    with open(summary_csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r["run_name"] for r in rows[since_count:]]


def find_best_weights(results_root: Path, sweep_name: str, run_names: list, max_mb: float, log) -> list:
    """새로 완료된 run들의 best.pt 경로를 모은다.

    Gmail 등 SMTP 서버의 첨부 용량 제한(보통 총 25MB) 대응으로, 파일이
    max_mb를 넘으면 첨부에서 빼고 로그로만 경고한다 (표 발송 자체는 막지 않음).
    """
    found = []
    for run_name in run_names:
        best_pt = results_root / sweep_name / run_name / "weights" / "best.pt"
        if not best_pt.exists():
            log.warning(f"[{run_name}] best.pt 없음 -> 첨부 스킵")
            continue
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            log.warning(
                f"[{run_name}] best.pt {size_mb:.1f}MB > 제한 {max_mb}MB -> 첨부 스킵 "
                f"(결과 폴더에서 직접 확인: {best_pt})"
            )
            continue
        found.append((run_name, best_pt))
    return found


def load_mail_state(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_sent_count": 0, "last_sent_at": None}


def send_report(mail_cfg: dict, sweep_name: str, attachment_path: Path, row_count: int, weight_paths: list = None) -> None:
    weight_paths = weight_paths or []

    msg = EmailMessage()
    msg["Subject"] = f"[AI_CCTV] {sweep_name} 스윕 결과 ({row_count}건 완료)"
    msg["From"] = mail_cfg["sender_email"]
    msg["To"] = mail_cfg["recipient_email"]

    if weight_paths:
        weight_note = "첨부된 가중치: " + ", ".join(f"{name}_best.pt" for name, _ in weight_paths)
    else:
        weight_note = "이번에 첨부된 가중치 없음 (용량 제한에 걸렸거나 새로 완료된 run이 없음 — 로그 참고)"
    msg.set_content(
        f"{sweep_name} 스윕 자동 실행 결과입니다.\n"
        f"현재까지 완료된 run: {row_count}건\n"
        f"전송 시각: {now_iso()}\n"
        f"{weight_note}\n\n"
        f"첨부된 표에서 imgsz/epochs/batch별 mAP50, mAP50-95 등을 확인하세요."
    )

    data = attachment_path.read_bytes()
    if attachment_path.suffix.lower() == ".xlsx":
        maintype, subtype = "application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        maintype, subtype = "text", "csv"
    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=attachment_path.name)

    for run_name, weight_path in weight_paths:
        msg.add_attachment(
            weight_path.read_bytes(),
            maintype="application",
            subtype="octet-stream",
            filename=f"{run_name}_best.pt",
        )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(mail_cfg["smtp_host"], mail_cfg["smtp_port"], context=context) as server:
        server.login(mail_cfg["sender_email"], mail_cfg["sender_app_password"])
        server.send_message(msg)


def wait_for_sweep_done(flag_path: Path, log, poll_interval: int = 60) -> None:
    """모든 run이 끝나기 전까지는 인터넷 확인 없이 조용히 대기한다.

    학습이 며칠씩 걸릴 수 있는데 그동안 메일러가 계속 인터넷을 확인하며
    "실행"되고 있을 필요가 없어서, run_sweep.py가 스윕 전체를 마치고 남기는
    SWEEP_DONE.flag가 생기기 전까지는 로컬 파일 존재 여부만 가볍게 확인한다.
    """
    if flag_path.exists():
        return
    log.info(f"아직 학습 진행 중 ({flag_path} 없음) -> 모든 run이 끝날 때까지 대기 (인터넷 확인 안 함)")
    while not flag_path.exists():
        time.sleep(poll_interval)
    log.info("모든 run 완료 확인됨 -> 지금부터 인터넷 연결을 주기적으로 확인합니다")


def try_send_if_ready(cfg: dict, mail_cfg: dict, mail_state_path: Path, log) -> bool:
    """조건이 맞으면 메일을 보내고 True, 아니면 False."""
    if not internet_available(mail_cfg["smtp_host"], mail_cfg["smtp_port"]):
        log.info("인터넷 연결 안 됨 -> 대기")
        return False

    summary_csv = Path(cfg["workspace"]["summary_csv"])
    row_count = count_result_rows(summary_csv)
    if row_count <= 0:
        log.info("아직 완료된 run 없음 -> 대기")
        return False

    mail_state = load_mail_state(mail_state_path)
    last_sent_count = mail_state.get("last_sent_count", 0)
    if row_count <= last_sent_count:
        log.info(f"새로 보낼 결과 없음 (이미 {last_sent_count}건 전송함) -> 대기")
        return False

    xlsx_path = summary_csv.with_suffix(".xlsx")
    attachment = report.build_excel_summary(summary_csv, xlsx_path)

    results_root = Path(cfg["workspace"]["results_root"])
    new_run_names = get_new_run_names(summary_csv, last_sent_count)
    max_mb = mail_cfg.get("max_weight_attachment_mb", 20)
    weight_paths = find_best_weights(results_root, cfg["sweep_name"], new_run_names, max_mb, log)

    log.info(f"인터넷 연결 확인됨, 새 결과 {row_count}건 발견 (가중치 {len(weight_paths)}개 첨부 대상) -> 메일 발송 시도")
    send_report(mail_cfg, cfg["sweep_name"], Path(attachment), row_count, weight_paths)

    mail_state["last_sent_count"] = row_count
    mail_state["last_sent_at"] = now_iso()
    atomic_write_json(mail_state_path, mail_state)
    log.info("메일 발송 완료")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="스윕 설정 YAML 경로")
    ap.add_argument("--mail-config", required=True, help="SMTP/수신자 설정 YAML 경로 (mail_config.yaml)")
    ap.add_argument("--interval", type=int, default=None, help="연결 확인 주기(초). 생략하면 mail_config.yaml의 check_interval_sec(기본 3시간) 사용")
    ap.add_argument("--once", action="store_true", help="학습 완료 여부와 무관하게 1회만 즉시 확인 (지금 막 인터넷 연결했을 때 수동 실행용)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    mail_cfg = load_yaml(args.mail_config)

    results_root = Path(cfg["workspace"]["results_root"])
    log = setup_logger(f"mailer-{cfg['sweep_name']}", results_root.parent / f"{cfg['sweep_name']}.mailer.log")
    mail_state_path = results_root.parent / f"{cfg['sweep_name']}.mail_state.json"
    sweep_done_flag = results_root / cfg["sweep_name"] / "SWEEP_DONE.flag"

    interval = args.interval if args.interval is not None else mail_cfg.get("check_interval_sec", 10800)

    if args.once:
        try_send_if_ready(cfg, mail_cfg, mail_state_path, log)
        return

    wait_for_sweep_done(sweep_done_flag, log)

    log.info(f"메일러 감시 시작 (주기 {interval}초). 워크스테이션이 오프라인이어도 계속 떠 있어도 됩니다.")
    while True:
        try:
            try_send_if_ready(cfg, mail_cfg, mail_state_path, log)
        except Exception as e:
            log.error(f"메일 발송 시도 중 오류: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()

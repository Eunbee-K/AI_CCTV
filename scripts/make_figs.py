"""발표용 시각자료를 만든다. matplotlib으로 그려 PNG로 저장.

참고 PPT(260203_처장님.pptx)의 색을 그대로 쓴다 — 같은 팀 자료끼리 톤을 맞춘다.
"""
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 한글 폰트 — 참고 PPT와 같은 맑은 고딕
for cand in ("Malgun Gothic", "맑은 고딕", "NanumGothic"):
    if any(cand == f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#017BB1"       # 참고 PPT 강조색
NAVY = "#3057B9"
GRAY = "#5A5A5A"
LIGHT = "#EAEAEA"
ORANGE = "#E8833A"
GREEN = "#2E9E5B"


def fig_architecture(path):
    """판독 3중 구조 — 프레임에서 셋으로 갈라졌다가 합집합으로."""
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, title, lines, color, tcolor="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=color, ec="none"))
        # 제목은 위, 설명은 아래쪽에 고르게 — 줄 수가 달라도 상자 밖으로
        # 나가지 않게 남은 높이를 나눠 쓴다.
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top",
                fontsize=12.5, fontweight="bold", color=tcolor)
        if lines:
            top, bot = y + h - 8.0, y + 2.4
            step = (top - bot) / max(len(lines), 1)
            for i, ln in enumerate(lines):
                ax.text(x + w / 2, top - i * step, ln, ha="center", va="top",
                        fontsize=9.5, color=tcolor)

    def arrow(x1, y1, x2, y2, style="-", color=GRAY, lw=2.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=lw, color=color,
                                     linestyle=style, shrinkA=0, shrinkB=0))

    # 입력
    box(35, 44, 30, 7, "영상 → 1초 간격 프레임", [], NAVY)

    # 세 판독기
    box(2, 20, 30, 19, "Classification (CLS)",
        ["EfficientNet-B0 · 24종", "결함 구간 선별", "+ 1차 이름"], BLUE)
    box(35, 20, 30, 19, "Object Detection (DET)",
        ["YOLO11 · 16종", "위치(박스)", "+ 2차 이름"], BLUE)
    box(68, 20, 30, 19, "Multimodal LLM",
        ["GPT / Gemini · 8종", "이름 보강", "(선택 · 기본 꺼짐)"], "#8FB8CC")

    arrow(45, 44, 20, 39.5)
    arrow(50, 44, 50, 39.5)
    arrow(55, 44, 80, 39.5, style="--", color="#9AA5AB")
    ax.text(70, 41.5, "선택", fontsize=9, color="#9AA5AB")

    # 합집합
    box(28, 10, 44, 8, "합집합 — 셋 중 하나라도 찾으면 결함 후보", [], GREEN)
    arrow(17, 20, 40, 18.5)
    arrow(50, 20, 50, 18.5)
    arrow(83, 20, 60, 18.5, style="--", color="#9AA5AB")

    box(28, 0.5, 44, 7, "감독자 확인 → 조사표 · 야장 PDF", [], "#4A4A4A")
    arrow(50, 10, 50, 7.8)

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", path)


def fig_performance(path):
    """합집합 효과 — DET 61.7 / CLS 40.8 / 합침 90.3."""
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
    names = ["DET 단독\n(YOLO)", "CLS 단독\n(분류기)", "둘을 합치면"]
    vals = [61.7, 40.8, 90.3]
    colors = [BLUE, "#7FB3D5", GREEN]
    bars = ax.bar(names, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%",
                ha="center", fontsize=15, fontweight="bold",
                color=GREEN if v > 80 else "#333333")
    ax.set_ylim(0, 105)
    ax.set_ylabel("결함 이름 정확도 (%)", fontsize=11)
    ax.set_title("같은 사진으로 잰 두 판독기 — 겹치는 것은 12%뿐",
                 fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    fig.text(0.5, -0.02,
             "두 모델 학습분을 모두 제외한 공통 검증셋 · 결함 12종 360장",
             ha="center", fontsize=9, color=GRAY)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", path)


def fig_outside(path):
    """관 밖 오탐 — 이전(v3) 39% → 지금 0%."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=200)
    names = ["이전 버전\n(v3 필터)", "현재\n(CLS)"]
    vals = [39, 0]
    bars = ax.bar(names, vals, color=["#C0392B", GREEN], width=0.45)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%",
                ha="center", fontsize=17, fontweight="bold",
                color="#C0392B" if v else GREEN)
    ax.set_ylim(0, 48)
    ax.set_ylabel("관 밖(맨홀·지상)을 결함으로 오판한 비율 (%)", fontsize=10)
    ax.set_title("학습에 '관 밖 정상'을 넣어 오탐을 없앴다",
                 fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", path)


def fig_userflow(path):
    """유저플로우 5단계 — 시연 영상과 같은 순서."""
    fig, ax = plt.subplots(figsize=(12.5, 3.0), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
    steps = [("①", "영상 올리기", "드래그앤드롭"),
             ("②", "분석 실행", "진행 로그 실시간"),
             ("③", "결과 확인", "결함 목록 + 박스"),
             ("④", "검수·수정", "지우기 / 추가"),
             ("⑤", "보고서 출력", "엑셀 · 야장 PDF")]
    w, gap = 16.5, 4.3
    for i, (num, title, sub) in enumerate(steps):
        x = 1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 4), w, 13, boxstyle="round,pad=0.5",
                                    fc=BLUE if i % 2 == 0 else NAVY, ec="none"))
        ax.text(x + w / 2, 14.2, num, ha="center", fontsize=15,
                fontweight="bold", color="white")
        ax.text(x + w / 2, 10.2, title, ha="center", fontsize=12,
                fontweight="bold", color="white")
        ax.text(x + w / 2, 6.6, sub, ha="center", fontsize=9.5, color="#DCEAF2")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, 10.5), (x + w + gap - 0.4, 10.5),
                                         arrowstyle="-|>", mutation_scale=15,
                                         lw=2.2, color=GRAY))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", path)


def fig_efficiency(path):
    """업무 효율화 — 40시간 → 4시간."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=200)
    names = ["현행\n(육안 판독)", "AI 도입 후\n(확인만)"]
    vals = [40, 4]
    bars = ax.barh(names, vals, color=["#B9B9B9", GREEN], height=0.45)
    for b, v in zip(bars, vals):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v}시간",
                va="center", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 48)
    ax.set_xlabel("현장 1개소(30km) 판독 소요시간", fontsize=11)
    ax.set_title("판독 시간 90% 이상 절감", fontsize=14, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", path)


if __name__ == "__main__":
    fig_architecture("fig_arch.png")
    fig_performance("fig_perf.png")
    fig_outside("fig_outside.png")
    fig_userflow("fig_flow.png")
    fig_efficiency("fig_eff.png")

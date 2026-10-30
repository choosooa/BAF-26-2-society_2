"""S7. EDA 결과를 반영한 인과 지도(DAG) PNG."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import savefig

LAT = ("#F1EFE8", "#5F5E5A", "--")
OBS = ("#EEEDFE", "#3C3489", "-")
OUT = ("#FAECE7", "#712B13", "-")
NEW = ("#FAEEDA", "#633806", "--")

NODES = {  # key: (x, y, 제목, 부제, 스타일)
    "A": (3.4, 5.2, "호우사상", "잠재 · 152개 블록", LAT),
    "E": (0.2, 5.2, "유역환경", "7변수 · 관측소 상수", OBS),
    "S": (6.6, 5.2, "관측소 고유특성", "기준면·보·하도", LAT),
    "M": (0.2, 3.6, "관측 강우 3종", "IMERG·지상·레이더", OBS),
    "B": (3.4, 3.6, "t까지 실제 강우", "잠재", LAT),
    "F": (6.6, 3.6, "미래 6h 강우", "미관측", LAT),
    "H": (0.2, 2.0, "과거 수위", "wl_t−24h … t−1h", OBS),
    "W": (3.4, 2.0, "현재 수위 wl_t", "선행상태", OBS),
    "P": (6.6, 2.0, "미래 최대수위", "max WL(t+1…6)", OUT),
    "T": (6.6, 0.4, "target_maxrise_6h", "= 미래최대 − wl_t", OUT),
    "D": (9.8, 3.6, "조석 (신규)", "하구 15곳 · 수위 전체에 주기", NEW),
}
W, H = 2.6, 0.9


def c(k, side):
    x, y = NODES[k][:2]
    return {"t": (x + W / 2, y + H), "b": (x + W / 2, y), "l": (x, y + H / 2), "r": (x + W, y + H / 2)}[side]


def arrow(ax, p, q, color="#5F5E5A", ls="-", lw=1.1, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(-0.2, 12.7); ax.set_ylim(-0.6, 6.4); ax.axis("off")
    for k, (x, y, t, s, (fc, ec, ls)) in NODES.items():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1, ls=ls))
        ax.text(x + W / 2, y + 0.58, t, ha="center", va="center", fontsize=11, color=ec, weight="bold")
        ax.text(x + W / 2, y + 0.26, s, ha="center", va="center", fontsize=9, color=ec)
    arrow(ax, c("A", "b"), c("B", "t"))
    arrow(ax, c("A", "r"), c("F", "t"), rad=-0.25)
    arrow(ax, c("B", "l"), c("M", "r"), ls="--")
    arrow(ax, c("B", "b"), c("W", "t"))
    arrow(ax, (3.9, 3.6), c("H", "t"), rad=0.15)
    arrow(ax, c("B", "r"), (7.2, 2.9), color="#1D9E75", lw=2.2, rad=0.25)
    arrow(ax, c("F", "b"), c("P", "t"))
    arrow(ax, c("H", "r"), c("W", "l"))
    arrow(ax, c("W", "r"), c("P", "l"))
    arrow(ax, c("P", "b"), c("T", "t"))
    arrow(ax, c("W", "b"), c("T", "l"), color="#D85A30", ls="--", rad=0.3)
    ax.plot([2.8, 3.1, 3.1, 6.9], [5.65, 5.65, 1.7, 1.7], color="#5F5E5A", lw=1.1)
    arrow(ax, (6.9, 1.7), (6.9, 2.0))
    arrow(ax, c("S", "r"), (9.2, 2.45), rad=-0.3)
    arrow(ax, c("D", "b"), c("P", "r"), color="#BA7517", lw=2.0, rad=-0.2)
    legend = [("잠재·미관측", LAT), ("관측 입력", OBS), ("결과·타깃", OUT), ("EDA로 추가된 요인", NEW)]
    for i, (lab, (fc, ec, ls)) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((0.2 + i * 2.4, -0.5), 0.35, 0.25, boxstyle="round,pad=0.01", fc=fc, ec=ec, ls=ls))
        ax.text(0.65 + i * 2.4, -0.38, lab, va="center", fontsize=9)
    ax.plot([9.9, 10.3], [-0.38, -0.38], color="#1D9E75", lw=2.2); ax.text(10.4, -0.38, "핵심 강우 경로", va="center", fontsize=9)
    ax.plot([9.9, 10.3], [-0.05, -0.05], color="#D85A30", lw=1.2, ls="--"); ax.text(10.4, -0.05, "정의상 결합(−)", va="center", fontsize=9)
    ax.set_title("EDA 반영 인과 지도: 강우-유출 경로 + 조석 경로", fontsize=13)
    savefig(fig, "07_dag_updated.png")


if __name__ == "__main__":
    main()

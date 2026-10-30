"""S2. 시간창 도식, 병합 구조(공유 격자/소권역) 실측, 커버리지 히트맵."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from common import load, ENV, IMERG, SFC, RADAR, savefig, save_json, peak_rss, RESULTS


def fig_time_window():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    rows = [
        ("수위 lag (wl_t−kh)", [(-24, "#7F77DD"), (-12, "#7F77DD"), (-6, "#7F77DD"), (-3, "#7F77DD"), (-1, "#7F77DD"), (0, "#534AB7")], "pt"),
        ("강우 누적 24h", (-23, 0), "#1D9E75"),
        ("강우 누적 6h", (-5, 0), "#1D9E75"),
        ("강우 누적 1h / 레이더", (0, 0), "#1D9E75"),
        ("타깃: max WL(t+1…t+6) − wl_t", (1, 6), "#D85A30"),
    ]
    for i, r in enumerate(rows):
        y = len(rows) - i
        if r[2] == "pt":
            for h, c in r[1]:
                ax.plot(h, y, "o", color=c, ms=8)
        else:
            a, b = r[1]
            ax.add_patch(Rectangle((a - 0.45, y - 0.3), b - a + 0.9, 0.6, color=r[2], alpha=0.75))
        ax.text(-25.5, y, r[0], ha="right", va="center", fontsize=10)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.text(0, len(rows) + 0.8, "현재 시점 t", ha="center", fontsize=10)
    ax.set_xlim(-25, 8); ax.set_ylim(0.3, len(rows) + 1.2)
    ax.set_yticks([]); ax.set_xlabel("시각 (h, t 기준)")
    ax.set_title("한 행이 보는 시간창: 과거(입력) vs 미래(타깃)", fontsize=12)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    ax = axes[1]
    for i, t0 in enumerate([0, 1, 2]):
        y = 3 - i
        ax.add_patch(Rectangle((t0 + 0.55, y - 0.3), 5.9, 0.6, color="#D85A30", alpha=0.35 + 0.2 * i))
        ax.plot(t0, y, "o", color="#534AB7", ms=8)
        ax.text(-1.2, y, f"행 t{'+' + str(t0) if t0 else ''}", ha="right", va="center", fontsize=10)
    ax.add_patch(Rectangle((2.55, 0.5), 4.9, 3.0, fill=False, ec="k", ls=":", lw=1.2))
    ax.text(5, 0.25, "세 행의 타깃 구간이 5시간 겹침 → 이웃 행이 train/valid로 갈리면 누수", ha="center", fontsize=10)
    ax.set_xlim(-4, 9); ax.set_ylim(0, 3.6); ax.set_yticks([]); ax.set_xlabel("시각 (h)")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    savefig(fig, "02a_time_window.png")


def shared_units(tr):
    out = {}
    env = tr.groupby("station_id", observed=True)[ENV].first()
    grp = env.round(4).astype(str).agg("|".join, axis=1)
    sizes = grp.value_counts()
    out["env_groups"] = {"n_unique_env": int(sizes.size), "n_station": int(len(env)),
                         "stations_in_shared_groups": int(sizes[sizes > 1].sum()),
                         "largest_group": int(sizes.max()), "group_size_hist": sizes.value_counts().sort_index().to_dict()}
    # 가장 큰(관측소 수 최대 + 긴) train 사상 블록에서 강우·레이더 시계열 해시 비교
    bl = pd.read_csv(RESULTS / "s01_event_blocks.csv")
    b = bl[bl.split == "train"].sort_values(["n_station", "n_hours"], ascending=False).iloc[0]["event_block"]
    sub = tr[tr.event_block == b]
    for name, cols in [("imerg", IMERG[:1]), ("sfc", SFC[:1]), ("radar", RADAR[:1])]:
        piv = sub.pivot_table(index="station_id", columns="hour", values=cols[0], observed=True)
        piv = piv.dropna(axis=1, thresh=int(0.95 * len(piv))).dropna()
        h = piv.round(3).astype(str).agg("|".join, axis=1)
        vc = h.value_counts()
        out[f"{name}_identical_series"] = {"block": int(b), "n_station_checked": int(len(h)), "n_unique_series": int(vc.size),
                                           "stations_sharing": int(vc[vc > 1].sum()), "largest_share": int(vc.max())}
    return out, sizes


def fig_merge(shared):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(-0.6, 5.4)
    srcs = [
        ("WAMIS 수위", "wl_t, lag 5개, 타깃", "관측소(점)", "1h 시계열", "#EEEDFE", "#534AB7"),
        ("RAIN-F IMERG", "imerg_rain 1/6/24h", "격자셀", "과거 누적", "#E1F5EE", "#0F6E56"),
        ("RAIN-F 지상기상", "sfc_rain 1/6/24h", "격자(AWS/ASOS)", "과거 누적", "#E1F5EE", "#0F6E56"),
        ("RAIN-F 레이더", "reflectivity 3종", "격자셀 요약", "t 시점", "#E1F5EE", "#0F6E56"),
        ("DataON1112", "유역환경 7변수", "표준소권역(SBSNCD)", "시간불변", "#FAECE7", "#993C1D"),
    ]
    ax.text(0.2, 5.1, "원천", fontsize=11, weight="bold"); ax.text(4.0, 5.1, "공간 단위 / 시간", fontsize=11, weight="bold")
    ax.text(7.8, 5.1, "결합 키 → 1행", fontsize=11, weight="bold")
    for i, (n, cols, sp, tm, fc, ec) in enumerate(srcs):
        y = 4.3 - i * 0.95
        ax.add_patch(FancyBboxPatch((0.2, y - 0.35), 3.2, 0.7, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=0.8))
        ax.text(0.35, y + 0.1, n, fontsize=10.5, color=ec, weight="bold"); ax.text(0.35, y - 0.18, cols, fontsize=9, color=ec)
        ax.text(4.0, y + 0.08, sp, fontsize=10); ax.text(4.0, y - 0.2, tm, fontsize=9, color="#5F5E5A")
        ax.annotate("", xy=(7.6, 2.4), xytext=(6.4, y), arrowprops=dict(arrowstyle="->", color="#888780", lw=0.8))
    ax.add_patch(FancyBboxPatch((7.7, 1.7), 3.1, 1.4, boxstyle="round,pad=0.02", fc="#F1EFE8", ec="#5F5E5A"))
    ax.text(7.85, 2.75, "(station_id, timestamp)", fontsize=10, weight="bold")
    ax.text(7.85, 2.4, "= 1행 (935,920행)", fontsize=9.5)
    ax.text(7.85, 2.05, "사상 블록 152개로 묶임", fontsize=9.5)
    e = shared["env_groups"]; im = shared["imerg_identical_series"]
    note = (f"실측: 유역환경 조합 {e['n_unique_env']}종 / 관측소 {e['n_station']}개 (공유 그룹 소속 {e['stations_in_shared_groups']}개), "
            f"IMERG 시계열 {im['n_unique_series']}종 / 관측소 {im['n_station_checked']}개 (공유 {im['stations_sharing']}개)")
    ax.text(0.2, -0.45, note, fontsize=9.5, color="#444441")
    savefig(fig, "02b_merge_structure.png")


def fig_coverage():
    tr = load("train_sorted", ["station_code", "event_block"])
    te = load("test_sorted", ["station_code", "event_block"])
    bl = pd.read_csv(RESULTS / "s01_event_blocks.csv").sort_values("h0")
    order = bl["event_block"].to_numpy()
    mats = []
    for df, val in [(tr, 1), (te, 2)]:
        c = df.groupby(["station_code", "event_block"]).size().unstack(fill_value=0)
        mats.append((c > 0).astype(int) * val)
    m = mats[0].add(mats[1], fill_value=0).reindex(columns=order, fill_value=0).fillna(0)
    # 블록 폭 = 시간 길이에 비례하도록 반복
    widths = bl.set_index("event_block").loc[order, "n_hours"].clip(upper=40).to_numpy()
    mm = np.repeat(m.to_numpy(), widths, axis=1)
    frac = np.repeat(m.to_numpy(), 1, axis=1)
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.imshow(mm, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#F1EFE8", "#534AB7", "#D85A30"]), vmin=0, vmax=2)
    ax.set_xlabel("사상 블록 (합성 시간순, 폭 ∝ 길이, 최대 40h로 절단)"); ax.set_ylabel("관측소 (465)")
    ax.set_title("관측소 × 사상 커버리지  (보라=train, 주황=test, 회색=없음)")
    ax.set_xticks([]);
    savefig(fig, "02c_coverage_heatmap.png")
    return {"station_block_presence_rate": float((frac > 0).mean())}


def fig_blocks():
    bl = pd.read_csv(RESULTS / "s01_event_blocks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for s, c in [("train", "#534AB7"), ("test", "#D85A30")]:
        d = bl[bl.split == s]
        axes[0].hist(d.n_hours, bins=np.arange(0, 160, 5), alpha=0.6, color=c, label=f"{s} ({len(d)}개)")
        axes[1].hist(d.n_rows, bins=30, alpha=0.6, color=c, label=s)
    axes[0].set_xlabel("사상 길이 (시간)"); axes[0].set_ylabel("사상 수"); axes[0].legend()
    axes[1].set_xlabel("사상당 행 수"); axes[1].legend()
    fig.suptitle("복원된 호우사상 블록의 크기")
    fig.tight_layout()
    savefig(fig, "02d_event_blocks.png")


def main():
    fig_time_window()
    tr = load("train_sorted", ["station_id", "hour", "event_block"] + ENV + IMERG[:1] + SFC[:1] + RADAR[:1])
    shared, _ = shared_units(tr)
    del tr
    fig_merge(shared)
    shared["coverage"] = fig_coverage()
    fig_blocks()
    save_json(shared, "s02_structure.json")
    print(shared)
    peak_rss("s02")


if __name__ == "__main__":
    main()

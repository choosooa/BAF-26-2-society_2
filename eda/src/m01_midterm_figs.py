"""중간보고서(EDA_중간보고서.md)용 새 그림 7장.

실행: python eda/src/m01_midterm_figs.py  (캐시 사용)
출력: eda/figures/midterm/m*.png, eda/cache/results/m01_midterm.json
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm

from common import CACHE, RESULTS, EDA, CAT, INK, INK2, MUTED, GRID, BASE, div_cmap, save_json
from tb_common import load_tb, station_table, style, figsize, TARGET, ENV, C_ALL, C_NONTIDAL, C_TIDAL

FIG_M = EDA / "figures" / "midterm"
FIG_M.mkdir(parents=True, exist_ok=True)
OUT = {}


def save(fig, slug):
    path = FIG_M / f"{slug}.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}")


def terc(s):
    """관측소 단위 값을 순위 기준 3분위(하/중/상)로."""
    return pd.qcut(s.rank(method="first"), 3, labels=["하", "중", "상"]).astype(str)


# ============================================================
# 0. 데이터
# ============================================================
style("report")
d = load_tb("train_sorted")
oof = pd.read_parquet(CACHE / "oof_s09.parquet", columns=["station_code", "hour", "oof_base"])
d = d.merge(oof, on=["station_code", "hour"], how="left", validate="one_to_one")
ev_len = d.groupby("event_block")["hour"].agg(lambda h: h.max() - h.min() + 1)
d["ev_len"] = d["event_block"].map(ev_len)
nontidal = ~d["is_tidal"] & ~d["is_stuck"]
nt = d[nontidal]
dam_ids = json.load(open(RESULTS / "nb07_summary.json", encoding="utf-8"))["dam_ids"]


# ============================================================
# m1. 문제 정의: 타깃은 무엇이고, 왜 사상 단위로 나눠야 하나
# ============================================================
def fig_problem():
    # 예시: 사상 #125의 ST0265(연결망 예시 관측소). t = 사상 22시간째:
    # 6시간 강우 45mm가 이미 내렸지만 수위는 1시간에 6cm만 오른 상태 → 이후 6시간 동안 1.56m 상승
    ex = d[(d.event_block == 125) & (d.station_id == "ST0265")].sort_values("hour").reset_index(drop=True)
    h = ex["hour"] - ex["hour"].min()
    ex = ex[h <= 36].copy()
    h = h[h <= 36]
    i_t = int(np.flatnonzero(h.to_numpy() == 22)[0])
    t0 = h[i_t]
    win = ex.loc[i_t + 1:i_t + 6]
    i_max = int(win["wl_t"].idxmax())
    OUT["m1_example"] = {"station": "ST0265", "event": 125, "t": int(t0), "target": float(ex.loc[i_t, TARGET])}

    fig = plt.figure(figsize=figsize(17, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.2], width_ratios=[1.35, 1], hspace=0.08, wspace=0.28)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_s = fig.add_subplot(gs[:, 1])

    ax_r.bar(h, ex["sfc_rain_1h"], width=0.8, color=CAT[0], alpha=0.85)
    ax_r.set_ylabel("지상강수\n1h (mm)")
    ax_r.tick_params(labelbottom=False)
    ax_r.set_title("타깃 = 앞으로 6시간 최고 수위 − 지금 수위", loc="left", fontsize=10)

    ax_w.axvspan(t0 + 0.5, t0 + 6.5, color=CAT[1], alpha=0.10, lw=0)
    ax_w.plot(h[:i_t + 1], ex["wl_t"][:i_t + 1], color=INK, lw=1.8)
    ax_w.plot(h[i_t:], ex["wl_t"][i_t:], color=MUTED, lw=1.4, ls="--")
    ax_w.scatter([t0], [ex.loc[i_t, "wl_t"]], s=36, color=INK, zorder=3, edgecolor="white", linewidth=1.5)
    ax_w.scatter([h[i_max]], [ex.loc[i_max, "wl_t"]], s=36, color=CAT[1], zorder=3, edgecolor="white", linewidth=1.5)
    y0, y1 = ex.loc[i_t, "wl_t"], ex.loc[i_max, "wl_t"]
    ax_w.annotate("", xy=(h[i_max] + 0.6, y1), xytext=(h[i_max] + 0.6, y0),
                  arrowprops=dict(arrowstyle="<->", color=CAT[1], lw=1.5))
    ax_w.hlines(y0, t0, h[i_max] + 0.8, color=CAT[1], lw=0.8, ls=":")
    ax_w.text(h[i_max] + 1.0, (y0 + y1) / 2, f"타깃\n+{y1 - y0:.2f} m", color=INK, va="center", fontsize=9, fontweight="bold")
    ax_w.axvline(t0, color=INK2, lw=0.8)
    ax_w.text(t0 - 0.4, ax_w.get_ylim()[1], "예측 시점 t\n(여기까지만 입력)", ha="right", va="top", fontsize=8, color=INK2)
    ax_w.text(t0 + 3.5, ax_w.get_ylim()[0] + 0.05, "예측 창\nt+1 ~ t+6", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax_w.set_ylabel("수위 (m)")
    ax_w.set_xlabel("사상 시작 후 경과 시간 (h)  —  사상 #125, ST0265")

    # 오른쪽: 이웃 행의 예측 창이 겹친다
    ax_s.set_xlim(-1.2, 13.2)
    ax_s.set_ylim(-0.7, 6.6)
    ax_s.invert_yaxis()
    ax_s.grid(False)
    for spine in ax_s.spines.values():
        spine.set_visible(False)
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    split = {0: "train", 1: "valid", 2: "train", 3: "valid", 4: "train"}
    for r in range(5):
        c = CAT[0] if split[r] == "train" else CAT[2]
        ax_s.add_patch(Rectangle((r + 0.5, r - 0.28), 6, 0.56, color=c, alpha=0.22, lw=0))
        ax_s.plot([r], [r], "o", color=c, ms=6)
        ax_s.text(r - 0.35, r, f"t+{r}" if r else "t", va="center", ha="right", fontsize=8, color=INK2)
        ax_s.text(r + 6.7, r, split[r], va="center", fontsize=7.5, color=c, fontweight="bold")
    ax_s.add_patch(Rectangle((1.5, -0.45), 5, 1.9, fill=False, ec=CAT[1], lw=1.2, ls="--"))
    ax_s.annotate("", xy=(11.8, 4.75), xytext=(-0.5, 4.75), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax_s.text(12.0, 4.75, "시간", ha="left", va="center", fontsize=7.5, color=MUTED)
    ax_s.text(-0.5, 5.05, "● 행(예측 시점)   ▬ 그 행의 예측 창 6시간\n점선 = t행과 t+1행이 공유하는 5시간",
              fontsize=7.5, color=INK2, va="top")
    ax_s.set_title("행을 무작위로 나누면 정답이 검증 쪽으로 샌다", loc="left", fontsize=10)
    ax_s.text(-0.5, 6.1, "무작위 KFold RMSE 0.198 vs 사상 GroupKFold 0.226 → 약 12% 낙관적",
              fontsize=8, color=INK, va="top", fontweight="bold")
    save(fig, "m1_problem")


# ============================================================
# m2. 오차는 어디에 사나: 0 예측 vs 기준 모델
# ============================================================
def fig_error_share():
    y = d[TARGET].to_numpy()
    se0 = y ** 2
    seb = (y - d["oof_base"].to_numpy()) ** 2
    segs = {
        "상위 1% 타깃 행": y >= np.quantile(y, 0.99),
        "조석 관측소 15곳": d["is_tidal"].to_numpy(),
        "방류 의심 관측소 23곳": d["station_id"].isin(dam_ids).to_numpy(),
        "6h 지상강수 15mm 이상": (d["sfc_rain_6h"] >= 15).to_numpy(),
        "25h 이상 긴 사상": (d["ev_len"] >= 25).to_numpy(),
    }
    tab = pd.DataFrame({k: {"행 비중": m.mean(), "0 예측 오차 비중": se0[m].sum() / se0.sum(),
                            "기준 모델 오차 비중": seb[m].sum() / seb.sum()} for k, m in segs.items()}).T
    OUT["m2_error_share"] = tab.round(4).to_dict(orient="index")
    tab.to_csv(FIG_M / "m2_error_share.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=figsize(16, 8))
    cols = [("행 비중", C_ALL), ("0 예측 오차 비중", CAT[1]), ("기준 모델 오차 비중", CAT[0])]
    yy = np.arange(len(tab))[::-1]
    hgt = 0.26
    for j, (c, col) in enumerate(cols):
        vals = tab[c].to_numpy() * 100
        pos = yy + (1 - j) * hgt
        ax.barh(pos, vals, height=hgt - 0.03, color=col, label=c)
        for p, v in zip(pos, vals):
            ax.text(v + 0.8, p, f"{v:.0f}%", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels(tab.index)
    ax.set_xlim(0, 85)
    ax.set_xlabel("비중 (%)  —  오차 비중 = 해당 행의 제곱오차 합 ÷ 전체 제곱오차 합")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, frameon=False)
    ax.grid(axis="y", visible=False)
    ax.set_title("기준 모델이 조석은 거의 풀었다. 남은 오차는 강한 비·긴 사상·큰 상승에 몰려 있다", loc="left")
    save(fig, "m2_error_share")


# ============================================================
# m3. 신호는 짧은 창에 있다: 창 길이별 타깃과의 순위상관 (비조석)
# ============================================================
def fig_signal_decay():
    ry = nt[TARGET].rank()

    def rho(c):
        return float(np.corrcoef(nt[c].rank(), ry)[0, 1])

    lines = {
        "수위 변화 dwl_k": ([1, 3, 6, 12, 24], [rho(f"dwl_{k}h") for k in [1, 3, 6, 12, 24]], INK),
        "지상강수 k시간 누적": ([1, 6, 24], [rho(f"sfc_rain_{k}h") for k in [1, 6, 24]], CAT[0]),
        "IMERG k시간 누적": ([1, 6, 24], [rho(f"imerg_rain_{k}h") for k in [1, 6, 24]], CAT[2]),
    }
    radar = rho("radar_reflectivity_mean")
    env = {c: rho(c) for c in ENV}
    OUT["m3_spearman_nontidal"] = {k: dict(zip(map(str, v[0]), v[1])) for k, v in lines.items()}
    OUT["m3_spearman_nontidal"]["radar_mean"] = radar
    OUT["m3_spearman_nontidal"]["env"] = env

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize(17, 7), gridspec_kw={"width_ratios": [1.6, 1]})
    for name, (x, v, col) in lines.items():
        a1.plot(x, v, "-o", color=col, lw=2, ms=6, mec="white", mew=1.2, label=name)
    a1.scatter([1], [radar], marker="D", s=34, color=CAT[1], zorder=3, edgecolor="white", linewidth=1)
    a1.annotate("레이더 평균(현재 시각)", (1, radar), xytext=(1.5, 0.435), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    a1.set_xscale("log")
    a1.set_xticks([1, 3, 6, 12, 24])
    a1.set_xticklabels(["1h", "3h", "6h", "12h", "24h"])
    a1.set_xlim(0.8, 32)
    a1.set_ylim(0, 0.46)
    a1.set_xlabel("창 길이 k (로그 축)")
    a1.set_ylabel("타깃과의 Spearman ρ")
    a1.legend(loc="lower left", frameon=False, fontsize=7.5)
    a1.set_title("최근 1~6시간이 가장 강하다", loc="left")

    env_s = pd.Series(env).reindex(ENV)
    names = {"cat_area": "소권역 면적", "wsarea": "상류 유역면적", "mean_elevation": "평균 표고", "mean_slope": "평균 경사",
             "urban_ratio": "도시화율", "forest_ratio": "산림 비율", "stream_order": "하천차수"}
    env_s = env_s.sort_values()
    a2.barh([names[k] for k in env_s.index], env_s.values, color=C_ALL, height=0.6)
    a2.axvline(0, color=BASE, lw=0.8)
    a2.set_xlim(-0.46, 0.46)
    a2.set_xlabel("타깃과의 Spearman ρ")
    a2.grid(axis="y", visible=False)
    a2.set_title("유역환경은 단독으로 |ρ| < 0.1", loc="left")
    fig.subplots_adjust(wspace=0.55)
    fig.text(0.01, -0.1, "비조석 관측소 기준(조석 15곳·고정 1곳 제외, 72.6만 행)", ha="left", fontsize=7.5, color=INK2)
    save(fig, "m3_signal_decay")


# ============================================================
# m4. 강우 소스가 일치하는가
# ============================================================
def fig_rain_sources():
    cats = ["둘 다", "IMERG만", "지상만", "둘 다 없음"]
    colors = {"둘 다": CAT[0], "IMERG만": CAT[2], "지상만": CAT[1], "둘 다 없음": "#d3d1c9"}
    share, stat = {}, {}
    for w in ["1h", "6h", "24h"]:
        i = d[f"imerg_rain_{w}"] > 0
        s = d[f"sfc_rain_{w}"] > 0
        cat = pd.Series(np.select([i & s, i & ~s, ~i & s], cats[:3], cats[3]), index=d.index)
        share[w] = cat.value_counts(normalize=True).reindex(cats)
        g = d.loc[nontidal, TARGET].groupby(cat[nontidal])
        stat[w] = pd.DataFrame({"평균": g.mean(), "p95": g.quantile(0.95)}).reindex(cats)
    OUT["m4_share"] = {w: s.round(4).to_dict() for w, s in share.items()}
    OUT["m4_target_nontidal"] = {w: s.round(4).to_dict(orient="index") for w, s in stat.items()}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize(17, 6.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ws = ["1h", "6h", "24h"]
    for r, w in enumerate(ws):
        left = 0
        for c in cats:
            v = share[w][c] * 100
            a1.barh(r, v, left=left, color=colors[c], height=0.62, edgecolor="#fcfcfb", linewidth=1.5,
                    label=c if r == 0 else None)
            if v >= 6:
                a1.text(left + v / 2, r, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if c in ("둘 다",) else INK)
            left += v
    a1.set_yticks(range(3))
    a1.set_yticklabels([f"{w} 창" for w in ws])
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel("행 비율 (%)  —  '비 옴' = 해당 창 누적 > 0")
    a1.grid(axis="y", visible=False)
    a1.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=7.5)
    a1.set_title("짧은 창에서는 IMERG만 비를 보는 행이 35%", loc="left")

    s6 = stat["6h"]
    yy = np.arange(len(cats))
    a2.barh(yy, s6["p95"], color=[colors[c] for c in cats], height=0.6)
    a2.scatter(s6["평균"], yy, marker="|", s=160, color=INK, lw=2, zorder=3, label="평균")
    for y_, (m, p) in enumerate(zip(s6["평균"], s6["p95"])):
        a2.text(p + 0.02, y_, f"p95 {p:.2f}", va="center", fontsize=7.5, color=INK2)
    a2.set_yticks(yy)
    a2.set_yticklabels(cats)
    a2.invert_yaxis()
    a2.set_xlim(0, 0.95)
    a2.set_xlabel("타깃 (m) · 6h 창 · 비조석")
    a2.grid(axis="y", visible=False)
    a2.legend(loc="lower right", frameon=False, fontsize=7.5)
    a2.set_title("두 소스가 함께 볼 때만 크게 오른다 (막대 = p95)", loc="left")
    fig.subplots_adjust(wspace=0.42)
    save(fig, "m4_rain_sources")


# ============================================================
# m5. 유역 조절 효과: 섞어 본 차이 × 사상별 재현율 (비조석, 강우량 통제)
# ============================================================
def fig_moderation():
    st = station_table(d)
    X = st[ENV].astype(float).copy()
    for c in ["cat_area", "wsarea", "mean_elevation", "stream_order"]:
        X[c] = np.log1p(X[c])
    Z = (X - X.mean()) / X.std()
    U, S, _ = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    pcs = pd.DataFrame(U * S, index=Z.index, columns=[f"PC{i + 1}" for i in range(len(S))])
    if np.corrcoef(pcs.PC1, X.urban_ratio)[0, 1] < 0:
        pcs["PC1"] *= -1
    if np.corrcoef(pcs.PC2, X.wsarea)[0, 1] < 0:
        pcs["PC2"] *= -1
    mods = {"상류 유역면적": X.wsarea, "PC2 하천 규모": pcs.PC2, "도시화율": X.urban_ratio,
            "PC1 저지·도시": pcs.PC1, "산림 비율": X.forest_ratio, "평균 경사": X.mean_slope}
    mod_terc = {k: terc(v) for k, v in mods.items()}

    rows = []
    for w in ["1h", "6h", "24h"]:
        wet = nt[nt[f"sfc_rain_{w}"] > 0][["station_id", "event_block", TARGET, f"sfc_rain_{w}"]].copy()
        r = wet[f"sfc_rain_{w}"]
        wet["oct"] = pd.qcut(r.rank(method="first"), 8, labels=False) + 1
        # 강우량 통제: 강우 40분위 칸 평균을 뺀 잔차로 비교 (3분위 간 강우량 차이 제거)
        wet["bin"] = pd.qcut(r.rank(method="first"), 40, labels=False)
        wet["res"] = wet[TARGET] - wet.groupby("bin")[TARGET].transform("mean")
        for lvl, sel in {"중간 비": wet["oct"].between(4, 6), "강한 비": wet["oct"] == 8}.items():
            h = wet[sel]
            rng = (float(h[f"sfc_rain_{w}"].min()), float(h[f"sfc_rain_{w}"].max()))
            for name, tt in mod_terc.items():
                t = h["station_id"].map(tt)
                pooled = h["res"][t == "상"].mean() - h["res"][t == "하"].mean()
                g = pd.DataFrame({"e": h["event_block"], "t": t, "y": h["res"]}).groupby(["e", "t"])["y"].agg(["mean", "size"]).unstack("t")
                ok = (g["size"]["상"] >= 20) & (g["size"]["하"] >= 20)
                diff = (g["mean"]["상"] - g["mean"]["하"])[ok]
                e = diff.index.to_numpy()
                same = np.sign(diff) == np.sign(pooled)
                rows.append({"창": w, "강도": lvl, "강우 범위": rng, "조절 변수": name, "섞어 본 차이": pooled,
                             "사상 수": int(ok.sum()), "재현율": same.mean(),
                             "짝": same[e % 2 == 0].mean(), "홀": same[e % 2 == 1].mean()})
    res = pd.DataFrame(rows)
    res.to_csv(FIG_M / "m5_moderation.csv", index=False, encoding="utf-8-sig")
    OUT["m5_moderation"] = res.assign(**{"강우 범위": res["강우 범위"].astype(str)}).round(3).to_dict(orient="records")

    col_order = [("중간 비", "1h"), ("중간 비", "6h"), ("중간 비", "24h"), ("강한 비", "1h"), ("강한 비", "6h"), ("강한 비", "24h")]
    row_order = list(mods)
    eff = np.full((len(row_order), len(col_order)), np.nan)
    rep = np.full_like(eff, np.nan)
    for j, (lvl, w) in enumerate(col_order):
        for i, m in enumerate(row_order):
            r = res[(res["강도"] == lvl) & (res["창"] == w) & (res["조절 변수"] == m)].iloc[0]
            eff[i, j], rep[i, j] = r["섞어 본 차이"], r["재현율"]

    fig, ax = plt.subplots(figsize=figsize(17, 8.4))
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)
    im = ax.imshow(eff, cmap=div_cmap(), norm=norm, aspect="auto")
    for i in range(eff.shape[0]):
        for j in range(eff.shape[1]):
            ok = rep[i, j] >= 0.7
            txt_c = "white" if abs(eff[i, j]) > 0.15 else INK
            ax.text(j, i - 0.13, f"{eff[i, j]:+.2f}", ha="center", va="center", fontsize=8.5,
                    color=txt_c, fontweight="bold" if ok else "normal")
            ax.text(j, i + 0.2, f"사상 {rep[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7,
                    color=txt_c if ok else MUTED)
            if not ok:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="#b9b7ae", lw=0))
    rng_lbl = {}
    for (lvl, w) in col_order:
        r = res[(res["강도"] == lvl) & (res["창"] == w)].iloc[0]["강우 범위"]
        rng_lbl[(lvl, w)] = f"{lvl}\n{w} {r[0]:g}~{r[1]:g}mm" if lvl == "중간 비" else f"{lvl}\n{w} ≥{r[0]:g}mm"
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([rng_lbl[c] for c in col_order], fontsize=7.5)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([f"{m} (상 − 하)" for m in row_order])
    ax.axvline(2.5, color="white", lw=3)
    ax.grid(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("3분위 상위 − 하위 타깃 차이 (m, 강우량 통제)", fontsize=7.5)
    cb.outline.set_visible(False)
    ax.set_title("같은 비라도 누가 더 오르나: 중간 비에선 작은·도시 하천, 긴 강한 비에선 큰 하천", loc="left")
    fig.text(0.01, -0.02, "글자 = 섞어 본 차이(m) / 사상 재현율(같은 방향인 사상 비율). 빗금 = 재현율 70% 미만(채택 기준 미달). "
             "비조석·비 온 행, 강우 8분위 기준(중간 = 4~6분위, 강한 = 8분위), 사상마다 3분위별 20행 이상인 사상만.",
             fontsize=7, color=INK2)
    save(fig, "m5_moderation")


# ============================================================
# m6. 현재 위치: 기준선 사다리
# ============================================================
def fig_ladder():
    b = pd.read_csv(RESULTS / "s05_baseline.csv")
    g = b[b["split"].str.startswith("Group")].set_index("method")
    r = b[b["split"].str.startswith("무작위")].set_index("method")
    net = json.load(open(RESULTS / "s09_network.json", encoding="utf-8"))["value_check"]["전체"]["rmse_net"]
    order = ["0 예측", "전체 평균", "관측소 평균", "관측소×강우6h구간 평균", "HGB 원변수", "HGB 원변수+차분+관측소통계"]
    labels = ["0으로 예측", "전체 평균", "관측소 평균", "관측소 × 6h 강우 구간 평균", "HGB · 원변수", "HGB · 원변수 + 차분 + 관측소 통계"]
    vals = [g.loc[m, "rmse"] for m in order] + [net]
    labels = labels + ["(규정 확인 후) + 상류 관측소 변수"]
    OUT["m6_ladder"] = dict(zip(labels, vals))

    fig, ax = plt.subplots(figsize=figsize(16, 7.4))
    yy = np.arange(len(vals))[::-1]
    cols = [C_ALL] * 4 + [CAT[0], CAT[0], "#9ec5f4"]
    ax.barh(yy, vals, color=cols, height=0.62)
    for y_, v, m in zip(yy, vals, order + [None]):
        ax.text(v + 0.004, y_, f"{v:.3f}", va="center", fontsize=8, color=INK)
        if m in r.index:
            rv = r.loc[m, "rmse"]
            ax.plot([rv, rv], [y_ - 0.31, y_ + 0.31], color=CAT[1], lw=2)
    ax.plot([], [], color=CAT[1], lw=2, label="같은 모델, 무작위 KFold (낙관적)")
    ax.axvline(vals[5], color=CAT[0], lw=0.8, ls=":")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.46)
    ax.set_xlabel("사상 단위 GroupKFold RMSE (m, 낮을수록 좋음)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("현재 기준 모델: RMSE 0.226m (0 예측 대비 −45%). 통계 평균만으로는 0.35m 벽을 못 넘는다", loc="left")
    save(fig, "m6_ladder")


# ============================================================
# m7. 향후 로드맵
# ============================================================
def fig_roadmap():
    stages = [
        ("1", "검증 틀 고정", "사상 단위 GroupKFold · 행 수 균형 fold\n전체 / 비조석 / 상위 5사상 제외 RMSE", "완료", "덩어리 1"),
        ("2", "기준 모델", "HGB · 원변수 + 차분 + 관측소 통계\nOOF RMSE 0.226", "완료", "덩어리 2"),
        ("3", "핵심 파생변수", "수위 추세(dwl, 부호 × 강우) · 겹치지 않는 강우 창\n사상 누적강우 · 경과 시간 · 강우 소스 일치도", "다음", "덩어리 4·5"),
        ("4", "유역 반응", "수위 분위 q × 강우 · 규모 × 긴 강한 비\n도시화 × 중간 비 (방향 역전 → 단조 제약 금지)", "다음", "덩어리 6"),
        ("5", "관측소 특성", "구간 플래그(조석 세기 · 방류 의심 23 · 초단기 33)\nfold 안 관측소 통계 · 조석 분리 모델은 필요할 때만", "다음", "덩어리 3"),
        ("6", "변수 선택 실험", "A 전체  vs  B 중복 축소  vs  C 핵심 + 상호작용\n같은 fold에서 비교, 상관만 보고 삭제하지 않음", "다음", "덩어리 7"),
        ("7", "꼬리 대응과 앙상블", "강한 비 행 가중치 · 손실 함수 비교\nLightGBM / XGBoost / CatBoost 앙상블 · 상류 변수(규정 확인)", "이후", "덩어리 2"),
    ]
    fig, ax = plt.subplots(figsize=figsize(17, 15))
    ax.set_xlim(0, 100)
    ax.set_ylim(len(stages) * 11 + 3, -2.5)
    ax.axis("off")
    ax.plot([6, 6], [3, (len(stages) - 1) * 11 + 3], color=BASE, lw=2, zorder=0)
    for k, (num, title, body, status, ref) in enumerate(stages):
        y = k * 11 + 3
        done = status == "완료"
        fc = CAT[0] if done else "#fcfcfb"
        ax.scatter([6], [y], s=420, color=fc, edgecolor=CAT[0], linewidth=1.8, zorder=3)
        ax.text(6, y, num, ha="center", va="center", fontsize=10, fontweight="bold", color="white" if done else CAT[0], zorder=4)
        box = FancyBboxPatch((11, y - 4.6), 86, 9.2, boxstyle="round,pad=0.2,rounding_size=1.2",
                             fc="#f0f5fc" if done else "#fcfcfb", ec=CAT[0] if done else BASE, lw=1)
        ax.add_patch(box)
        ax.text(13, y - 2.7, title, fontsize=10.5, fontweight="bold", color=INK, va="center")
        ax.text(13, y + 1.9, body, fontsize=8, color=INK2, va="center", linespacing=1.35)
        tag = {"완료": "✓ 완료", "다음": "다음 단계", "이후": "이후"}[status]
        ax.text(95.5, y - 2.7, tag, fontsize=8, ha="right", va="center", color=CAT[0] if done else INK2,
                fontweight="bold" if done else "normal")
        ax.text(95.5, y + 1.9, f"근거: {ref}", fontsize=7.5, ha="right", va="center", color=MUTED)
    ax.text(11, len(stages) * 11 + 0.5,
            "모든 단계의 채택 기준: 사상 GroupKFold RMSE 개선 + 5개 fold 중 4개 이상에서 개선 + 상위 5사상 제외 점수 악화 없음",
            fontsize=8, color=INK, va="center")
    ax.set_title("향후 계획: EDA 발견을 순서대로 실험에 옮긴다", loc="left", fontsize=11)
    save(fig, "m7_roadmap")


if __name__ == "__main__":
    fig_problem()
    fig_error_share()
    fig_signal_decay()
    fig_rain_sources()
    fig_moderation()
    fig_ladder()
    fig_roadmap()
    save_json(OUT, "m01_midterm.json")

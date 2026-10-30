"""T6. 6장 시계열·수문(그림 6-1~6-6) + 7장 train/test·검증(그림 7-1~7-3)."""
import numpy as np
import pandas as pd
from scipy import stats

from tb_common import (load_tb, style, figsize, save, save_table, save_json, load_json, plt, RESULTS,
                       C_ALL, C_NONTIDAL, C_TIDAL, CAT, SEQ, FEATURES, TARGET, TIDAL, WEAK_TIDAL, OPERATED_24H)
from common import load

GCOL = {"일반": C_NONTIDAL, "조석": C_TIDAL, "약한 조석": CAT[3], "24h 조작": CAT[4], "고정": "black"}


# ---------------- 6.1 수문곡선 ----------------
def fig_6_1(tr):
    d = tr[tr.st_group == "일반"]
    blk = d.groupby("event_block").size().idxmax()
    b = d[d.event_block == blk]
    peak = b.groupby("station_id", observed=True).agg(rise=(TARGET, "max"), area=("wsarea", "first"))
    peak = peak[peak.rise > 0.5]
    peak["t"] = pd.qcut(peak.area.rank(method="first"), 3, labels=["소", "중", "대"])
    pick = peak.sort_values("rise").groupby("t", observed=True).tail(1)
    fig, ax = plt.subplots(3, 1, figsize=figsize(16, 12), sharex=True)
    res = {}
    for a, (sid, row) in zip(ax, pick.sort_values("area").iterrows()):
        s = b[b.station_id == sid].sort_values("timestamp")
        h = (s.timestamp - s.timestamp.min()).dt.total_seconds() / 3600
        a.plot(h, s.wl_t - s.wl_t.iloc[0], color=C_NONTIDAL, label="수위 변화 (m)")
        a.set_ylabel("수위 − 시작값 (m)")
        a2 = a.twinx()
        a2.bar(h, s.sfc_rain_1h, color=CAT[3], alpha=.6, width=.9, label="지상 1h 강우")
        a2.invert_yaxis(); a2.set_ylabel("강우 (mm)", color=CAT[3]); a2.grid(False)
        a2.set_ylim(s.sfc_rain_1h.max() * 3 + 1, 0)
        a.set_title(f"{sid}  유역 '{row.t}' (상류면적 {row.area:,.0f})", fontsize=8)
        res[str(sid)] = float(row.area)
    ax[-1].set_xlabel(f"사상 #{blk} 시작 후 경과 시간 (h)")
    fig.suptitle("그림 6-1  강우(위, 거꾸로)와 수위 반응: 유역이 클수록 느리고 완만하다", x=0.02, ha="left")
    save(fig, "6-1", "hydrographs")
    return {"block": int(blk), "stations": res}


# ---------------- 6.2 자기상관 ----------------
def acf_by_station(tr, lags):
    d = tr[["station_id", "timestamp", "dwl_1h"]].copy()
    d["h"] = (d.timestamp - d.timestamp.min()).dt.total_seconds() // 3600
    key = d.set_index(["station_id", "h"])["dwl_1h"]
    out = {}
    for k in lags:
        sh = d[["station_id", "h"]].copy(); sh["h"] += k
        y = key.reindex(pd.MultiIndex.from_frame(sh)).to_numpy()
        m = ~np.isnan(y)
        t = pd.DataFrame({"s": d.station_id.to_numpy()[m], "x": d.dwl_1h.to_numpy()[m], "y": y[m]})
        out[k] = t.groupby("s", observed=True)[["x", "y"]].apply(lambda g: g.x.corr(g.y))
    return pd.DataFrame(out)


def fig_6_2_3(tr):
    lags = list(range(1, 27))
    A = acf_by_station(tr, lags)
    fp = pd.read_csv(RESULTS / "s08_station_fingerprints.csv", index_col=0)
    grp = tr.groupby("station_id", observed=True).st_group.first()
    typ = fp["type"].map({0: "C1", 1: "C2", 2: "C3", 3: "C4"}) if fp["type"].min() == 0 else fp["type"].map(lambda v: f"C{int(v)}")
    lab = {"C1": "C1 조용한 산지 소하천", "C2": "C2 약하게 반응하는 저지·도시", "C3": "C3 크게 반응하는 중규모", "C4": "C4 비 없이도 움직이는 대하천"}
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 6.5))
    gen = A[grp.reindex(A.index) == "일반"]
    for (t, col) in zip(sorted(lab), CAT):
        idx = gen.index.intersection(typ[typ == t].index)
        ax[0].plot(lags, gen.loc[idx].median(), "-o", ms=2.5, color=col, label=f"{lab[t]} (n={len(idx)})")
    ax[0].plot(lags, A.loc[A.index.intersection(TIDAL)].median(), "--", color=C_TIDAL, label="조석 15곳")
    ax[0].axhline(0, color=C_ALL, lw=.6); ax[0].set_xlabel("시차 (h)"); ax[0].set_ylabel("자기상관 (중앙값)")
    ax[0].legend(fontsize=5, loc="lower right", ncol=1); ax[0].set_xticks([1, 6, 12, 18, 24])
    ax[0].set_title("지속성: 유형별 dwl_1h 자기상관", fontsize=9)
    g = grp.reindex(A.index)
    for k, col in GCOL.items():
        m = g == k
        ax[1].scatter(A.loc[m, 12], A.loc[m, 24], s=8 if k == "일반" else 26, color=col, alpha=.5 if k == "일반" else 1,
                      label=f"{k} (n={int(m.sum())})")
    ax[1].axhline(0, color=C_ALL, lw=.6); ax[1].axvline(0, color=C_ALL, lw=.6)
    ax[1].set_xlabel("시차 12h 자기상관"); ax[1].set_ylabel("시차 24h 자기상관"); ax[1].legend(fontsize=6, loc="lower right")
    ax[1].set_title("주기성 평면")
    fig.suptitle("그림 6-2  수위 변화의 자기상관: 지속성과 주기성", x=0.02, ha="left")
    save(fig, "6-2", "acf")

    # 6-3 주기 관측소 예시
    ex = [(TIDAL[0], "조석"), (WEAK_TIDAL[0], "약한 조석"), (OPERATED_24H[0], "24h 조작")]
    fig, ax = plt.subplots(3, 1, figsize=figsize(16, 10), sharex=True)
    for a, (sid, k) in zip(ax, ex):
        s = tr[tr.station_id == sid].sort_values("timestamp")
        blk = s.groupby("event_block").size().idxmax()
        s = s[s.event_block == blk]
        h = (s.timestamp - s.timestamp.min()).dt.total_seconds() / 3600
        a.plot(h, s.wl_t, color=GCOL[k]); a.set_ylabel("wl_t (m)")
        a2 = a.twinx(); a2.bar(h, s.sfc_rain_1h, color=CAT[3], alpha=.4, width=.9); a2.invert_yaxis(); a2.grid(False)
        a2.set_ylim(max(s.sfc_rain_1h.max() * 3, 5), 0)
        a.set_title(f"{sid} ({k})  12h ACF {A.loc[sid, 12]:.2f} · 24h ACF {A.loc[sid, 24]:.2f}", fontsize=8)
    ax[-1].set_xlabel("가장 긴 사상 안 경과 시간 (h)")
    fig.suptitle("그림 6-3  비와 무관하게 주기적으로 움직이는 관측소 (막대 = 지상 강우)", x=0.02, ha="left")
    save(fig, "6-3", "periodic_examples")
    save_table(A.round(3), "6-1", "acf_by_station")
    per = {k: {"acf12_median": float(A.loc[g == k, 12].median()), "acf24_median": float(A.loc[g == k, 24].median())} for k in GCOL}
    typ_acf = {t: gen.loc[gen.index.intersection(typ[typ == t].index), [1, 3, 6]].median().round(3).to_dict() for t in lab}
    return {"by_group": per, "by_type_acf_1_3_6": typ_acf}


# ---------------- 6.4 지체시간 ----------------
def fig_6_4():
    s = pd.read_csv(RESULTS / "s04_station_summary.csv").set_index("station_id")
    s = s[~s.index.isin(TIDAL + WEAK_TIDAL + OPERATED_24H + ["ST0027"])]
    al = load_json("s10_events_residuals.json")["alignment"]["train"]["IMERG(t) vs 지상(t+k)"]["median_curve"]
    fig, ax = plt.subplots(1, 3, figsize=figsize(16, 6))
    bins = np.arange(-.5, 7.5, 1)
    ax[0].hist([s.lag_sfc_rain_1h, s.lag_imerg_rain_1h], bins=bins, color=[CAT[1], CAT[0]], label=["지상", "IMERG"])
    ax[0].set_xlabel("최대 상관 시차 (h)"); ax[0].set_ylabel("관측소 수"); ax[0].legend(); ax[0].set_title("강우 → dwl_1h 지체시간")
    res = {}
    for i, (src, col) in enumerate([("sfc", CAT[1]), ("imerg", CAT[0])]):
        lag = s[f"lag_{src}_rain_1h"].clip(upper=6)
        r = stats.spearmanr(s.wsarea, s[f"lag_{src}_rain_1h"]).statistic
        data = [np.log10(s.wsarea[lag == k]) for k in range(7)]
        bp = ax[1].boxplot(data, positions=np.arange(7) + (i - .5) * .38, widths=.34, showfliers=False, patch_artist=True)
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_alpha(.7); b.set_edgecolor("none")
        ax[1].plot([], [], color=col, lw=6, label=f"{'지상' if src == 'sfc' else 'IMERG'} ρ={r:.2f}")
        res[src] = {"lag_median": float(s[f"lag_{src}_rain_1h"].median()), "rho_wsarea": float(r),
                    "share_le_6h": float((s[f"lag_{src}_rain_1h"] <= 6).mean())}
    ax[1].set_xticks(range(7)); ax[1].set_xticklabels([str(k) for k in range(6)] + ["6+"])
    ax[1].set_xlabel("지체시간 (h)"); ax[1].set_ylabel("log10 상류 유역면적")
    ax[1].legend(fontsize=7, loc="lower right"); ax[1].set_title("유역이 클수록 늦게 반응")
    k = np.array(sorted(map(int, al))); v = np.array([al[str(i)] for i in k])
    ax[2].bar(k, v, color=[C_NONTIDAL if i == k[v.argmax()] else C_ALL for i in k])
    ax[2].set_xticks(k); ax[2].set_xlabel("k (h)"); ax[2].set_ylabel("상관 (관측소 중앙값)"); ax[2].set_title("IMERG(t)와 지상(t+k)의 상관")
    fig.suptitle(f"그림 6-4  교차상관으로 본 지체시간 (조석·조작·고정 제외 n = {len(s)})", x=0.02, ha="left")
    save(fig, "6-4", "lag_time")
    res["source_lag_argmax"] = int(k[v.argmax()]); res["source_curve"] = al
    return res


# ---------------- 6.5 사상 안 시간 ----------------
def fig_6_5(tr):
    fig, ax = plt.subplots(figsize=figsize(16, 6))
    res = {}
    for g, c in [("전체", C_ALL), ("비조석", C_NONTIDAL)]:
        d = tr if g == "전체" else tr[tr.st_group == "일반"]
        p = d.pos_in_block.clip(upper=48)
        m = d.groupby(p)[TARGET].agg(["mean", "std", "size"])
        m = m[m["size"] >= 500]
        ax.plot(m.index, m["mean"], "-", color=c, label=f"{g} 평균")
        ax.plot(m.index, m["std"], ":", color=c, label=f"{g} 표준편차")
        res[g] = {"mean_0h": float(m["mean"].iloc[0]), "mean_24h": float(m["mean"].get(24, np.nan)), "std_0h": float(m["std"].iloc[0]), "std_24h": float(m["std"].get(24, np.nan))}
    ax.set_xlabel("사상 시작 후 경과 시간 (h, 48 이상은 48)"); ax.set_ylabel("타깃 (m)"); ax.legend(fontsize=7)
    ax.set_title("그림 6-5  사상이 진행될수록 타깃의 평균과 분산이 커진다 (구간당 500행 이상)")
    save(fig, "6-5", "target_by_event_pos")
    return res


# ---------------- 6.6 관측소 간 선행 ----------------
def fig_6_6(tr):
    L = pd.read_csv(RESULTS / "s09_leaders.csv", index_col=0)
    net = load_json("s09_network.json")
    ex = net["example_pair"]
    strong = L[(L["corr"] >= .5) & ~L.index.isin(TIDAL + WEAK_TIDAL + OPERATED_24H + ["ST0027"])]
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 6), gridspec_kw={"width_ratios": [1, 1.8]})
    vc = strong["lag"].value_counts().sort_index()
    ax[0].bar(vc.index, vc.values, color=C_NONTIDAL); ax[0].set_xlabel("선행 간격 (h)"); ax[0].set_ylabel("관측소 수")
    ax[0].set_title(f"뚜렷한 선행 관측소가 있는 {len(strong)}곳")
    b = tr[tr.event_block == ex["block"]]
    for sid, col, shift, lab in [(ex["leader"], CAT[1], ex["lag"], f"{ex['leader']} (선행, {ex['lag']}h 늦춤)"), (ex["follower"], C_NONTIDAL, 0, f"{ex['follower']} (후행)")]:
        s = b[b.station_id == sid].sort_values("timestamp")
        h = (s.timestamp - b.timestamp.min()).dt.total_seconds() / 3600 + shift
        ax[1].plot(h, s.dwl_1h, color=col, label=lab, lw=1.3)
    ax[1].set_xlabel(f"사상 #{ex['block']} 경과 시간 (h)"); ax[1].set_ylabel("dwl_1h (m)"); ax[1].legend(fontsize=7)
    ax[1].set_title(f"예: 상관 {ex['corr']:.2f}")
    fig.suptitle("그림 6-6  상류 관측소의 수위 변화가 1~2시간 먼저 나타난다 (공통 호우 신호 제거 후)", x=0.02, ha="left")
    save(fig, "6-6", "lead_lag")
    return {"n_strong": int(len(strong)), "lag_share": (vc / vc.sum()).round(3).to_dict(),
            "follower_larger_area_share": net["network"]["follower_larger_area_share(diff_env)"]}


# ---------------- 7장 ----------------
def fig_7_1():
    tr = load("train_sorted", ["timestamp", "event_block", "station_id"]); te = load("test_sorted", ["timestamp", "event_block", "station_id"])
    rows = []
    for name, d in [("train", tr), ("test", te)]:
        g = d.groupby("event_block").agg(start=("timestamp", "min"), end=("timestamp", "max"), n_st=("station_id", "nunique"))
        g["split"] = name; rows.append(g)
    g = pd.concat(rows).sort_values("start").reset_index(drop=True)
    g["dur"] = (g.end - g.start).dt.total_seconds() / 3600 + 1
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 6), gridspec_kw={"width_ratios": [2, 1]})
    for sp, col in [("train", C_NONTIDAL), ("test", C_TIDAL)]:
        m = g.split == sp
        ax[0].scatter(g.index[m], g.dur[m], s=10, color=col, label=f"{sp} 사상 {int(m.sum())}개")
    ax[0].set_yscale("log"); ax[0].set_xlabel("사상 순서 (합성 시각 기준)"); ax[0].set_ylabel("사상 길이 (h, log)"); ax[0].legend()
    ax[0].set_title("test 사상은 시간축 전체에 흩어져 있다")
    ax[1].hist([g.n_st[g.split == "train"], g.n_st[g.split == "test"]], bins=20, color=[C_NONTIDAL, C_TIDAL])
    ax[1].set_xlabel("사상당 관측소 수"); ax[1].set_title("관측소는 거의 모두 공유")
    fig.suptitle("그림 7-1  분할 구조: 사상 단위로 완전히 분리", x=0.02, ha="left")
    save(fig, "7-1", "split_structure")
    return {"n_blocks": g.split.value_counts().to_dict(), "stations_per_block_median": float(g.n_st.median())}


def fig_7_2():
    tr = load("train", FEATURES); te = load("test", FEATURES)
    ks = pd.Series({c: stats.ks_2samp(tr[c].sample(100_000, random_state=0), te[c].sample(100_000, random_state=0)).statistic for c in FEATURES}).sort_values()
    adv = load_json("s06_adversarial.json")
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax[0].barh(range(len(ks)), ks.values, color=C_NONTIDAL); ax[0].set_yticks(range(len(ks))); ax[0].set_yticklabels(ks.index, fontsize=6)
    ax[0].axvline(0.1, color=C_ALL, ls="--", lw=1); ax[0].set_xlabel("KS 통계량 (표본 10만 행씩)"); ax[0].set_title("변수별 분포 차이")
    for c, col in [("imerg_rain_1h", CAT[0]), ("radar_reflectivity_ge20_fraction", CAT[2])]:
        for d, ls, lab in [(tr, "-", "train"), (te, "--", "test")]:
            x = np.sort(d[c].sample(50_000, random_state=0)); ax[1].plot(x, np.linspace(0, 1, len(x)), ls, color=col, label=f"{c[:14]} {lab}")
    ax[1].set_xscale("symlog", linthresh=0.1); ax[1].set_ylabel("누적 비율"); ax[1].legend(fontsize=6); ax[1].set_title("차이가 가장 큰 변수의 누적분포")
    fig.suptitle(f"그림 7-2  train과 test의 분포는 거의 같다 (적대적 검증 AUC = {adv['auc_groupkfold']:.2f})", x=0.02, ha="left")
    save(fig, "7-2", "train_test_dist")
    save_table(ks.round(4).to_frame("KS"), "7-1", "ks")
    return {"ks_max": float(ks.max()), "ks_argmax": ks.idxmax(), "auc": adv["auc_groupkfold"]}


def fig_7_3():
    b = pd.read_csv(RESULTS / "s05_baseline.csv")
    ms = ["관측소 평균", "관측소×강우6h구간 평균", "HGB 원변수", "HGB 원변수+차분+관측소통계"]
    fig, ax = plt.subplots(figsize=figsize(16, 6.5))
    x = np.arange(len(ms)); res = {}
    for i, (sp, col) in enumerate([("GroupKFold(사상)", C_NONTIDAL), ("무작위 KFold", C_ALL)]):
        d = b[b.split == sp].set_index("method").loc[ms]
        ax.bar(x + (i - .5) * .38, d.rmse, .38, yerr=d.fold_std, capsize=3, color=col, label=sp)
        res[sp] = d[["rmse", "fold_std"]].round(4).to_dict("index")
    ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=7); ax.set_ylabel("RMSE (m), 오차막대 = fold 표준편차"); ax.legend()
    g, r = res["GroupKFold(사상)"][ms[-1]], res["무작위 KFold"][ms[-1]]
    ax.set_title(f"그림 7-3  무작위 KFold는 {1 - r['rmse'] / g['rmse']:.0%} 낙관적이고 fold 편차를 1/{g['fold_std'] / r['fold_std']:.0f}로 과소평가한다")
    save(fig, "7-3", "cv_comparison")
    return res


def main():
    style()
    tr = load_tb("train_sorted", ["timestamp", "event_block", "pos_in_block", "wl_t", "dwl_1h", "sfc_rain_1h", "wsarea", TARGET])
    res = {"6-1": fig_6_1(tr), "6-2": fig_6_2_3(tr), "6-4": fig_6_4(), "6-5": fig_6_5(tr), "6-6": fig_6_6(tr),
           "7-1": fig_7_1(), "7-2": fig_7_2(), "7-3": fig_7_3()}
    save_json(res, "t06_ts_split.json")
    for k, v in res.items():
        print(k, str(v)[:500])


if __name__ == "__main__":
    main()

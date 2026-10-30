"""X1. 산탄잡음(shot-noise) 커널 변수가 기준 모델을 개선하는지 시험.

수위(유량) Q(t) = Σ_τ r(τ)·g(t−τ)  — 과거 비 펄스마다 응답 커널 g를 붙여 더한 것.
미래 비를 0으로 두면, 이미 내린 비만으로 기대되는 앞으로 s시간의 상승은
    ΔQ_s(t) = Σ_j r(t−j)·[g(j+0.5+s) − g(j+0.5)]
이고, 타깃과 같은 모양의 변수 sn_rise = max_{s=1..6} ΔQ_s 를 만든다.
함께 sn_pend = Σ_j r(t−j)·[G(j+6.5) − G(j+0.5)] (앞으로 6시간 안에 도착할 비의 양)도 만든다.
g는 모양 3의 감마 커널(Nash 저수지 3개). j = 비가 내린 뒤 지난 시간.

변형
  A  기준: 원변수 + 차분 + 관측소 통계 (s05 최고 모델과 같음)
  B  A + 고정 커널 θ ∈ {1,2,4,8}h × {지상, IMERG} × {rise, pend} 16개  → 누수 A등급(행 + 같은 관측소 과거 행)
  C  A + 관측소 커널 (θ를 지체 L에서 정함, L은 학습 fold 안에서만 추정) 4개  → B등급
  D  A + B + C
검증: 사상 GroupKFold(5). 채택 기준: 전체 RMSE 개선 + 5 fold 중 4개 이상 개선 + 상위 5사상 제외 RMSE 악화 없음.
조석·24h 주기·고정 관측소는 커널 변수를 NaN으로 둔다(비와 무관).
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from common import load, FEATURES, TARGET, RESULTS, rmse, save_json, peak_rss

DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]
ST = ["st_mean", "st_std", "st_q95"]
J = 72                      # 커널이 보는 과거 시간 (h)
K = 3                       # 감마 모양 모수
THETAS = [1, 2, 4, 8]       # 고정 커널 척도 (h). 최빈 도달 = 2θ
SRC = {"sfc": ("sfc_rain_1h", "sfc_rain_6h", "sfc_rain_24h"),
       "imerg": ("imerg_rain_1h", "imerg_rain_6h", "imerg_rain_24h")}
NO_RAIN_GROUPS = ["조석", "24h 주기", "고정"]


# ============================================================
# 1. 커널
# ============================================================
def gamma_pdf(a, th):
    a = np.clip(a, 0, None)
    return a ** (K - 1) * np.exp(-a / th) / (2 * th ** K)          # K=3 → 분모 Γ(3)=2


def gamma_cdf(a, th):
    x = np.clip(a, 0, None) / th
    return 1 - np.exp(-x) * (1 + x + x ** 2 / 2)


def kernel_weights(th):
    """과거 j시간(j=0..J-1) 비에 곱할 가중치: rise는 s=1..6별(6×J), pend는 J."""
    age = np.arange(J) + 0.5
    rise = np.stack([gamma_pdf(age + s, th) - gamma_pdf(age, th) for s in range(1, 7)])
    pend = gamma_cdf(age + 6, th) - gamma_cdf(age, th)
    return rise, pend


def theta_from_lag(L):
    """dwl_1h와 비의 상관이 최대가 되는 지체 L = g의 상승 변곡점 θ(2−√2) 로 맞춘다."""
    return (np.asarray(L, dtype=float) + 0.5) / (2 - np.sqrt(2))


# ============================================================
# 2. 과거 시간별 강우 행렬 R (n × J) — 같은 관측소·같은 사상의 과거 행만 사용
# ============================================================
def grid_index(df):
    ts = pd.to_datetime(df.timestamp).to_numpy().astype("datetime64[h]").astype(np.int64)
    g = df.groupby(["station_code", "event_block"], sort=False).ngroup().to_numpy()
    start = pd.Series(ts).groupby(g).transform("min").to_numpy()
    h = ts - start
    size = pd.Series(h).groupby(g).max().to_numpy() + 1
    off = np.concatenate([[0], np.cumsum(size)[:-1]])
    return off[g], h, int(size.sum())


def rain_matrix(df, src, gi):
    c1, c6, c24 = SRC[src]
    base, h, total = gi
    r1, r6, r24 = (df[c].to_numpy(np.float64) for c in (c1, c6, c24))
    flat = np.full(total, np.nan)
    flat[base + h] = r1
    R = np.empty((len(df), J), dtype=np.float32)
    fill_1_5 = np.clip(r6 - r1, 0, None) / 5          # 과거 행이 없을 때: 창 합을 고르게 나눔
    fill_6_23 = np.clip(r24 - r6, 0, None) / 18
    for j in range(J):
        idx = h - j
        ok = idx >= 0
        v = np.full(len(df), np.nan)
        v[ok] = flat[base[ok] + idx[ok]]
        fill = fill_1_5 if 1 <= j <= 5 else fill_6_23 if 6 <= j <= 23 else 0.0
        R[:, j] = np.where(np.isnan(v), fill, v)
    return R


def check_rain_matrix(df, R, src):
    """정합성: 과거 5행이 모두 있으면 R[:,0:6] 합 = 6h 누적이어야 한다."""
    c1, c6, _ = SRC[src]
    s = R[:, :6].sum(1)
    err = np.abs(s - df[c6].to_numpy())
    print(f"  [{src}] |R[0:6] 합 − {c6}| 중앙값 {np.median(err):.4f}, 99% {np.quantile(err, .99):.4f}")


def kernel_feats(R, th):
    rise_w, pend_w = kernel_weights(th)
    rise = (R @ rise_w.T.astype(np.float32)).max(1)
    pend = R @ pend_w.astype(np.float32)
    return rise, pend


# ============================================================
# 3. 관측소 지체 L (학습 fold 안에서만): corr(r(t−L), dwl_1h(t)) 최대, L=0..12
# ============================================================
def station_lag(tr_idx, df, R_sfc, fallback):
    sub = pd.DataFrame({"st": df.station_code.to_numpy()[tr_idx], "y": df.dwl_1h.to_numpy()[tr_idx]})
    best = None
    corrs = {}
    for L in range(13):
        sub["x"] = R_sfc[tr_idx, L]
        g = sub.groupby("st")
        m = g[["x", "y"]].mean()
        cov = (sub.x * sub.y).groupby(sub.st).mean() - m.x * m.y
        sd = np.sqrt(((sub.x ** 2).groupby(sub.st).mean() - m.x ** 2) * ((sub.y ** 2).groupby(sub.st).mean() - m.y ** 2))
        corrs[L] = cov / sd.replace(0, np.nan)
    C = pd.DataFrame(corrs)
    ok = C.notna().any(axis=1)
    Lst = pd.Series(np.nan, index=C.index)
    Lst[ok] = C[ok].idxmax(axis=1)
    return Lst.fillna(fallback)


# ============================================================
# 4. 모델·평가
# ============================================================
def model(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50,
                                         random_state=seed)


def main(rep=0):
    """rep=0: 기본 분할·시드로 A~D. rep≥1: 사상을 섞은 분할 + 모델 시드 rep로 A·C만 (반복 확인)."""
    t0 = time.time()
    df = load("train_sorted", ["station_id", "station_code", "event_block", "timestamp", TARGET] + FEATURES + DWL)
    df = df.reset_index(drop=True)
    reg = pd.read_csv(RESULTS / "nb07_regimes.csv", index_col=0)
    df["group"] = df.station_id.map(reg.group)
    df["regime"] = df.station_id.map(reg.regime)
    no_rain = df.group.isin(NO_RAIN_GROUPS).to_numpy()
    y = df[TARGET].to_numpy(np.float64)

    gi = grid_index(df)
    R = {}
    fixed = []
    for src in SRC:
        R[src] = rain_matrix(df, src, gi)
        check_rain_matrix(df, R[src], src)
        for th in THETAS:
            rise, pend = kernel_feats(R[src], th)
            for nm, v in (("rise", rise), ("pend", pend)):
                col = f"sn_{nm}_{src}_t{th}"
                df[col] = np.where(no_rain, np.nan, v)
                fixed.append(col)
    print(f"고정 커널 변수 {len(fixed)}개 완료 {time.time() - t0:.0f}s", flush=True)
    peak_rss("features")

    # 타깃과의 순위상관 (비조석·비 오는 행) — 참고용
    wet = (~no_rain) & (df.sfc_rain_24h.to_numpy() > 0)
    sp = {c: float(pd.Series(df[c].to_numpy()[wet]).corr(pd.Series(y[wet]), method="spearman")) for c in fixed}
    sp["sfc_rain_6h"] = float(pd.Series(df.sfc_rain_6h.to_numpy()[wet]).corr(pd.Series(y[wet]), method="spearman"))

    STC = ["sn_rise_sfc_st", "sn_pend_sfc_st", "sn_rise_imerg_st", "sn_pend_imerg_st"]
    base = FEATURES + DWL + ST
    variants = {"A": base, "B": base + fixed, "C": base + STC, "D": base + fixed + STC}
    if rep:
        variants = {k: variants[k] for k in ("A", "C")}
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=rep) if rep else GroupKFold(n_splits=5)
    oof = {k: np.zeros(len(df)) for k in variants}
    folds = []
    fallback_L = 1.0                                               # 관측소 L 중앙값(nb09)
    for f, (tri, vai) in enumerate(splitter.split(df, groups=df.event_block)):
        t1 = time.time()
        # 관측소 통계·지체 L은 학습 fold 안에서만
        g = df.iloc[tri].groupby("station_code")[TARGET].agg(st_mean="mean", st_std="std",
                                                           st_q95=lambda s: s.quantile(.95))
        stats_ = df[["station_code"]].join(g, on="station_code")[ST]
        Lst = station_lag(tri, df, R["sfc"], fallback_L)
        th_row = theta_from_lag(df.station_code.map(Lst).fillna(fallback_L).to_numpy())
        stc = {c: np.full(len(df), np.nan) for c in STC}
        for th in np.unique(th_row):
            m = th_row == th
            for src in SRC:
                rise, pend = kernel_feats(R[src][m], th)
                stc[f"sn_rise_{src}_st"][m] = rise
                stc[f"sn_pend_{src}_st"][m] = pend
        X = pd.concat([df[FEATURES + DWL + fixed], stats_, pd.DataFrame(stc)], axis=1)
        for c in STC:
            X.loc[no_rain, c] = np.nan
        r = {}
        for k, feats in variants.items():
            oof[k][vai] = model(rep).fit(X.iloc[tri][feats], y[tri]).predict(X.iloc[vai][feats])
            r[k] = rmse(y[vai], oof[k][vai])
        folds.append(r)
        print(f"fold {f}", {k: round(v, 4) for k, v in r.items()}, f"L 분포 {Lst.describe()[['25%', '50%', '75%']].to_dict()}",
              f"{time.time() - t1:.0f}s", flush=True)

    # ---- 평가 ----
    fd = pd.DataFrame(folds)
    nontidal = ~df.group.isin(["조석", "고정"]).to_numpy()
    sse_ev = pd.Series((y - oof["A"]) ** 2).groupby(df.event_block).sum()
    top5 = sse_ev.nlargest(5).index
    keep5 = ~df.event_block.isin(top5).to_numpy()
    big = y > np.quantile(y, 0.99)
    heavy = df.sfc_rain_6h.to_numpy() >= np.quantile(df.sfc_rain_6h[df.sfc_rain_6h > 0], 0.9)
    ev_row = df.event_block.to_numpy()
    out = {"top5_events": list(map(int, top5)), "spearman_wet_nontidal": sp, "variants": {}}
    for k in variants:
        e2 = (y - oof[k]) ** 2
        e2a = (y - oof["A"]) ** 2
        ev = pd.DataFrame({"e": ev_row, "k": e2, "a": e2a}).groupby("e").mean()
        by_regime = {rg: rmse(y[m], oof[k][m]) for rg in df.regime.unique() for m in [df.regime.to_numpy() == rg]}
        out["variants"][k] = {
            "n_feats": len(variants[k]),
            "rmse": rmse(y, oof[k]),
            "rmse_nontidal": rmse(y[nontidal], oof[k][nontidal]),
            "rmse_excl_top5": rmse(y[keep5], oof[k][keep5]),
            "rmse_top1pct": rmse(y[big], oof[k][big]),
            "rmse_heavy_rain": rmse(y[heavy], oof[k][heavy]),
            "fold_rmse": fd[k].tolist(),
            "folds_better_than_A": int((fd[k] < fd.A).sum()),
            "events_better_than_A_pct": float((ev.k < ev.a).mean() * 100),
            "rmse_by_regime": by_regime,
        }
    save_json(out, f"x01_shot_noise_rep{rep}.json" if rep else "x01_shot_noise.json")
    summ = pd.DataFrame({k: {m: v for m, v in d.items() if not isinstance(v, (list, dict))}
                         for k, d in out["variants"].items()}).T
    print(summ.round(4).to_string())
    print(pd.DataFrame({k: d["rmse_by_regime"] for k, d in out["variants"].items()}).round(4).to_string())
    print("타깃 순위상관(비조석·비 온 행) 상위:",
          dict(sorted(sp.items(), key=lambda kv: -abs(kv[1]))[:6]))
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    for r in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else [0]):
        main(r)

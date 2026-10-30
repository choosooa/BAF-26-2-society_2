"""X2. 랜덤워크 러닝 최대값 공식(Spitzer) 변수 + 교차항(q / log 상류면적) 시험.

수위 걸음 X_i ~ N(μ, σ²) 독립이라 두면, S_k = X_1+…+X_k 에 대해
    E[max_{1≤k≤6} S_k] = μ + Σ_{k=1}^{5} E[S_k⁺]/k        (첫 걸음을 떼면 나머지는 Spitzer 항등식)
    E[S_k⁺] = m·Φ(m/s) + s·φ(m/s),  m = kμ, s = √k·σ
μ: dwl_1h (rm1) 또는 dwl_3h/3 (rm3). σ: 관측소 dwl_1h 표준편차(학습 fold 안). → B등급
q: 관측소 학습 fold 수위 분포에서 wl_t 의 분위. → B등급

변형
  A   기준: 원변수 + 차분 + 관측소 통계 (s05 최고 모델)
  E   A + rm1, rm3
  Fq  E + rm1·q, rm3·q
  Fa  E + rm1·log1p(wsarea), rm3·log1p(wsarea)
실행: python x02_running_max.py 0 1 2 3   (0 = 기본 분할·시드 0, r≥1 = 사상 섞은 분할 + 시드 r)
      python x02_running_max.py s1 s2 s3  (간이 확인: 사상 40개 무작위 표본 × 사상 4-fold, HGB 150회. 숫자 = 표본 시드)
채택: 20 fold 합계에서 대부분(≥ 4/5 비율) 개선 + 평균 RMSE 개선 + 상위 5사상 제외 악화 없음.
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from common import load, FEATURES, TARGET, RESULTS, rmse, save_json

DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]
ST = ["st_mean", "st_std", "st_q95"]
RM = ["rm1", "rm3"]
XQ = ["rm1_q", "rm3_q"]
XA = ["rm1_lws", "rm3_lws"]
H = 6


# ============================================================
# 1. 러닝 최대값 기댓값
# ============================================================
def expected_running_max(mu, sigma, n=H):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    out = mu.copy()
    for k in range(1, n):
        m, s = k * mu, np.sqrt(k) * sigma
        z = m / s
        out += (m * norm.cdf(z) + s * norm.pdf(z)) / k
    return out


def check_formula():
    rng = np.random.default_rng(0)
    for mu, sg in [(0.0, 0.05), (0.03, 0.05), (-0.03, 0.05), (0.1, 0.02)]:
        S = np.cumsum(rng.normal(mu, sg, size=(200_000, H)), axis=1)
        print(f"  μ={mu:+.2f} σ={sg:.2f}: 공식 {expected_running_max([mu], [sg])[0]:.4f} vs 모의 {S.max(1).mean():.4f}")


# ============================================================
# 2. fold 안 관측소 통계: 타깃 통계, σ, q
# ============================================================
def fold_station_feats(df, tri):
    tr = df.iloc[tri]
    g = tr.groupby("station_code")
    st = g[TARGET].agg(st_mean="mean", st_std="std", st_q95=lambda s: s.quantile(.95))
    sig = g["dwl_1h"].std().rename("sigma")
    stats_ = df[["station_code"]].join(st, on="station_code").join(sig, on="station_code")
    # q: 관측소별 학습 수위의 경험 분포
    q = np.full(len(df), np.nan)
    wl = df.wl_t.to_numpy()
    codes = df.station_code.to_numpy()
    tr_sorted = {c: np.sort(v) for c, v in tr.groupby("station_code").wl_t}
    for c, idx in pd.Series(np.arange(len(df))).groupby(codes):
        ref = tr_sorted.get(c)
        if ref is not None and len(ref):
            q[idx.to_numpy()] = np.searchsorted(ref, wl[idx.to_numpy()], side="right") / len(ref)
    stats_["q"] = q
    return stats_


N_SAMPLE_EVENTS = 40
MAX_ITER = 300


def model(seed=0):
    return HistGradientBoostingRegressor(max_iter=MAX_ITER, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50,
                                         random_state=seed)


# ============================================================
# 3. 한 회차 실행
# ============================================================
def run(df, rep, n_splits=5):
    y = df[TARGET].to_numpy(np.float64)
    lws = np.log1p(df.wsarea.to_numpy())
    base = FEATURES + DWL + ST
    variants = {"A": base, "E": base + RM, "Fq": base + RM + XQ, "Fa": base + RM + XA}
    oof = {k: np.zeros(len(df)) for k in variants}
    splitter = GroupKFold(n_splits, shuffle=True, random_state=rep) if rep else GroupKFold(n_splits)
    folds = []
    for f, (tri, vai) in enumerate(splitter.split(df, groups=df.event_block)):
        t1 = time.time()
        s = fold_station_feats(df, tri)
        X = df[FEATURES + DWL].copy()
        X[ST] = s[ST].to_numpy()
        X["rm1"] = expected_running_max(df.dwl_1h, s.sigma.fillna(s.sigma.median()))
        X["rm3"] = expected_running_max(df.dwl_3h / 3, s.sigma.fillna(s.sigma.median()))
        X["rm1_q"], X["rm3_q"] = X.rm1 * s.q.to_numpy(), X.rm3 * s.q.to_numpy()
        X["rm1_lws"], X["rm3_lws"] = X.rm1 * lws, X.rm3 * lws
        r = {}
        for k, feats in variants.items():
            oof[k][vai] = model(rep).fit(X.iloc[tri][feats], y[tri]).predict(X.iloc[vai][feats])
            r[k] = rmse(y[vai], oof[k][vai])
        folds.append(r)
        print(f"  rep{rep} fold {f}", {k: round(v, 4) for k, v in r.items()}, f"{time.time() - t1:.0f}s", flush=True)
    return oof, pd.DataFrame(folds), variants


def evaluate(df, oof, fd, variants, rep, tag="rep"):
    y = df[TARGET].to_numpy(np.float64)
    nontidal = ~df.group.isin(["조석", "고정"]).to_numpy()
    top5 = pd.Series((y - oof["A"]) ** 2).groupby(df.event_block).sum().nlargest(5).index
    keep5 = ~df.event_block.isin(top5).to_numpy()
    big = y > np.quantile(y, 0.99)
    ev_row = df.event_block.to_numpy()
    res = {}
    for k in variants:
        ev = pd.DataFrame({"e": ev_row, "k": (y - oof[k]) ** 2, "a": (y - oof["A"]) ** 2}).groupby("e").mean()
        res[k] = {
            "rmse": rmse(y, oof[k]),
            "rmse_nontidal": rmse(y[nontidal], oof[k][nontidal]),
            "rmse_excl_top5": rmse(y[keep5], oof[k][keep5]),
            "rmse_top1pct": rmse(y[big], oof[k][big]),
            "folds_better": int((fd[k] < fd.A).sum()),
            "events_better_pct": float((ev.k < ev.a).mean() * 100),
            "fold_rmse": fd[k].tolist(),
            "rmse_by_regime": {rg: rmse(y[m], oof[k][m]) for rg in df.regime.unique()
                               for m in [df.regime.to_numpy() == rg]},
        }
    save_json({"rep": rep, "top5_events": list(map(int, top5)), "variants": res}, f"x02_running_max_{tag}{rep}.json")
    return res


def main(reps, sample=False):
    global MAX_ITER
    tag = "sample" if sample else "rep"
    if sample:
        MAX_ITER = 150
    t0 = time.time()
    print("공식 점검 (몬테카를로 20만 회)")
    check_formula()
    full = load("train_sorted", ["station_id", "station_code", "event_block", TARGET] + FEATURES + DWL).reset_index(drop=True)
    reg = pd.read_csv(RESULTS / "nb07_regimes.csv", index_col=0)
    full["group"] = full.station_id.map(reg.group)
    full["regime"] = full.station_id.map(reg.regime)

    rows = []
    for rep in reps:
        df = full
        if sample:
            evs = np.random.default_rng(rep).choice(full.event_block.unique(), N_SAMPLE_EVENTS, replace=False)
            df = full[full.event_block.isin(evs)].reset_index(drop=True)
            print(f"표본 {rep}: 사상 {N_SAMPLE_EVENTS}개, {len(df):,}행", flush=True)
        oof, fd, variants = run(df, rep, n_splits=4 if sample else 5)
        res = evaluate(df, oof, fd, variants, rep, tag)
        for k, d in res.items():
            rows.append({"rep": rep, "variant": k, **{m: v for m, v in d.items() if not isinstance(v, (list, dict))}})
        print(pd.DataFrame(rows).query("rep == @rep").set_index("variant").drop(columns="rep").round(4).to_string(), flush=True)

    t = pd.DataFrame(rows)
    a = t[t.variant == "A"].set_index("rep")
    t["d_rmse"] = t.apply(lambda r: r.rmse - a.loc[r.rep, "rmse"], axis=1)
    t["d_excl_top5"] = t.apply(lambda r: r.rmse_excl_top5 - a.loc[r.rep, "rmse_excl_top5"], axis=1)
    t["d_nontidal"] = t.apply(lambda r: r.rmse_nontidal - a.loc[r.rep, "rmse_nontidal"], axis=1)
    agg = t.groupby("variant").agg(rmse=("rmse", "mean"), d_rmse=("d_rmse", "mean"), d_nontidal=("d_nontidal", "mean"),
                                   d_excl_top5=("d_excl_top5", "mean"), folds_better=("folds_better", "sum"),
                                   reps_better=("d_rmse", lambda s: int((s < 0).sum())),
                                   events_better_pct=("events_better_pct", "mean"))
    agg["d_rmse_pct"] = agg.d_rmse / t[t.variant == "A"].rmse.mean() * 100
    print("\n=== 회차 합계 ===")
    print(agg.round(5).to_string())
    save_json({"reps": list(reps), "per_rep": rows, "summary": agg.reset_index().to_dict("records")},
              f"x02_running_max_{tag}_summary.json")
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    smp = bool(args) and args[0].startswith("s")
    main([int(a.lstrip("s")) for a in args] or [0], sample=smp)

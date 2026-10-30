"""S9. 탐색: 좌표 없이 관측소 간 선행-후행(상류→하류 추정) 연결망을 데이터만으로 복원.

같은 합성시각 안에서 관측소들의 시간은 정렬돼 있다(같은 격자 강우 시계열이 시각별로 일치 — S2).
관측소 i의 '지금' 수위변화가 관측소 j의 L시간 뒤 수위변화와 얼마나 같이 움직이는지를 전 쌍에 대해 계산한다.
"""
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from common import (load, FEATURES, TARGET, RESULTS, CACHE, CAT, INK2, MUTED, SEQ, ENV,
                    savefig, save_json, peak_rss, style, rmse)

style()
RES = {}
MAXLAG = 8
DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]


def station_matrix(df, n_st):
    hours = np.sort(df.hour.unique())
    hidx = pd.Series(np.arange(len(hours)), index=hours)
    Z = np.full((len(hours), n_st), np.nan)
    x = df.dwl_1h.to_numpy(np.float64)
    code = df.station_code.to_numpy()
    # 관측소별 강건 척도: 조석처럼 큰 관측소가 상관을 지배하지 않도록
    g = pd.Series(x).groupby(code)
    mad = g.apply(lambda v: np.median(np.abs(v - np.median(v)))) * 1.4826
    scale = np.maximum(np.maximum(mad, 0.5 * g.std()), 0.005)
    z = np.clip(x / scale.reindex(code).to_numpy(), -8, 8)
    Z[hidx.reindex(df.hour.to_numpy()).to_numpy(), code] = z
    Z = Z - np.nanmedian(Z, axis=1, keepdims=True)  # 전 관측소 공통 흐름(호우 전체 신호) 제거
    return Z, hours


def masked_corr(A, B, min_n=150):
    MA, MB = ~np.isnan(A), ~np.isnan(B)
    A0, B0 = np.where(MA, A, 0.0), np.where(MB, B, 0.0)
    fa, fb = MA.astype(float), MB.astype(float)
    n = fa.T @ fb
    sa, sb = A0.T @ fb, fa.T @ B0
    saa, sbb, sab = (A0 ** 2).T @ fb, fa.T @ (B0 ** 2), A0.T @ B0
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / n
        C = cov / np.sqrt((saa - sa ** 2 / n) * (sbb - sb ** 2 / n))
    C[n < min_n] = np.nan
    return C


def lagged_corrs(Z, hours, row_ok=None):
    if row_ok is not None:
        Z = np.where(row_ok[:, None], Z, np.nan)
    Cs = {}
    for L in range(0, MAXLAG + 1):
        i = np.arange(len(hours) - L)
        ok = hours[i + L] - hours[i] == L  # 같은 사상 안에서만 짝짓기
        Cs[L] = masked_corr(Z[i[ok]], Z[i[ok] + L])  # Cs[L][a, b] = corr(a(t), b(t+L))
    return Cs


def leaders(Cs, exclude):
    C0 = Cs[0].copy()
    stack = np.stack([Cs[L] for L in range(1, MAXLAG + 1)])  # (L, leader, follower)
    stack = np.where(np.isnan(stack), -np.inf, stack)
    best = stack.max(axis=0)
    lag = stack.argmax(axis=0) + 1
    n = best.shape[0]
    best[np.arange(n), np.arange(n)] = -np.inf
    best[exclude, :] = -np.inf
    gain = best - np.nan_to_num(C0, nan=0.0)
    cand = np.where(gain > 0.05, best, -np.inf)  # 동시 상관보다 '앞서서' 더 잘 맞아야 선행
    lead = cand.argmax(axis=0)
    val = cand.max(axis=0)
    out = pd.DataFrame({"leader": lead, "corr": val, "lag": lag[lead, np.arange(n)],
                        "corr0": C0[lead, np.arange(n)]})
    out.loc[~np.isfinite(out["corr"]) | np.isin(np.arange(n), exclude), ["leader", "corr", "lag", "corr0"]] = np.nan
    twin = np.where(np.eye(n, dtype=bool), -np.inf, np.nan_to_num(C0, nan=-np.inf))
    out["twin"] = twin.argmax(axis=0)
    out["twin_corr0"] = twin.max(axis=0)
    return out


def main():
    t0 = time.time()
    cols = ["station_id", "station_code", "hour", "event_block", TARGET] + FEATURES + DWL
    df = load("train_sorted", cols)
    fp = pd.read_csv(RESULTS / "s08_station_fingerprints.csv", index_col=0)
    cats = sorted(df.station_id.astype(str).unique())
    code_of = {s: i for i, s in enumerate(cats)}
    per = RESULTS / "s08_fingerprint.json"
    j = json.load(open(per, encoding="utf-8"))
    special = set(j["periodic"]["strong_tidal"] + j["periodic"]["weak_tidal"] + j["periodic"]["daily_only"]
                  + j["resolution"]["never_changes"])
    exclude = np.array(sorted(code_of[s] for s in special))
    n_st = len(cats)
    Z, hours = station_matrix(df, n_st)
    Cs = lagged_corrs(Z, hours)
    L = leaders(Cs, exclude)
    L.index = cats
    L["leader_id"] = [cats[int(v)] if np.isfinite(v) else None for v in L.leader]
    L["twin_id"] = [cats[int(v)] for v in L.twin]
    env = fp[ENV]
    strong = L[(L["corr"] >= 0.5)].copy()
    strong["ws_f"] = env.wsarea.reindex(strong.index).to_numpy()
    strong["ws_l"] = env.wsarea.reindex(strong.leader_id).to_numpy()
    strong["so_f"] = env.stream_order.reindex(strong.index).to_numpy()
    strong["so_l"] = env.stream_order.reindex(strong.leader_id).to_numpy()
    same_env = (env.reindex(strong.index).round(4).to_numpy() == env.reindex(strong.leader_id).round(4).to_numpy()).all(axis=1)
    diff = ~same_env
    recip = sum(1 for f, r in strong.iterrows() if r.leader_id in strong.index and strong.loc[r.leader_id, "leader_id"] == f)
    RES["network"] = {
        "stations_considered": int(n_st - len(exclude)),
        "with_leader_any": int(L["corr"].notna().sum()),
        "strong_leader(corr>=0.5)": int(len(strong)),
        "strong_share": float(len(strong) / (n_st - len(exclude))),
        "lag_counts": strong.lag.value_counts().sort_index().astype(int).to_dict(),
        "corr_quantiles": strong["corr"].quantile([.1, .5, .9]).round(3).to_dict(),
        "gain_over_lag0_median": float((strong["corr"] - strong.corr0).median()),
        "follower_larger_area_share(diff_env)": float((strong.ws_f[diff] > strong.ws_l[diff]).mean()),
        "follower_higher_order_share(diff_env)": float((strong.so_f[diff] > strong.so_l[diff]).mean()),
        "pairs_same_subbasin": int(same_env.sum()),
        "reciprocal_pairs": int(recip // 2),
        "twin_corr0_quantiles": L.twin_corr0.drop(index=[cats[i] for i in exclude]).quantile([.1, .5, .9]).round(3).to_dict(),
        "hubs(top leaders)": strong.leader_id.value_counts().head(8).to_dict(),
    }
    L.to_csv(RESULTS / "s09_leaders.csv")
    print(json.dumps(RES["network"], ensure_ascii=False, indent=1), f"{time.time()-t0:.0f}s", flush=True)

    fig_network(strong, fp)
    fig_example(df, strong, Cs, code_of)

    # ---- 발견의 예측 가치(가볍게): 사상 단위 GroupKFold, 선행 관측소는 학습 fold 시각만으로 선정 ----
    value_check(df, Z, hours, exclude, cats, code_of)
    save_json(RES, "s09_network.json")
    peak_rss("s09")


def fig_network(strong, fp):
    G = nx.DiGraph()
    for f, r in strong.iterrows():
        G.add_edge(r.leader_id, f, lag=int(r.lag), w=float(r["corr"]))
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    RES["network"]["components"] = {"n": len(comps), "sizes_top10": [len(c) for c in comps[:10]]}
    H = G.subgraph(set().union(*comps[:12]))
    pos = nx.spring_layout(H.to_undirected(), seed=3, k=0.35, iterations=200)
    fig, ax = plt.subplots(figsize=(12, 8))
    lag_col = {1: SEQ[2], 2: SEQ[3], 3: SEQ[4], 4: SEQ[5]}
    for u, v, d in H.edges(data=True):
        c = lag_col.get(d["lag"], SEQ[6])
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=0.6 + 1.6 * (d["w"] - 0.5) / 0.5, shrinkA=4, shrinkB=4))
    ws = fp.wsarea.reindex(list(H.nodes())).fillna(10)
    xy = np.array([pos[n] for n in H.nodes()])
    ax.scatter(xy[:, 0], xy[:, 1], s=12 + 18 * np.log10(ws.to_numpy() + 1), color="#b4b2a9", edgecolor="#fcfcfb", lw=0.6, zorder=3)
    hubs = [n for n, d in H.out_degree() if d >= 3]
    for n in hubs:
        ax.annotate(n, pos[n], fontsize=7.5, color=INK2, xytext=(4, 4), textcoords="offset points", zorder=4)
    ax.set_axis_off()
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"{k}h 앞섬") for k, c in lag_col.items()] + \
              [plt.Line2D([], [], color=SEQ[6], lw=2, label="5h 이상")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, title="화살표 = 선행 → 후행", title_fontsize=8)
    ax.set_title(f"데이터로 복원한 관측소 선행-후행 연결망 (상위 12개 덩어리, 노드 크기 = 유역면적)", fontsize=11)
    savefig(fig, "09a_station_network.png")


def fig_example(df, strong, Cs, code_of):
    cand = strong[(strong.lag >= 2)].sort_values("corr", ascending=False)
    if cand.empty:
        return
    f, r = cand.index[0], cand.iloc[0]
    lid, lag = r.leader_id, int(r.lag)
    d = df[df.station_id.astype(str).isin([f, lid])]
    b = d[d.station_id.astype(str) == f].groupby("event_block").dwl_1h.apply(lambda v: v.abs().sum()).idxmax()
    d = d[d.event_block == b]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    for ax, shift in zip(axes, [0, lag]):
        for sid, col, lbl in [(lid, CAT[0], f"선행 {lid}"), (f, CAT[1], f"후행 {f}")]:
            dd = d[d.station_id.astype(str) == sid].sort_values("hour")
            t = dd.hour - dd.hour.min() + (shift if sid == lid else 0)
            yv = (dd.wl_t - dd.wl_t.min()) / max(dd.wl_t.max() - dd.wl_t.min(), 0.01)
            ax.plot(t, yv, color=col, lw=2, label=lbl + (f" (+{shift}h 이동)" if sid == lid and shift else ""))
        ax.set_ylabel("수위 (사상 내 0~1 정규화)"); ax.legend(fontsize=8, loc="upper left")
    axes[0].set_title(f"선행 관측소의 움직임을 {lag}시간 늦추면 후행 관측소와 겹친다 (상관 {r['corr']:.2f})")
    axes[1].set_xlabel(f"사상 {b} 시작 후 시간 (h)")
    fig.tight_layout(); savefig(fig, "09b_lead_lag_example.png")
    RES["example_pair"] = {"leader": lid, "follower": f, "lag": lag, "corr": float(r["corr"]), "block": int(b)}


def value_check(df, Z, hours, exclude, cats, code_of):
    y = df[TARGET].to_numpy(np.float64)
    key = df.station_code.to_numpy(np.int64) * 1_000_000 + df.hour.to_numpy(np.int64)
    vals = {c: pd.Series(df[c].to_numpy(np.float64), index=key) for c in ["dwl_1h", "dwl_3h", "sfc_rain_1h", "imerg_rain_6h"]}
    # 지역(동시각 전 관측소) 요약 — 입력값만 사용, test에서도 같은 시각 행으로 계산 가능
    hr = df.groupby("hour").agg(reg_imerg6=("imerg_rain_6h", "mean"), reg_sfc1=("sfc_rain_1h", "mean"),
                                reg_wet=("imerg_rain_6h", lambda v: (v > 1).mean()),
                                reg_rise=("dwl_1h", lambda v: (v > 0.02).mean()))
    for c in hr.columns:
        df[c] = df.hour.map(hr[c]).astype(np.float32)
    hmap = pd.Series(df.event_block.to_numpy(), index=df.hour.to_numpy()).groupby(level=0).first()
    blocks_of_hours = hmap.reindex(hours).to_numpy()
    base_feats = FEATURES + DWL
    net_feats = ["lead_dwl1", "lead_dwl3", "lead_sfc1", "lead_imerg6", "lead_lag", "lead_corr", "reg_imerg6", "reg_sfc1", "reg_wet", "reg_rise"]
    oof = {"base": np.zeros(len(df)), "net": np.zeros(len(df))}
    tidal_ids = set(pd.read_csv(RESULTS / "s04_station_summary.csv", index_col=0).query("acf12 > 0.6 and acf6 < 0").index)
    is_tidal = df.station_id.astype(str).isin(tidal_ids).to_numpy()
    for fold, (tri, vai) in enumerate(GroupKFold(5).split(df, groups=df.event_block)):
        tr_blocks = set(df.event_block.iloc[tri].unique())
        row_ok = np.isin(blocks_of_hours, list(tr_blocks))
        Ld = leaders(lagged_corrs(Z, hours, row_ok), exclude)
        Ld = Ld[Ld["corr"] >= 0.5]
        lead_code = pd.Series(Ld.leader.to_numpy(), index=Ld.index).reindex(np.arange(len(cats)))
        lc = lead_code.reindex(df.station_code.to_numpy()).to_numpy()
        ok = np.isfinite(lc)
        lkey = np.where(ok, np.nan_to_num(lc).astype(np.int64) * 1_000_000 + df.hour.to_numpy(np.int64), -1)
        f = df.copy()
        for c, nm in [("dwl_1h", "lead_dwl1"), ("dwl_3h", "lead_dwl3"), ("sfc_rain_1h", "lead_sfc1"), ("imerg_rain_6h", "lead_imerg6")]:
            f[nm] = np.where(ok, vals[c].reindex(lkey).to_numpy(), np.nan)
        f["lead_lag"] = pd.Series(Ld.lag.to_numpy(), index=Ld.index).reindex(df.station_code.to_numpy()).to_numpy()
        f["lead_corr"] = pd.Series(Ld["corr"].to_numpy(), index=Ld.index).reindex(df.station_code.to_numpy()).to_numpy()
        tr, va = f.iloc[tri], f.iloc[vai]
        st = tr.groupby("station_code")[TARGET].agg(st_mean="mean", st_std="std", st_q95=lambda s: s.quantile(.95))
        tr, va = tr.join(st, on="station_code"), va.join(st, on="station_code")
        for name, feats in [("base", base_feats + ["st_mean", "st_std", "st_q95"]),
                            ("net", base_feats + ["st_mean", "st_std", "st_q95"] + net_feats)]:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, random_state=0)
            m.fit(tr[feats], tr[TARGET])
            oof[name][vai] = m.predict(va[feats])
        print(f"fold {fold}: base {rmse(y[vai], oof['base'][vai]):.4f}  net {rmse(y[vai], oof['net'][vai]):.4f}", flush=True)
    local_dry = ((df.imerg_rain_24h < 1) & (df.sfc_rain_24h < 1)).to_numpy()
    big = y > 0.5
    subsets = {"전체": np.ones(len(y), bool), "비조석": ~is_tidal, "조석": is_tidal,
               "비조석·큰 상승(>0.5m)": ~is_tidal & big, "비조석·큰 상승·지역 무강우": ~is_tidal & big & local_dry}
    RES["value_check"] = {k: {"n": int(m.sum()), "rmse_base": rmse(y[m], oof["base"][m]), "rmse_net": rmse(y[m], oof["net"][m])}
                          for k, m in subsets.items()}
    RES["big_rise_local_dry_share"] = float((big & local_dry & ~is_tidal).sum() / (big & ~is_tidal).sum())
    pd.DataFrame({"station_code": df.station_code.to_numpy(), "hour": df.hour.to_numpy(),
                  "oof_base": oof["base"], "oof_net": oof["net"]}).to_parquet(CACHE / "oof_s09.parquet", index=False)
    print(json.dumps(RES["value_check"], ensure_ascii=False, indent=1), RES["big_rise_local_dry_share"])


if __name__ == "__main__":
    main()

"""S5. 기준선 RMSE: 사상 단위 GroupKFold vs 무작위 KFold."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from common import load, FEATURES, TARGET, RESULTS, rmse, save_json, peak_rss

DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]
RAIN_BINS = [-np.inf, 0, 1, 5, 20, np.inf]


def pred_station_mean(tr, va):
    m = tr.groupby("station_code")[TARGET].mean()
    return va["station_code"].map(m).fillna(tr[TARGET].mean()).to_numpy()


def pred_station_rain(tr, va):
    k_tr = tr["station_code"].astype(str) + "_" + pd.cut(tr.imerg_rain_6h, RAIN_BINS, labels=False).astype(str)
    k_va = va["station_code"].astype(str) + "_" + pd.cut(va.imerg_rain_6h, RAIN_BINS, labels=False).astype(str)
    g = tr.groupby(k_tr)[TARGET].agg(["mean", "size"])
    m = g["mean"].where(g["size"] >= 20)
    p = k_va.map(m)
    return p.fillna(pd.Series(pred_station_mean(tr, va), index=va.index)).to_numpy()


def hgb(feats, station_stats=False):
    def f(tr, va):
        feats_ = list(feats)
        if station_stats:  # 관측소 범주(465)는 HGB 한도(255) 초과 → fold 내부에서 계산한 관측소 타깃 통계로 대체
            g = tr.groupby("station_code")[TARGET].agg(st_mean="mean", st_std="std", st_q95=lambda s: s.quantile(.95))
            tr = tr.join(g, on="station_code"); va = va.join(g, on="station_code")
            feats_ += ["st_mean", "st_std", "st_q95"]
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50,
                                              random_state=0)
        model.fit(tr[feats_], tr[TARGET])
        return model.predict(va[feats_])
    return f


def main():
    tidal = pd.read_csv(RESULTS / "s04_station_summary.csv", index_col=0)
    tidal_ids = set(tidal.index[(tidal.acf12 > 0.6) & (tidal.acf6 < 0)])
    df = load("train_sorted", ["station_id", "station_code", "event_block", TARGET] + FEATURES + DWL)
    df["is_tidal"] = df.station_id.astype(str).isin(tidal_ids)
    y = df[TARGET].to_numpy(np.float64)

    methods = {
        "0 예측": lambda tr, va: np.zeros(len(va)),
        "전체 평균": lambda tr, va: np.full(len(va), tr[TARGET].mean()),
        "관측소 평균": pred_station_mean,
        "관측소×강우6h구간 평균": pred_station_rain,
        "HGB 원변수": hgb(FEATURES),
        "HGB 원변수+차분+관측소통계": hgb(FEATURES + DWL, station_stats=True),
    }
    splitters = {"GroupKFold(사상)": GroupKFold(n_splits=5), "무작위 KFold": KFold(n_splits=5, shuffle=True, random_state=0)}
    rows, oof_store = [], {}
    for sname, sp in splitters.items():
        for mname, fn in methods.items():
            if sname == "무작위 KFold" and mname in ("0 예측", "전체 평균"):
                continue
            t0 = time.time()
            oof = np.zeros(len(df))
            fold_rmse = []
            for tri, vai in sp.split(df, groups=df.event_block):
                p = fn(df.iloc[tri], df.iloc[vai])
                oof[vai] = p
                fold_rmse.append(rmse(y[vai], p))
            r = {"split": sname, "method": mname, "rmse": rmse(y, oof), "fold_std": float(np.std(fold_rmse)),
                 "rmse_tidal": rmse(y[df.is_tidal], oof[df.is_tidal]), "rmse_nontidal": rmse(y[~df.is_tidal], oof[~df.is_tidal]),
                 "sec": round(time.time() - t0, 1)}
            rows.append(r); print(r, flush=True)
            if sname.startswith("Group"):
                oof_store[mname] = oof
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "s05_baseline.csv", index=False)

    # SSE 기여 분해 (최고 모델, GroupKFold)
    best = res[res.split.str.startswith("Group")].sort_values("rmse").iloc[0]["method"]
    e2 = (y - oof_store[best]) ** 2
    extra = {"best_group_model": best,
             "best_sse_share_tidal": float(e2[df.is_tidal].sum() / e2.sum()),
             "best_rmse_if_tidal_perfect": float(np.sqrt(e2[~df.is_tidal].sum() / len(e2)))}
    save_json(extra, "s05_extra.json"); print(extra)
    peak_rss("s05")


if __name__ == "__main__":
    main()

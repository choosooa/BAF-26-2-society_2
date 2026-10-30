"""보조 타깃(3시간 뒤 최대 상승량) 스태킹이 본 타깃 CV를 개선하는지 비교.

A: 기준 HGB (원변수+차분+관측소통계)
B: A + 보조모델 예측(aux3). 보조 타깃 y3 = max(wl(t+1..t+3)) - wl(t), train 행끼리만 계산.
   학습 fold 행의 aux3는 내부 GroupKFold OOF, 검증 fold 행은 학습 fold 전체로 적합한 보조모델로 예측.
   검증 fold(=가상의 test 사상)의 수위는 보조 학습에 전혀 쓰지 않는다.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from common import load, FEATURES, TARGET, RESULTS, rmse

DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]
ST = ["st_mean", "st_std", "st_q95"]


def add_y3(df):
    ts = pd.to_datetime(df.timestamp)
    key = df[["station_code", "event_block"]].copy()
    key["ts"] = ts
    lut = pd.Series(df.wl_t.to_numpy(), index=pd.MultiIndex.from_frame(key))
    fut = []
    for k in (1, 2, 3):
        k2 = key.copy(); k2["ts"] = ts + pd.Timedelta(hours=k)
        fut.append(lut.reindex(pd.MultiIndex.from_frame(k2)).to_numpy())
    fut = np.vstack(fut)
    y3 = np.nanmax(np.where(np.isnan(fut), -np.inf, fut), axis=0) - df.wl_t.to_numpy()
    y3[np.all(np.isnan(fut), axis=0)] = np.nan
    return y3


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                         min_samples_leaf=50, random_state=0)


def st_stats(tr):
    return tr.groupby("station_code")[TARGET].agg(st_mean="mean", st_std="std", st_q95=lambda s: s.quantile(.95))


def fit_aux(tr, feats):
    m = tr.y3.notna()
    return model().fit(tr.loc[m, feats], tr.loc[m, "y3"])


def main():
    df = load("train_sorted", ["station_id", "station_code", "event_block", "timestamp", TARGET] + FEATURES + DWL)
    df = df.reset_index(drop=True)
    df["y3"] = add_y3(df)
    print("y3 결측률", df.y3.isna().mean().round(3), "| corr(y3, y6)", df[["y3", TARGET]].corr().iloc[0, 1].round(3))
    y = df[TARGET].to_numpy(np.float64)
    base = FEATURES + DWL
    oof = {"A": np.zeros(len(df)), "B": np.zeros(len(df)), "aux_only": np.zeros(len(df))}
    folds = []
    for f, (tri, vai) in enumerate(GroupKFold(4).split(df, groups=df.event_block)):
        t0 = time.time()
        tr, va = df.iloc[tri].copy(), df.iloc[vai].copy()
        g = st_stats(tr)
        tr, va = tr.join(g, on="station_code"), va.join(g, on="station_code")
        feats = base + ST
        # A
        oof["A"][vai] = model().fit(tr[feats], tr[TARGET]).predict(va[feats])
        # B: 학습 fold 안에서 보조 OOF (관측소 통계도 내부 fold 기준으로 새로 계산해 과적합 방지)
        tr["aux3"] = np.nan
        for itr, iva in GroupKFold(3).split(tr, groups=tr.event_block):
            a, b = tr.iloc[itr].drop(columns=ST), tr.iloc[iva].drop(columns=ST)
            gi = st_stats(a); a, b = a.join(gi, on="station_code"), b.join(gi, on="station_code")
            tr.iloc[iva, tr.columns.get_loc("aux3")] = fit_aux(a, feats).predict(b[feats])
        va["aux3"] = fit_aux(tr, feats).predict(va[feats])
        oof["aux_only"][vai] = va.aux3
        fb = feats + ["aux3"]
        oof["B"][vai] = model().fit(tr[fb], tr[TARGET]).predict(va[fb])
        r = {k: rmse(y[vai], oof[k][vai]) for k in ("A", "B")}
        folds.append(r); print(f, r, round(time.time() - t0), "s", flush=True)
    fd = pd.DataFrame(folds)
    big = y > np.quantile(y, 0.99)
    res = pd.DataFrame({
        "rmse": {k: rmse(y, oof[k]) for k in ("A", "B")},
        "rmse_top1pct": {k: rmse(y[big], oof[k][big]) for k in ("A", "B")},
        "fold_std": fd.std(),
    })
    res["folds_B_better"] = int((fd.B < fd.A).sum())
    # 사상별 개선 비율
    ev = pd.DataFrame({"e": df.event_block, "a": (y - oof["A"]) ** 2, "b": (y - oof["B"]) ** 2}).groupby("e").mean()
    res["events_B_better_pct"] = round((ev.b < ev.a).mean() * 100, 1)
    print(res)
    res.to_csv(RESULTS / "t09_aux_target.csv")


if __name__ == "__main__":
    main()

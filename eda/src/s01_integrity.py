"""S1. 정합성 검산 + 사상(event_block)/run 복원.

key = station_code * 10^6 + 합성시각(시간 단위 정수) 로 (관측소, 시각) → 값 조회를 벡터화한다.
"""
import numpy as np
import pandas as pd

from common import load, CACHE, WL_LAGS, TARGET, save_json, peak_rss

TOL = 1e-3
BASE = pd.Timestamp("2100-01-01")


def add_keys(df, cats):
    df["station_code"] = pd.Categorical(df["station_id"].astype(str), categories=cats).codes.astype(np.int64)
    df["hour"] = ((df["timestamp"] - BASE) // pd.Timedelta(hours=1)).astype(np.int64)
    df["key"] = df["station_code"] * 1_000_000 + df["hour"]
    df.sort_values(["station_code", "hour"], inplace=True, ignore_index=True)
    return df


def lookup(df, col, dh):
    """같은 관측소, hour+dh 행의 col 값 (없으면 NaN)."""
    s = pd.Series(df[col].to_numpy(np.float64), index=df["key"].to_numpy())
    return s.reindex(df["key"].to_numpy() + dh).to_numpy()


def event_blocks(train_ts, test_ts):
    """전역 합성시각을 train/test 라벨과 함께 정렬, 1h 초과 공백 또는 라벨 전환에서 새 블록."""
    t = pd.DataFrame({"hour": np.concatenate([train_ts, test_ts]),
                      "split": ["train"] * len(train_ts) + ["test"] * len(test_ts)})
    t = t.sort_values("hour", ignore_index=True)
    gap = t["hour"].diff().fillna(999) != 1
    t["block_gap_only"] = gap.cumsum() - 1
    brk = gap | (t["split"] != t["split"].shift())
    t["event_block"] = brk.cumsum() - 1
    return t


def continuity(df, res, name):
    out = {}
    prev_h = lookup(df, "hour", -1)
    out["has_prev_1h_row"] = float(np.isfinite(prev_h).mean())
    for k, col in WL_LAGS.items():
        prev = lookup(df, "wl_t", -k)
        ok = np.isfinite(prev)
        match = np.abs(prev[ok] - df[col].to_numpy(np.float64)[ok]) <= TOL
        out[f"lag{k}h"] = {"row_exists": float(ok.mean()), "match_rate_when_exists": float(match.mean()) if ok.any() else None}
    res[name]["continuity"] = out


def rain_accum(df, res, name):
    out = {}
    for src in ["imerg", "sfc"]:
        c1, c6, c24 = f"{src}_rain_1h", f"{src}_rain_6h", f"{src}_rain_24h"
        v1, v6, v24 = (df[c].to_numpy(np.float64) for c in (c1, c6, c24))
        o = {"viol_1h_gt_6h": float((v1 > v6 + TOL).mean()), "viol_6h_gt_24h": float((v6 > v24 + TOL).mean())}
        for win, vw in [(6, v6), (24, v24)]:
            acc = np.zeros(len(df))
            for j in range(win):
                acc += lookup(df, c1, -j) if j else v1
            ok = np.isfinite(acc)
            err = np.abs(acc[ok] - vw[ok])
            tol = np.maximum(0.05, 0.01 * np.abs(vw[ok]))
            o[f"sum1h_vs_{win}h"] = {"rows_checkable": float(ok.mean()), "match_rate": float((err <= tol).mean()),
                                    "median_abs_err": float(np.median(err)), "corr": float(np.corrcoef(acc[ok], vw[ok])[0, 1])}
        out[src] = o
    res[name]["rain_accum"] = out


def future_max(df):
    fut = np.full(len(df), -np.inf)
    n_ok = np.zeros(len(df), dtype=int)
    for j in range(1, 7):
        v = lookup(df, "wl_t", j)
        ok = np.isfinite(v)
        n_ok += ok
        fut = np.where(ok, np.maximum(fut, v), fut)
    return fut, n_ok


def main():
    tr = load("train")
    te = load("test")
    cats = sorted(tr["station_id"].astype(str).unique())
    tr, te = add_keys(tr, cats), add_keys(te, cats)
    res = {"train": {}, "test": {}}

    # --- 사상 블록 ---
    blocks = event_blocks(np.sort(tr["hour"].unique()), np.sort(te["hour"].unique()))
    bmap = blocks.set_index("hour")["event_block"]
    for df in (tr, te):
        df["event_block"] = bmap.reindex(df["hour"].to_numpy()).to_numpy()
        df["pos_in_block"] = df["hour"] - df.groupby("event_block")["hour"].transform("min")
    bl = blocks.groupby("event_block").agg(split=("split", "first"), n_hours=("hour", "size"),
                                           h0=("hour", "min"), h1=("hour", "max"))
    bl["n_rows"] = pd.concat([tr, te]).groupby("event_block").size()
    bl["n_station"] = pd.concat([tr[["event_block", "station_code"]], te[["event_block", "station_code"]]]) \
        .groupby("event_block")["station_code"].nunique()
    bl.to_csv(CACHE / "results" / "s01_event_blocks.csv")
    res["blocks"] = {
        "n_blocks_total": int(len(bl)), "n_train": int((bl.split == "train").sum()), "n_test": int((bl.split == "test").sum()),
        "gap_only_blocks": int(blocks["block_gap_only"].nunique()),
        "hours_per_block": bl.groupby("split")["n_hours"].describe().round(1).to_dict("index"),
        "stations_per_block": bl.groupby("split")["n_station"].describe().round(1).to_dict("index"),
    }

    # --- run (관측소별 연속 구간) ---
    for df in (tr, te):
        new = (df["station_code"].diff() != 0) | (df["hour"].diff() != 1)
        df["run_id"] = new.cumsum().astype(np.int64) - 1
    for name, df in [("train", tr), ("test", te)]:
        rl = df.groupby("run_id").size()
        runs_per_sb = df.groupby(["event_block", "station_code"])["run_id"].nunique()
        res[name]["runs"] = {"n_runs": int(len(rl)), "len": rl.describe().round(1).to_dict(),
                             "share_station_block_single_run": float((runs_per_sb == 1).mean())}
        continuity(df, res, name)
        rain_accum(df, res, name)

    # --- 타깃 재계산 (train) ---
    fut, n_ok = future_max(tr)
    full = n_ok == 6
    rec = fut - tr["wl_t"].to_numpy(np.float64)
    y = tr[TARGET].to_numpy(np.float64)
    err = np.abs(rec[full] - y[full])
    part = (n_ok > 0) & ~full
    res["train"]["target_recompute"] = {
        "rows_with_6_future": float(full.mean()),
        "rows_with_partial_future": float(part.mean()),
        "rows_with_no_future": float((n_ok == 0).mean()),
        "match_rate_full": float((err <= TOL).mean()),
        "median_abs_err_full": float(np.median(err)),
        "partial_future_lower_bound_ok": float((rec[part] <= y[part] + TOL).mean()) if part.any() else None,
    }
    tr["n_future_rows"] = n_ok.astype(np.int8)

    # --- test 누수 위험: 존재 비율만 (값은 계산·저장하지 않음) ---
    n_te = np.zeros(len(te), dtype=int)
    for j in range(1, 7):
        n_te += np.isfinite(lookup(te, "hour", j))
    res["test"]["future_rows_in_test"] = {
        "all_6": float((n_te == 6).mean()), "at_least_1": float((n_te >= 1).mean()), "none": float((n_te == 0).mean()),
    }
    te["n_future_rows"] = n_te.astype(np.int8)

    # --- 파생열 + 저장 ---
    for df in (tr, te):
        for k, col in WL_LAGS.items():
            df[f"dwl_{k}h"] = (df["wl_t"] - df[col]).astype(np.float32)
        df.drop(columns=["key"], inplace=True)
    tr.to_parquet(CACHE / "train_sorted.parquet", index=False)
    te.to_parquet(CACHE / "test_sorted.parquet", index=False)
    save_json(res, "s01_integrity.json")

    import json
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:6000])
    peak_rss("s01")


if __name__ == "__main__":
    main()

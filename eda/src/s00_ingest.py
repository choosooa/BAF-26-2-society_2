"""S0. CSV → parquet(float32/category) + 기본 프로파일."""
import time

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from common import ROOT, CACHE, ID, FEATURES, TARGET, IMERG, SFC, save_json, peak_rss


def ingest(name):
    t0 = time.time()
    tbl = pacsv.read_csv(ROOT / f"{name}.csv")
    df = tbl.to_pandas()
    del tbl
    mem_raw = df.memory_usage(deep=True).sum() / 1e6
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
    df["station_id"] = df["station_id"].astype("category")
    num = [c for c in FEATURES + [TARGET] if c in df.columns]
    df[num] = df[num].astype(np.float32)
    mem_new = df.memory_usage(deep=True).sum() / 1e6
    df.to_parquet(CACHE / f"{name}.parquet", index=False)
    print(f"{name}: {df.shape}, {mem_raw:,.0f}MB → {mem_new:,.0f}MB, {time.time()-t0:.1f}s")
    return df, mem_raw, mem_new


def profile(df, name):
    num = df.select_dtypes("number")
    out = {
        "shape": list(df.shape),
        "n_null": int(df.isna().sum().sum()),
        "n_dup_rows": int(df.drop(columns="row_id").duplicated().sum()),
        "n_dup_station_time": int(df.duplicated(["station_id", "timestamp"]).sum()),
        "n_station": int(df["station_id"].nunique()),
        "n_timestamp": int(df["timestamp"].nunique()),
        "ts_min": str(df["timestamp"].min()), "ts_max": str(df["timestamp"].max()),
        "describe": num.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.round(4).to_dict("index"),
        "zero_ratio": {c: float((df[c] == 0).mean()) for c in IMERG + SFC},
        "negative_ratio": {c: float((num[c] < 0).mean()) for c in num.columns},
    }
    if TARGET in df:
        t = df[TARGET]
        out["target"] = {"neg": float((t < 0).mean()), "zero": float((t == 0).mean()), "pos": float((t > 0).mean())}
    return out


def main():
    res = {}
    stations = {}
    for name in ["train", "test"]:
        df, m0, m1 = ingest(name)
        p = profile(df, name)
        p["mem_raw_MB"], p["mem_f32_MB"] = m0, m1
        res[name] = p
        stations[name] = set(df["station_id"].astype(str).unique())
        del df
    res["station_sets"] = {
        "same": stations["train"] == stations["test"],
        "only_train": sorted(stations["train"] - stations["test"]),
        "only_test": sorted(stations["test"] - stations["train"]),
    }
    save_json(res, "s00_profile.json")
    for n in ["train", "test"]:
        r = res[n]
        print(n, r["shape"], "null", r["n_null"], "dup", r["n_dup_rows"], "dup(st,ts)", r["n_dup_station_time"],
              "stations", r["n_station"], "timestamps", r["n_timestamp"], r["ts_min"], "~", r["ts_max"])
    print("target sign:", res["train"]["target"])
    print("zero ratio:", {k: round(v, 3) for k, v in res["train"]["zero_ratio"].items()})
    print("station sets same:", res["station_sets"]["same"])
    peak_rss("s00")


if __name__ == "__main__":
    main()

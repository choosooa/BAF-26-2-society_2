import numpy as np, pandas as pd
def fit_cells(dev, col="imerg_rain_1h"):
    sig = dev.pivot_table(index="timestamp", columns="station_id", values=col).round(4)
    sig = sig.dropna(axis=0)          # 모든 관측소가 있는 시각만 (dev 301개)
    return sig.apply(lambda c: hash(tuple(c.values))).rename("cell")

def add_cell_neighbor(df, cells, src="dwl_1h", name="nb_dwl_1h"):
    df = df.merge(cells.reset_index(), on="station_id", how="left")
    g = df.groupby(["event_id", "timestamp", "cell"])[src]
    s, n = g.transform("sum"), g.transform("count")
    own, has = df[src].fillna(0), df[src].notna().astype(int)
    k = n - has
    df[name] = np.where(k > 0, (s - own) / k.clip(lower=1), np.nan)   # 자기 제외 평균
    return df.drop(columns="cell")

def fit_wl_ref(dev):
    return {s: np.sort(v.values) for s, v in dev.groupby("station_id")["wl_t"]}

def add_wl_quantile(df, ref):
    q = np.full(len(df), np.nan)
    for s, idx in df.groupby("station_id").indices.items():
        if s in ref:
            q[idx] = np.searchsorted(ref[s], df["wl_t"].values[idx], side="right") / len(ref[s])
    df["wl_q"] = q
    df["wl_q_x_sfc6"] = q * np.log1p(df["sfc_rain_6h"])
    return df

def add_arriving_rain(df, L):
    l = df["station_id"].map(L).fillna(L.median()).round().astype(int).values
    out = np.zeros(len(df))
    for k in range(6):
        col = df["sfc_rain_1h"] if k == 0 else df[f"sfc_rain_1h_lag{k}"]
        use = (k >= l - 6) & (k <= l - 1)
        out += np.where(use, col.fillna(0).values, 0.0)
    df["arriving_rain"] = out
    return df

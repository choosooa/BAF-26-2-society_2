import pandas as pd, numpy as np
from pathlib import Path
from pathlib import Path
HERE = Path(__file__).resolve().parent          # local_handong/scripts
REPO = HERE.parents[1]                          # 레포 루트
S = str(HERE.parent / 'cache') + '/'            # 중간 산출물
R = str(REPO) + '/'
ROOT=REPO
df=pd.read_csv(ROOT/"data/train.csv",usecols=['row_id','timestamp','station_id','wl_t','wl_t_minus_1h','imerg_rain_1h','sfc_rain_1h','sfc_rain_6h'],parse_dates=['timestamp'])
ts=pd.Series(np.sort(df.timestamp.unique())); eno=ts.diff().gt(pd.Timedelta(hours=1)).cumsum()
df['event_id']=df.timestamp.map(dict(zip(ts,'TR_E'+eno.astype(str).str.zfill(3))))
df['dwl_1h']=df.wl_t-df.wl_t_minus_1h
df=df.sort_values(['station_id','timestamp']).reset_index(drop=True)
g=df.groupby(['station_id','event_id'],sort=False)
for k in range(1,6):
    prev=g['sfc_rain_1h'].shift(k); ok=(df.timestamp-g['timestamp'].shift(k))==pd.Timedelta(hours=k)
    df[f'sfc_rain_1h_lag{k}']=prev.where(ok)
dev=df
exec(open(HERE / 'snips.py').read())
cells=fit_cells(dev); print('cells',cells.nunique())
out=add_cell_neighbor(df,cells); print('nb nonnull',out.nb_dwl_1h.notna().mean().round(3), out.shape==df.shape+(0,1) or out.shape)
ref=fit_wl_ref(dev); out=add_wl_quantile(out,ref); print('q range',out.wl_q.min(),out.wl_q.max())
L=pd.read_csv(ROOT/"eda/cache/results/nb09_station_lag.csv").set_index("station_id")["L"]
out=add_arriving_rain(out,L); print('arriving', out.arriving_rain.describe().round(2).to_dict())
# leak check: drop last 6h of each event, recompute, compare
end=df.groupby('event_id').timestamp.transform('max'); cut=df[df.timestamp<=end-pd.Timedelta(hours=6)].reset_index(drop=True)
o2=add_arriving_rain(add_wl_quantile(add_cell_neighbor(cut,cells),ref),L).set_index('row_id')
o1=out.set_index('row_id').loc[o2.index]
for c in ['nb_dwl_1h','wl_q_x_sfc6','arriving_rain']:
    print(c,'same',np.allclose(o1[c].values,o2[c].values,equal_nan=True))

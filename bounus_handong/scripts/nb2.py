import pandas as pd, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent          # local_handong/scripts
REPO = HERE.parents[1]                          # 레포 루트
S = str(HERE.parent / 'cache') + '/'            # 중간 산출물
R = str(REPO) + '/'
d=pd.read_parquet(S+'resid.parquet')
tr=pd.read_csv(R+'data/train.csv',usecols=['timestamp','station_id','imerg_rain_1h','sfc_rain_1h','imerg_rain_6h','sfc_rain_6h'],parse_dates=['timestamp'])
d['timestamp']=pd.to_datetime(d.timestamp)
rng=np.random.default_rng(1)
def common_share(df,keys):
    m=df.groupby(keys).r.transform('mean'); return float((m**2).sum()/(df.r**2).sum())
for src in ['imerg_rain_1h','sfc_rain_1h']:
    sig=tr.pivot_table(index='timestamp',columns='station_id',values=src).round(4)
    h=sig.apply(lambda c: hash(tuple(np.nan_to_num(c.values,nan=-1))))
    cell=h.rename('cell'); dd=d.merge(cell.reset_index(),on='station_id')
    sz=dd.groupby(['cell','timestamp']).r.transform('size'); dd=dd[sz>1]
    obs=common_share(dd,['cell','timestamp'])
    # null: permute station labels' cells? shuffle residuals within (event,timestamp) across stations
    nulls=[]
    for _ in range(3):
        x=dd.copy(); x['r']=x.groupby(['event_id','timestamp']).r.transform(lambda s: rng.permutation(s.values))
        nulls.append(common_share(x,['cell','timestamp']))
    # residual corr between cellmates: lift = (obs - null)
    print(src,'cells',h.nunique(),'stations in multi',int((h.groupby(h).transform('size')>1).sum()),'rows',len(dd),'obs',round(obs,3),'null',np.round(nulls,3), 'event-ts',round(common_share(dd,['event_id','timestamp']),3))

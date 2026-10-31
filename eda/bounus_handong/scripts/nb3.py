import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
from pathlib import Path
HERE = Path(__file__).resolve().parent          # local_handong/scripts
REPO = HERE.parents[1]                          # 레포 루트
S = str(HERE.parent / 'cache') + '/'            # 중간 산출물
R = str(REPO) + '/'
tr=pd.read_csv(R+'data/train.csv',usecols=['timestamp','station_id','imerg_rain_1h','sfc_rain_1h'],parse_dates=['timestamp'])
d=pd.read_parquet(S+'resid.parquet'); d['timestamp']=pd.to_datetime(d.timestamp)
rng=np.random.default_rng(1); res={}
def cs(df,keys): m=df.groupby(keys).r.transform('mean'); return float((m**2).sum()/(df.r**2).sum())
for col in ['imerg_rain_1h','sfc_rain_1h']:
    sig=tr.pivot_table(index='timestamp',columns='station_id',values=col).round(4).dropna(axis=0)
    cell=sig.apply(lambda c: hash(tuple(c.values))).rename('cell')
    dd=d.merge(cell.reset_index(),on='station_id')
    g=dd.groupby(['event_id','timestamp','cell'])
    n=g.dwl1.transform('count'); s=g.dwl1.transform('sum'); own=dd.dwl1.fillna(0); ownc=dd.dwl1.notna().astype(int)
    dd['nb']=np.where(n-ownc>0,(s-own)/(n-ownc).clip(lower=1),np.nan)
    sz=g.r.transform('size'); x=dd[sz>1]
    obs=cs(x,['cell','timestamp']); nul=[]
    for _ in range(3):
        y=x.copy(); y['r']=y.groupby(['event_id','timestamp']).r.transform(lambda v: rng.permutation(v.values)); nul.append(cs(y,['cell','timestamp']))
    mk=dd.nb.notna()
    ce=dd[mk].groupby('event_id').apply(lambda q: np.corrcoef(q.r,q.nb)[0,1] if q.nb.std()>0 else np.nan)
    nt=~dd.regime.isin(['조석','고정'])
    mk2=mk&nt
    res[col]=dict(cells=int(cell.nunique()),st_mate=int((cell.groupby(cell).transform('size')>1).sum()),rows_share=float((sz>1).mean()),
        obs=obs,null=float(np.mean(nul)),corr=float(np.corrcoef(dd.r[mk],dd.nb[mk])[0,1]),spear=float(spearmanr(dd.r[mk],dd.nb[mk]).correlation),
        corr_nt=float(np.corrcoef(dd.r[mk2],dd.nb[mk2])[0,1]),
        corr_own=float(np.corrcoef(dd.r[mk],dd.dwl1.fillna(0)[mk])[0,1]),ev_pos=float((ce>0).mean()))
    cell.to_csv(S+f'cells_{col}.csv')
print(json.dumps(res,indent=1,ensure_ascii=False))
json.dump(res,open(S+'nb3.json','w'),indent=1)

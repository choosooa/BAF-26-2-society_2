import pandas as pd, numpy as np, json
from pathlib import Path
HERE = Path(__file__).resolve().parent          # local_handong/scripts
REPO = HERE.parents[1]                          # 레포 루트
S = str(HERE.parent / 'cache') + '/'            # 중간 산출물
R = str(REPO) + '/'
cols=['row_id','timestamp','station_id','wl_t','wl_t_minus_1h','wl_t_minus_3h','imerg_rain_1h','imerg_rain_6h','sfc_rain_1h','sfc_rain_6h','sfc_rain_24h','wsarea','urban_ratio','target_maxrise_6h']
tr=pd.read_csv(R+'data/train.csv',usecols=cols,parse_dates=['timestamp'])
ts=pd.Series(np.sort(tr.timestamp.unique())); eno=ts.diff().gt(pd.Timedelta(hours=1)).cumsum()
tr['event_id']=tr.timestamp.map(dict(zip(ts,'TR_E'+eno.astype(str).str.zfill(3))))
oof=pd.read_csv(R+'outputs/handong_tuning/oof_기준.csv')
d=oof.merge(tr,on='row_id')
ef=pd.read_csv(R+'data/../outputs/common/event_folds.csv')
d=d.merge(ef[['event_id','fold']],on='event_id')
reg=pd.read_csv(R+'eda/cache/results/nb07_regimes.csv')[['station_id','regime']]
d=d.merge(reg,on='station_id',how='left')
y=d.target_maxrise_6h; p=d.oof; d['r']=y-p; d['se']=d.r**2
out={}
out['n_rows']=len(d); out['n_events']=d.event_id.nunique(); out['folds']=sorted(d.fold.unique().tolist()); out['n_st']=d.station_id.nunique()
out['rmse']=float(np.sqrt(d.se.mean()))
r=d.r; out['resid']=dict(mean=r.mean(),median=r.median(),sd=r.std(),skew=r.skew(),q01=r.quantile(.01),q99=r.quantile(.99),min=r.min(),max=r.max())
srt=np.sort(d.se.values)[::-1]; out['top1pct_sse_share']=srt[:len(srt)//100].sum()/srt.sum(); out['top01pct_sse_share']=srt[:len(srt)//1000].sum()/srt.sum()
# situation
d['dwl1']=d.wl_t-d.wl_t_minus_1h
rain=(d.sfc_rain_6h>0)|(d.imerg_rain_6h>=1)
sit=np.select([~rain & (d.dwl1.abs()<=0.05), rain & (d.dwl1.abs()<=0.05), d.dwl1>0.05, d.dwl1<-0.05],['무강우·정체','강우·정체','상승 중','하강 중'],'무강우·정체')
d['sit']=sit
def seg(g):
    return pd.Series({'row%':len(g)/len(d)*100,'sse%':g.se.sum()/d.se.sum()*100,'rmse':np.sqrt(g.se.mean()),'bias':g.r.mean(),'y_mean':g.target_maxrise_6h.mean()})
nt=~d.regime.isin(['조석','고정'])
out['sit_all']=d.groupby('sit').apply(seg).round(3).to_dict('index')
out['sit_nontidal']=d[nt].groupby('sit').apply(seg).round(3).to_dict('index')
out['regime']=d.groupby('regime').apply(seg).round(3).to_dict('index')
# onset
on=(d.dwl1<=0.05)&(y>0.5); out['onset']=dict(row=on.mean(),sse=d.se[on].sum()/d.se.sum(),y=y[on].mean(),p=p[on].mean())
# y-bins vs p-bins
yb=pd.cut(y,[-99,-0.1,0.1,0.5,1,2,99]); out['by_y']=d.groupby(yb,observed=True).apply(lambda g: pd.Series({'row%':len(g)/len(d)*100,'sse%':g.se.sum()/d.se.sum()*100,'bias':g.r.mean()})).round(3).to_dict('index')
pb=pd.cut(p,[-99,0,0.1,0.3,0.5,1,2,99]); out['by_p']=d.groupby(pb,observed=True).apply(lambda g: pd.Series({'n':len(g),'p':g.oof.mean(),'y':g.target_maxrise_6h.mean(),'bias':g.r.mean()})).round(3).to_dict('index')
# event share
ev=d.groupby('event_id').agg(sse=('se','sum'),n=('se','size')).sort_values('sse',ascending=False)
out['top_events']=(ev.sse.head(10)/d.se.sum()).round(3).to_dict(); out['top5_share']=ev.sse.head(5).sum()/d.se.sum()
top5=ev.index[:5]; m=~d.event_id.isin(top5); out['rmse_excl_top5']=float(np.sqrt(d.se[m].mean())); out['rmse_nontidal']=float(np.sqrt(d.se[nt].mean()))
# timestamp-common residual decomposition (within event): r = m_ts + e
d['m_ts']=d.groupby(['event_id','timestamp']).r.transform('mean')
d['n_ts']=d.groupby(['event_id','timestamp']).r.transform('size')
out['ts_common_share_of_sse']=float((d.m_ts**2).sum()/d.se.sum())
for name,g in [('top5',d[d.event_id.isin(top5)]),('rest',d[m]),('nontidal',d[nt])]:
    mm=g.groupby(['event_id','timestamp']).r.transform('mean')
    out['ts_common_'+name]=float((mm**2).sum()/(g.r**2).sum())
# null: shuffle residuals within event across timestamps
rng=np.random.default_rng(0); vals=[]
for _ in range(3):
    sh=d.groupby('event_id').r.transform(lambda s: pd.Series(rng.permutation(s.values),index=s.index))
    mm=sh.groupby([d.event_id,d.timestamp]).transform('mean'); vals.append(float((mm**2).sum()/d.se.sum()))
out['ts_common_null']=vals
# by-event consistency: share of events where ts-common share > null-ish (>5%)
pe=d.groupby('event_id').apply(lambda g: (g.groupby('timestamp').r.transform('mean')**2).sum()/max((g.r**2).sum(),1e-9))
out['ts_common_event_median']=float(pe.median()); out['ts_common_event_gt10']=float((pe>0.10).mean())
# fold bias
out['fold']=d.groupby('fold').apply(lambda g: pd.Series({'rmse':np.sqrt(g.se.mean()),'bias':g.r.mean()})).round(4).to_dict('index')
# neighbors: stations sharing identical IMERG series (same grid cell) in train
sig=tr.pivot_table(index='timestamp',columns='station_id',values='imerg_rain_1h')
sig=sig.round(4)
h=sig.apply(lambda c: hash(tuple(np.nan_to_num(c.values,nan=-1))))
grp=h.groupby(h).transform('size'); out['imerg_cells']=int(h.nunique()); out['stations_with_cellmate']=int((grp>1).sum())
cell=h.rename('cell').reset_index()
d=d.merge(cell,on='station_id')
g=d.groupby(['cell','timestamp'])
sm=g.dwl1.transform('sum'); cnt=g.dwl1.transform('size')
d['nb_dwl1']=np.where(cnt>1,(sm-d.dwl1)/(cnt-1).clip(lower=1),np.nan)
mk=d.nb_dwl1.notna()
out['nb_rows_share']=float(mk.mean())
from scipy.stats import spearmanr
out['corr_r_nb_dwl1']=float(np.corrcoef(d.r[mk],d.nb_dwl1[mk])[0,1]); out['spear_r_nb_dwl1']=float(spearmanr(d.r[mk],d.nb_dwl1[mk]).correlation)
out['corr_r_own_dwl1']=float(np.corrcoef(d.r[mk],d.dwl1[mk])[0,1])
d['nb_minus_own']=d.nb_dwl1-d.dwl1
out['corr_r_nb_minus_own']=float(np.corrcoef(d.r[mk],d.nb_minus_own[mk])[0,1])
# per-event consistency of sign
ce=d[mk].groupby('event_id').apply(lambda g: np.corrcoef(g.r,g.nb_minus_own)[0,1] if g.nb_minus_own.std()>0 else np.nan)
out['nb_event_pos_share']=float((ce>0).mean()); out['nb_event_n']=int(ce.notna().sum())
# xs dwl1 change over time (cross-section at t vs t-1) vs residual
xs=d.groupby(['event_id','timestamp']).dwl1.mean().rename('xsv').reset_index().sort_values(['event_id','timestamp'])
xs['xs_prev']=xs.groupby('event_id')['xsv'].shift(1); xs['xs_chg']=xs['xsv']-xs['xs_prev']
d=d.merge(xs[['event_id','timestamp','xs_chg']],on=['event_id','timestamp'],how='left')
mk2=d.xs_chg.notna(); out['corr_r_xs_chg']=float(np.corrcoef(d.r[mk2],d.xs_chg[mk2])[0,1])
# urban x rain 1h residual corr in nontidal moderate rain
mm=nt&(d.sfc_rain_1h>0)&(d.sfc_rain_1h<=40)
out['corr_r_urban_x_rain']=float(np.corrcoef(d.r[mm],(d.urban_ratio*np.log1p(d.sfc_rain_1h))[mm])[0,1])
out['corr_r_logarea_x_rain24']=float(np.corrcoef(d.r[nt],(np.log1p(d.wsarea)*np.log1p(d.sfc_rain_24h))[nt])[0,1])
def conv(o):
    if isinstance(o,dict): return {str(k):conv(v) for k,v in o.items()}
    return o
out=conv(out)
json.dump(out,open(S+'resid_eda.json','w'),ensure_ascii=False,indent=1,default=float)
d[['row_id','event_id','timestamp','station_id','fold','regime','sit','oof','target_maxrise_6h','r','m_ts','dwl1']].to_parquet(S+'resid.parquet')
print(json.dumps(out,ensure_ascii=False,indent=1,default=float))

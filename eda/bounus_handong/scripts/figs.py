import sys, json, pandas as pd, numpy as np, matplotlib
from pathlib import Path
HERE = Path(__file__).resolve().parent          # local_handong/scripts
REPO = HERE.parents[1]                          # 레포 루트
S = str(HERE.parent / 'cache') + '/'            # 중간 산출물
R = str(REPO) + '/'
matplotlib.use('Agg'); import matplotlib.pyplot as plt
IMG=sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / 'docs' / 'images')
plt.rcParams.update({'font.family':'AppleGothic','axes.unicode_minus':False,'font.size':10,
 'axes.edgecolor':'#c3c2b7','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e',
 'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','axes.grid':True,'axes.axisbelow':True,'grid.color':'#ebeae5','grid.linewidth':0.8})
B,O='#2a78d6','#eb6834'; T1,T2='#0b0b0b','#52514e'
d=pd.read_parquet(S+'resid.parquet'); y=d.target_maxrise_6h; p=d.oof
edges=[-0.5,0,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5,9]
# fig1 calibration both ways
fig,ax=plt.subplots(figsize=(6.4,4.6))
pb=pd.cut(p,edges); a=d.groupby(pb,observed=True).agg(x=('oof','mean'),v=('target_maxrise_6h','mean'),n=('oof','size')); a=a[a.n>=200]
yb=pd.cut(y,edges); b=d.groupby(yb,observed=True).agg(x=('target_maxrise_6h','mean'),v=('oof','mean'),n=('oof','size')); b=b[b.n>=200]
lim=4.5
ax.plot([0,lim],[0,lim],color='#a8a79f',lw=1,ls='--',zorder=1); ax.text(3.9,3.55,'y = x',color=T2,fontsize=9)
ax.plot(a.x,a.v,color=B,lw=2,marker='o',ms=6,mec='#fcfcfb',mew=2,zorder=3)
ax.plot(b.x,b.v,color=O,lw=2,marker='s',ms=6,mec='#fcfcfb',mew=2,zorder=3)
ax.set_xlim(-0.2,lim); ax.set_ylim(-0.2,lim); ax.set_xlabel('칸의 기준값 평균 (m)'); ax.set_ylabel('칸 안의 상대 평균 (m)')
ax.set_title('예측은 보정되어 있고, 과소예측은 정답 기준으로 묶을 때만 보인다',loc='left',fontsize=11,color=T1)
ax.legend(handles=[plt.Line2D([],[],color=B,marker='o',lw=2,label='예측값으로 묶은 칸의 실제 평균  E[y | 예측]'),plt.Line2D([],[],color=O,marker='s',lw=2,label='정답으로 묶은 칸의 예측 평균  E[예측 | y]')],frameon=False,loc='upper left')
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.text(0.01,0.005,'dev OOF 642,684행, LightGBM 기준(0.2178). 칸당 200행 이상만 표시',fontsize=8,color=T2)
fig.tight_layout(rect=(0,0.02,1,1)); fig.savefig(IMG+'/kmeda_calibration.png',dpi=160); plt.close()
# fig2 situation shares
o=json.load(open(S+'resid_eda.json'))['sit_all']; order=['강우·정체','상승 중','하강 중','무강우·정체']
row=[o[k]['row%'] for k in order]; sse=[o[k]['sse%'] for k in order]
fig,ax=plt.subplots(figsize=(6.4,3.6)); yy=np.arange(len(order)); h=0.36
ax.barh(yy-h/2-0.01,row,height=h,color=B,edgecolor='#fcfcfb',lw=2,label='행 비중'); ax.barh(yy+h/2+0.01,sse,height=h,color=O,edgecolor='#fcfcfb',lw=2,label='오차제곱합 비중')
for i,(r,s) in enumerate(zip(row,sse)):
    ax.text(r+1,i-h/2,f'{r:.1f}%',va='center',fontsize=8.5,color=T2); ax.text(s+1,i+h/2,f'{s:.1f}%',va='center',fontsize=8.5,color=T1)
ax.set_yticks(yy,order); ax.invert_yaxis(); ax.set_xlim(0,72); ax.set_xlabel('%'); ax.grid(axis='y',visible=False)
ax.set_title('오차는 "비가 오는데 아직 정체 중"과 "상승 중" 행에 몰려 있다',loc='left',fontsize=11,color=T1)
ax.legend(frameon=False,loc='lower right')
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.text(0.01,0.005,'정체 = |dwl_1h| ≤ 0.05m, 강우 = sfc_rain_6h > 0 또는 imerg_rain_6h ≥ 1mm',fontsize=8,color=T2)
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(IMG+'/kmeda_situation.png',dpi=160); plt.close()
# fig3 common error
labs=['같은 사상·같은 시각\n(전체 관측소)','같은 IMERG 격자·시각\n(이웃 관측소)','같은 지상강수 격자·시각\n(이웃 관측소)']
obs=[0.056,0.746,0.681]; nul=[0.016,0.439,0.368]
fig,ax=plt.subplots(figsize=(6.4,3.6)); x=np.arange(3); w=0.34
ax.bar(x-w/2-0.01,obs,width=w,color=B,edgecolor='#fcfcfb',lw=2,label='관측'); ax.bar(x+w/2+0.01,nul,width=w,color=O,edgecolor='#fcfcfb',lw=2,label='무작위 대조')
for i,(a_,b_) in enumerate(zip(obs,nul)):
    ax.text(i-w/2,a_+0.015,f'{a_:.2f}',ha='center',fontsize=8.5,color=T1); ax.text(i+w/2,b_+0.015,f'{b_:.2f}',ha='center',fontsize=8.5,color=T2)
ax.set_xticks(x,labs,fontsize=9); ax.set_ylim(0,0.9); ax.set_ylabel('묶음 평균 잔차가 설명하는 SSE 비율'); ax.grid(axis='x',visible=False)
ax.set_title('함께 틀리는 단위는 사상 전체가 아니라 이웃 관측소다',loc='left',fontsize=11,color=T1)
ax.legend(frameon=False,loc='upper left')
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.text(0.01,0.005,'대조: 같은 사상·시각 안에서 잔차를 관측소끼리 섞은 뒤 같은 계산 (3회 평균). 묶음이 작을수록 대조값이 커진다',fontsize=7.5,color=T2)
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(IMG+'/kmeda_common_error.png',dpi=160); plt.close()
print('ok')

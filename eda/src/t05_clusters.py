"""T5. 5.5 관측소 군집 재작도(그림 5-5a·b, 통일 스타일). 기존 s08 지문 결과 사용.

군집(s08.cluster)과 같은 공간에서 그린다: 지표 9개(BEHAV), 결측 = 중앙값, 분위수→정규 변환, PCA.
"""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from tb_common import style, figsize, save, plt, RESULTS, CAT, C_ALL

F = ["acf1", "acf12", "acf24", "dry_sd_log", "rain_rho", "tstd_log", "flat_share", "neg_share", "asym"]  # = s08.BEHAV
LAB = {1: "C1 조용한 산지 소하천", 2: "C2 약하게 반응하는 저지·도시", 3: "C3 크게 반응하는 중규모", 4: "C4 비 없이도 움직이는 대하천"}

style()
fp = pd.read_csv(RESULTS / "s08_station_fingerprints.csv", index_col=0)
X = fp[F].fillna(fp[F].median())
Q = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit_transform(X)
pca = PCA(2, random_state=0).fit(Q); P = pca.transform(Q); ev = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=figsize(16, 8))
for t, c in zip(LAB, CAT):
    m = (fp.type == t).to_numpy()
    ax.scatter(P[m, 0], P[m, 1], s=10, color=c, alpha=.7, label=f"{LAB[t]} (n={m.sum()})")
ax.set_xlabel(f"지문 PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"지문 PC2 ({ev[1]:.0%})"); ax.legend(fontsize=7)
ax.set_title("그림 5-5a  움직임 지문 9개로 본 관측소 4유형 (군집에 쓴 분위수 변환 공간의 주성분 평면)")
save(fig, "5-5a", "station_types")

med = pd.DataFrame(Q, index=fp.index, columns=F).groupby(fp.type).median().rename(index=LAB)
fig, ax = plt.subplots(figsize=figsize(16, 7))
x = np.arange(len(F)); w = .2
for i, (t, c) in enumerate(zip(med.index, CAT)):
    ax.bar(x + (i - 1.5) * w, med.loc[t], w, color=c, label=t)
ax.axhline(0, color=C_ALL, lw=.6); ax.set_xticks(x); ax.set_xticklabels(F, rotation=30, fontsize=7)
ax.set_ylabel("분위수 변환 값 (유형별 중앙값)"); ax.legend(fontsize=6, ncol=2)
ax.set_title("그림 5-5b  유형별 지문: C3는 지속성, C4는 12·24h 주기와 무강우 변동이 크다")
save(fig, "5-5b", "type_signatures")
print(med.round(2).T.to_string())

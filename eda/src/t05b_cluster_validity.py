"""T5b. 5.5 관측소 군집 타당성 점검: 실루엣 0.16을 어떻게 읽을 것인가.

군집(s08.cluster)과 같은 공간(지표 9개, 결측 = 중앙값, 분위수→정규 변환)에서
① 군집 없는 귀무 데이터와 실루엣 비교 ② GMM BIC ③ Hopkins ④ 안정성(ARI)
⑤ 유형별·유형 쌍별 실루엣 ⑥ 유역환경 → 연속 지문 직접 예측(R²)을 본다. 결과: s08_cluster_validity.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score as ARI
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from common import RESULTS, ENV, save_json

warnings.filterwarnings("ignore")
F = ["acf1", "acf12", "acf24", "dry_sd_log", "rain_rho", "tstd_log", "flat_share", "neg_share", "asym"]  # = s08.BEHAV
RES = {}
rng = np.random.default_rng(0)


def km(Z, k, seed=0, n_init=30):
    return KMeans(k, n_init=n_init, random_state=seed).fit(Z)


def sil(Z, k):
    return float(silhouette_score(Z, km(Z, k).labels_))


def null_silhouette(Q, reps=20):
    """같은 공분산 가우시안(단봉, 군집 없음)과 열별 셔플(상관도 제거)에서 k-means 실루엣."""
    C, mu = np.cov(Q.T), Q.mean(0)
    out = {}
    for k in range(2, 9):
        real = sil(Q, k)
        g = [sil(rng.multivariate_normal(mu, C, len(Q)), k) for _ in range(reps)]
        p = [sil(np.column_stack([rng.permutation(c) for c in Q.T]), k) for _ in range(reps)]
        out[k] = {"real": real, "gauss_mean": float(np.mean(g)), "gauss_sd": float(np.std(g)),
                  "gauss_z": float((real - np.mean(g)) / np.std(g)), "perm_mean": float(np.mean(p)), "perm_sd": float(np.std(p))}
    return out


def gmm_bic(Q):
    return {k: float(min(GaussianMixture(k, covariance_type="full", n_init=5, random_state=s).fit(Q).bic(Q) for s in range(3)))
            for k in range(1, 9)}


def hopkins(Z, m=50, reps=20):
    """0.5 = 무작위, 1에 가까울수록 군집 경향. 상관만으로도 올라가므로 귀무와 비교해서 읽는다."""
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    out = []
    for _ in range(reps):
        w = nn.kneighbors(Z[rng.choice(len(Z), m, replace=False)])[0][:, 1]
        u = nn.kneighbors(rng.uniform(Z.min(0), Z.max(0), (m, Z.shape[1])), 1)[0][:, 0]
        out.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(out))


def stability(Q):
    base = km(Q, 4).labels_
    seed = [float(ARI(base, km(Q, 4, s, n_init=1).labels_)) for s in range(1, 11)]
    out = {"seed_n_init1_ari": seed}
    for k, reps in ((4, 50), (3, 30), (5, 30), (6, 30)):
        bl = base if k == 4 else km(Q, k).labels_
        bs = []
        for b in range(reps):
            i = rng.choice(len(Q), len(Q), replace=True)
            bs.append(ARI(bl, km(Q[i], k, b, n_init=10).predict(Q)))
        out[f"bootstrap_ari_k{k}"] = {"mean": float(np.mean(bs)), "q10": float(np.quantile(bs, .1)), "q90": float(np.quantile(bs, .9))}
    return out, base


def per_type(Q, lab, fp):
    s = silhouette_samples(Q, lab)
    types = sorted(set(lab))
    out = {"per_type": {f"C{t}": {"n": int((lab == t).sum()), "median_sil": float(np.median(s[lab == t])),
                                  "neg_share": float(np.mean(s[lab == t] < 0))} for t in types},
           "neg_share_all": float(np.mean(s < 0)), "pairwise_sil": {}}
    for a in types:
        for b in types:
            if a < b:
                m = (lab == a) | (lab == b)
                out["pairwise_sil"][f"C{a}-C{b}"] = float(silhouette_score(Q[m], lab[m]))
    g2 = GaussianMixture(2, n_init=5, random_state=0).fit_predict(Q)
    out["k2_kmeans_vs_type"] = pd.crosstab(km(Q, 2).labels_, lab).to_dict()
    out["k2_gmm_vs_type"] = pd.crosstab(g2, lab).to_dict()
    out["k2_gmm_vs_tidal"] = pd.crosstab(g2, fp.tidal).to_dict()
    nt = ~fp.tidal.to_numpy()
    out["nontidal_k4_sil"] = sil(Q[nt], 4)
    out["drop_one_k4_sil"] = {c: sil(Q[:, [j for j, x in enumerate(F) if x != c]], 4) for c in F}
    out["pca_k4_sil"] = {f"{d}d": sil(PCA(d).fit_transform(Q), 4) for d in (2, 3)}
    return out


def env_r2(X, fp):
    """유형 라벨 없이: 유역환경 7변수로 연속 지문 각각을 맞히는 교차검증 R²."""
    Xe = np.log1p(fp[ENV].clip(lower=0))
    cv = KFold(5, shuffle=True, random_state=0)
    out = {}
    for c in F:
        y = X[c].to_numpy()
        p = cross_val_predict(RandomForestRegressor(400, min_samples_leaf=3, random_state=0), Xe, y, cv=cv)
        out[c] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return out


def main():
    fp = pd.read_csv(RESULTS / "s08_station_fingerprints.csv", index_col=0)
    X = fp[F].fillna(fp[F].median())
    Q = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit_transform(X)
    lab = fp.type.to_numpy()
    RES["n_stations"] = len(X)
    RES["nan_filled"] = fp[F].isna().sum().to_dict()
    RES["null_silhouette"] = null_silhouette(Q)
    RES["gmm_bic"] = gmm_bic(Q)
    RES["hopkins"] = {"real": hopkins(Q), "gauss_null": hopkins(rng.multivariate_normal(Q.mean(0), np.cov(Q.T), len(Q)))}
    RES["stability"], base = stability(Q)
    RES["saved_vs_refit_ari"] = float(ARI(lab, base))
    RES["types"] = per_type(Q, lab, fp)
    RES["env_r2"] = env_r2(X, fp)
    save_json(RES, "s08_cluster_validity.json")
    print(json.dumps(RES, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

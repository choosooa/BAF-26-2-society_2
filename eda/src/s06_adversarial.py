"""S6. 적대적 검증: train vs test 구분 가능성."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import load, FEATURES, TARGET, savefig, save_json, peak_rss

DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]
N = 150_000


def main():
    cols = FEATURES + DWL + ["event_block"]
    tr = load("train_sorted", cols).sample(N, random_state=0)
    te = load("test_sorted", cols)
    te = te.sample(min(N, len(te)), random_state=0)
    X = pd.concat([tr, te], ignore_index=True)
    lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    feats = FEATURES + DWL
    oof = np.zeros(len(X))
    # 사상 단위로 fold를 나눠야 "사상 고유 특성"이 아닌 분포 차이를 본다
    gkf = GroupKFold(n_splits=5)
    models = []
    for tri, vai in gkf.split(X, lab, groups=X.event_block):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
        m.fit(X.iloc[tri][feats], lab[tri])
        oof[vai] = m.predict_proba(X.iloc[vai][feats])[:, 1]
        models.append((m, vai))
    auc = roc_auc_score(lab, oof)
    m, vai = models[0]
    sub = np.random.default_rng(0).choice(vai, size=min(40_000, len(vai)), replace=False)
    pi = permutation_importance(m, X.iloc[sub][feats], lab[sub], scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)

    top = list(imp.index[:8])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    ks = {}
    from scipy.stats import ks_2samp
    for ax, c in zip(axes.flat, top):
        a, b = tr[c].dropna(), te[c].dropna()
        lo, hi = np.nanquantile(np.r_[a, b], [.005, .995])
        bins = np.linspace(lo, hi, 60)
        ax.hist(a.clip(lo, hi), bins=bins, density=True, alpha=.5, color="#534AB7", label="train")
        ax.hist(b.clip(lo, hi), bins=bins, density=True, alpha=.5, color="#D85A30", label="test")
        ks[c] = float(ks_2samp(a, b).statistic)
        ax.set_title(f"{c}\nKS={ks[c]:.3f}, 중요도={imp[c]:.4f}", fontsize=9); ax.set_yscale("log")
    axes.flat[0].legend()
    fig.suptitle(f"train vs test 분포 (적대적 검증 AUC = {auc:.3f})"); fig.tight_layout(); savefig(fig, "06a_adversarial.png")

    ks_all = {c: float(ks_2samp(tr[c].dropna(), te[c].dropna()).statistic) for c in feats}
    q = {c: {"train_p50": float(tr[c].median()), "test_p50": float(te[c].median()),
             "train_p99": float(tr[c].quantile(.99)), "test_p99": float(te[c].quantile(.99))} for c in top}
    res = {"auc_groupkfold": float(auc), "perm_importance_top": imp.head(12).round(5).to_dict(),
           "ks_top": dict(sorted(ks_all.items(), key=lambda kv: -kv[1])[:10]), "quantiles_top": q}
    save_json(res, "s06_adversarial.json")
    import json
    print(json.dumps(res, ensure_ascii=False, indent=1))
    peak_rss("s06")


if __name__ == "__main__":
    main()

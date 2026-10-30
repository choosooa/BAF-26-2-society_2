"""중간보고서(옵시디언 표시 반영판)용 그림.

- 기존 그림 잘라 쓰기: fig3-1 → Q-Q plot만, sh_4-3 → 그룹별 자기상관 곡선만
- 새 그림: sh_2-3c (sh_2-3b의 IMERG를 레이더로 바꾼 판), sh_3-0 (VIF)
실행: python eda/src/m02_marked_figs.py
출력: eda/figures/midterm2/*.png (옵시디언 이미지 폴더에도 복사)
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EDA = ROOT / "eda"
OUT = EDA / "figures" / "midterm2"
OUT.mkdir(parents=True, exist_ok=True)
VAULT_IMG = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/mooimong/90_아카이브/이미지"

# self.ipynb와 같은 스타일
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
C_BOTH, C_A, C_SFC, C_NONE = "#2F6DB5", "#8FB3DE", "#E07B39", "#D0D0D0"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [fig]", p.name)


# ============================================================
# 1. 기존 그림 잘라 쓰기
# ============================================================
def crop(src, box_frac, name):
    im = Image.open(VAULT_IMG / src)
    w, h = im.size
    x0, y0, x1, y1 = box_frac
    im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(OUT / name)
    print("  [crop]", name)


crop("fig3-1_target_dist.png", (0.500, 0.13, 1.0, 1.0), "fig3-1_qq_only.png")
crop("sh_4-3_periodicity_map.png", (0.5, 0.0, 1.0, 1.0), "sh_4-3_group_acf.png")

# ============================================================
# 2. 데이터
# ============================================================
df = pd.read_parquet(EDA / "cache" / "train.parquet")
reg = pd.read_csv(EDA / "cache" / "results" / "nb07_regimes.csv")
excl = set(reg.loc[reg["group"].isin(["조석", "고정"]), "station_id"])

# ============================================================
# 3. sh_2-3c: 레이더 × 지상 (sh_2-3b에서 IMERG 자리를 레이더로)
# ============================================================
CATS = ["둘 다", "레이더만", "지상만", "둘 다 없음"]
r = df["radar_reflectivity_p95"] >= 20          # 2.4절과 같은 '비 옴' 기준
s = df["sfc_rain_6h"] > 0                        # sh_2-3b와 같은 6h 창
d = df[["station_id", "radar_reflectivity_p95", "sfc_rain_6h", "target_maxrise_6h"]].assign(
    조합=np.select([r & s, r & ~s, ~r & s], CATS[:3], CATS[3]))

fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
for cat, c in [("레이더만", C_A), ("둘 다", C_BOTH)]:
    x = d.loc[d["조합"] == cat, "radar_reflectivity_p95"]
    axes[0].hist(x, bins=np.arange(20, 57, 1), density=True, alpha=.6, color=c, label=f"{cat} (중앙 {x.median():.1f})")
for cat, c in [("지상만", C_SFC), ("둘 다", C_BOTH)]:
    x = d.loc[d["조합"] == cat, "sfc_rain_6h"]
    axes[1].hist(x.clip(upper=25), bins=np.arange(0, 25.5, 0.5), density=True, alpha=.6, color=c, label=f"{cat} (중앙 {x.median():.1f}mm)")
axes[0].set(title="레이더가 비를 볼 때 레이더 반사도 p95", xlabel="radar_reflectivity_p95 (≥ 20)", ylabel="밀도")
axes[1].set(title="지상이 비를 볼 때 지상 6h 강우 (25mm 이상은 끝 칸)", xlabel="6h 강우 (mm)", ylabel="밀도")
axes[0].legend(); axes[1].legend()
share = d["조합"].value_counts(normalize=True)
for ax, (name, g) in zip(axes[2:], [("전체", d), ("비조석", d[~d["station_id"].isin(excl)])]):
    t = g.groupby("조합")["target_maxrise_6h"]
    pd.DataFrame({"평균": t.mean(), "p95": t.quantile(.95)}).loc[CATS].plot.bar(ax=ax, rot=0, color=[C_BOTH, C_A])
    ax.set_xticklabels([f"{c}\n({share[c]:.0%})" for c in CATS])
    ax.set(title=f"{name}: 레이더 × 지상 6h 조합별 타깃", xlabel="", ylabel="target_maxrise_6h (m)")
axes[3].sharey(axes[2])
plt.tight_layout()
save(fig, "sh_2-3c_radar_one_side.png")

nt = d[~d["station_id"].isin(excl)].groupby("조합")["target_maxrise_6h"]
print("share", share.round(3).to_dict())
print("nontidal p95", nt.quantile(.95).round(2).to_dict(), "mean", nt.mean().round(3).to_dict())
print("radar p95 median", d.groupby("조합")["radar_reflectivity_p95"].median().round(1).to_dict())
print("sfc6h median", d.groupby("조합")["sfc_rain_6h"].median().round(1).to_dict())

# ============================================================
# 4. sh_3-0: VIF (유역환경 7변수끼리, 관측소 단위 n=465)
# ============================================================
ENV = ["cat_area", "wsarea", "mean_elevation", "mean_slope", "urban_ratio", "forest_ratio", "stream_order"]
NAME = {"cat_area": "소권역 면적", "wsarea": "상류 유역면적", "mean_elevation": "평균표고", "mean_slope": "평균경사",
        "urban_ratio": "도시화율", "forest_ratio": "산림비율", "stream_order": "하천차수"}
st = df.groupby("station_id")[ENV].first().astype(float)
X = st.copy()
for c in ["cat_area", "wsarea"]:
    X[c] = np.log1p(X[c])                        # 면적은 왜도 7~12라 log
vif = pd.Series(np.diag(np.linalg.inv(X.corr().to_numpy())), index=ENV).sort_values()
AXIS = {"mean_slope": "산지", "forest_ratio": "산지", "mean_elevation": "산지", "wsarea": "규모", "stream_order": "규모",
        "urban_ratio": "기타", "cat_area": "기타"}
GC = {"산지": "#7FB77E", "규모": "#2F6DB5", "기타": "#BBBBBB"}

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh([NAME[c] for c in vif.index], vif.values, color=[GC[AXIS[c]] for c in vif.index])
for y, val in enumerate(vif.values):
    ax.text(val + 0.15, y, f"{val:.1f}", va="center", fontsize=10)
ax.axvline(5, color="gray", ls="--", lw=1)
ax.axvline(10, color="k", ls="--", lw=1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GC["산지"], label="산지 축(표고·경사·산림)"), Patch(color=GC["규모"], label="규모 축(면적·차수)"),
                   Patch(color=GC["기타"], label="기타")], loc="lower right")
ax.set(title="유역환경 7변수 VIF (관측소 단위 n=465, 면적 log1p)", xlabel="VIF")
plt.tight_layout()
save(fig, "sh_3-0_vif.png")
print("VIF", vif.round(1).to_dict())
r = st.corr(method="spearman")
print("spearman slope-forest", round(r.loc["mean_slope", "forest_ratio"], 2), "elev-forest", round(r.loc["mean_elevation", "forest_ratio"], 2),
      "elev-slope", round(r.loc["mean_elevation", "mean_slope"], 2), "wsarea-order", round(r.loc["wsarea", "stream_order"], 2),
      "urban-forest", round(r.loc["urban_ratio", "forest_ratio"], 2))

# ============================================================
# 5. 옵시디언 이미지 폴더로 복사
# ============================================================
for p in OUT.glob("*.png"):
    shutil.copy(p, VAULT_IMG / p.name)
print("copied to vault")

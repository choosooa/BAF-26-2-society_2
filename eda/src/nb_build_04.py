"""4권: 관측소 유형·유역환경."""
from nb_common import Book

b = Book()
b.md("""
# 🏞️ 4권 · 관측소 유형과 유역환경 — "어떤 하천이 크게 반응하나?"

> 시리즈: 1권 구조·품질 → 2권 타깃과 강우 → 3권 수위 변화·시계열 → **4권 관측소 유형·유역환경** → 5권 연결망 → 6권 검증 전략·오차 위치 · 별책 조석

### 한 줄 요약
**유역환경(면적·경사·도시화 등) 7개는 '산지↔저지'와 '하천 규모' 두 축으로 요약돼요. 혼자서는 약한 변수예요. 섞어 보면 강한 비에서 큰 유역이 더 오르는 것 같지만, 사상별로 보면 길고 큰 사상에서만 그래요. 관측소를 '물이 움직이는 방식'으로 묶으면 네 유형이 나오는데, 유역환경만으로는 그 유형의 절반 정도만 맞혀요. 관측소를 구별하는 정보는 수위 기록 쪽에 더 많아요.**

| 장 | 질문 |
|---|---|
| 1 | 유역환경 7개는 어떻게 생겼고, 몇 가지 축으로 요약되나 |
| 2 | 유역환경이 다르면 평균 상승도 다른가 |
| 3 | 강한 비가 올 때 유역환경의 효과가 커지나 (상호작용) — 섞은 그림과 사상별 그림이 다르다 |
| 4 | 관측소를 '움직이는 방식'으로 묶으면 (행동 지문과 유형) |
| 5 | 유역환경으로 유형을 맞힐 수 있나 |
| 6 | 누수 점검과 정리 |

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **상류 유역면적** (`wsarea`) | 이 관측소로 물이 모여드는 땅 전체의 넓이 | 한강 하류는 매우 큼 |
| **소권역 집수면적** (`cat_area`) | 관측소가 속한 작은 권역의 넓이 | |
| **하천차수** (`stream_order`) | 지류가 합쳐질수록 커지는 하천 크기 등급 | 1차 = 가장 작은 실개천 |
| **주성분분석(PCA)** | 서로 겹치는 변수 여러 개를 소수의 '축'으로 요약 | 키·몸무게 → '체격' 한 축 |
| **행동 지문** | 관측소의 수위가 움직이는 방식을 숫자 몇 개로 요약한 것 | 비에 얼마나 반응하나, 평소 얼마나 흔들리나 |
| **군집(KMeans)** | 비슷한 지문끼리 묶기 | |
| **실루엣** | 묶음이 얼마나 뚜렷하게 갈리는지(−1~1, 클수록 뚜렷) | 0.1대면 경계가 흐릿함 |
""")
b.setup()
b.code(r'''
from scipy import stats
cols = ["station_id", "station_code", "hour", "event_block", "pos_in_block", TARGET, "dwl_1h", "wl_t"] + IMERG + SFC + ENV
tr = add_flags(pd.read_parquet(CACHE / "train_sorted.parquet", columns=cols))
key = keyed(tr)
st = tr.groupby("station_id")[ENV + ["group"]].first()       # 관측소 단위 표 (n = 465)
print(f"관측소 {len(st)}곳 · 그룹별:", st.group.value_counts().to_dict())
''')

b.md("""
## 1. 유역환경 7개는 어떻게 생겼나

유역환경은 **시간에 따라 변하지 않아요.** 그래서 행 단위가 아니라 **관측소 단위(465곳)** 로 봐야 해요. 행 단위로 보면 행이 많은 관측소의 값이 여러 번 세어지거든요.
""")
b.code(r'''
fig, ax = plt.subplots(2, 4, figsize=(14, 6)); ax = ax.ravel()
for a, c in zip(ax, ENV):
    x = st[c].astype(float)
    a.hist(x, bins=30, color=C_NT); a.set_title(f"{c}  왜도 {stats.skew(x):.1f}", fontsize=9)
ax[7].hist(np.log10(st.wsarea), bins=30, color=C_GREEN); ax[7].set_title(f"log10(wsarea)  왜도 {stats.skew(np.log10(st.wsarea)):.1f}", fontsize=9)
fig.suptitle("관측소 465곳의 유역환경: 면적은 한쪽으로 크게 치우침", x=.02, ha="left"); fig.tight_layout(); plt.show()
''')
b.md("""
- 면적 두 개는 **작은 유역이 대부분이고 아주 큰 유역이 몇 곳** 있어요(왜도 7~12). 로그를 씌우면 훨씬 고르게 퍼져요.
- 하천차수는 소권역 평균이라 소수점이 있어요(1권).

### 1-1. 7개를 몇 가지 축으로 요약할 수 있나 (주성분분석)
면적·표고·차수는 로그를 씌우고, 모두 표준화한 뒤 주성분분석을 해요.
""")
b.code(r'''
X = st[ENV].astype(float).copy()
for c in ("cat_area", "wsarea", "mean_elevation", "stream_order"):
    X[c] = np.log(X[c])
Z = (X - X.mean()) / X.std()
U, S, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
ev = S ** 2 / (S ** 2).sum(); sc = U * S
load = pd.DataFrame(Vt[:3].T, index=ENV, columns=["PC1", "PC2", "PC3"])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
ax[0].bar(range(1, 8), ev * 100, color=C_NT); ax[0].plot(range(1, 8), np.cumsum(ev) * 100, "-o", color=C_TIDAL, ms=3)
ax[0].set_xlabel("주성분"); ax[0].set_ylabel("설명 비율 (%)"); ax[0].set_title("3개 축이 대부분 (선 = 누적)")
col = np.where(st.group == "조석", C_TIDAL, C_NT)
ax[1].scatter(sc[:, 0], sc[:, 1], s=8, c=col, alpha=.6)
k = np.abs(sc[:, :2]).max() / np.abs(Vt[:2]).max() * .8
for i, c in enumerate(ENV):
    ax[1].arrow(0, 0, Vt[0, i] * k, Vt[1, i] * k, color="k", width=.01, head_width=.12)
    ax[1].text(Vt[0, i] * k * 1.12, Vt[1, i] * k * 1.12, c, fontsize=8)
ax[1].set_xlabel(f"PC1 ({ev[0]:.0%})"); ax[1].set_ylabel(f"PC2 ({ev[1]:.0%})"); ax[1].set_title("관측소 지도 (주황 = 조석)")
fig.tight_layout(); plt.show()
print(f"누적 설명 비율: PC1 {ev[0]:.0%} · PC1~2 {ev[:2].sum():.0%} · PC1~3 {ev[:3].sum():.0%}")
load.round(2)
''')
b.md("""
**화살표 읽는 법**: 화살표 방향으로 갈수록 그 변수가 큰 관측소예요.
- **PC1 = 산지 ↔ 저지·도시**: 표고·경사·산림이 한쪽, 도시화율이 반대쪽이에요.
- **PC2 = 하천 규모**: 상류면적과 하천차수가 같이 움직여요.
- 3개 축으로 대부분이 설명돼요. → 신경망처럼 변수 수에 민감한 모델에는 **주성분 3개로 줄여** 넣을 수 있어요.
- 조석 관측소(주황)는 **저지·도시 쪽**에 모여 있어요. 바다와 가까운 하구니까 자연스러워요.
""")

b.md("""
## 2. 유역환경이 다르면 평균 상승도 다른가

비조석 관측소마다 **평균 타깃**을 구해, 유역환경 7개와 순위 상관을 봐요(관측소 단위).
""")
b.code(r'''
nt = tr[tr.nontidal]
m = nt.groupby("station_id")[TARGET].agg(["mean", "std"])
d = st.join(m, how="inner")
fig, ax = plt.subplots(2, 4, figsize=(14, 6)); ax = ax.ravel()
rows = {}
for a, c in zip(ax, ENV):
    xv = np.log10(d[c]) if c in ("cat_area", "wsarea") else d[c]
    r = stats.spearmanr(d[c], d["mean"]).statistic; rows[c] = r
    a.scatter(xv, d["mean"], s=6, alpha=.6, color=C_NT)
    a.set_title(f"{c}{' (log)' if c in ('cat_area', 'wsarea') else ''}  ρ = {r:+.2f}", fontsize=9)
ax[7].axis("off"); ax[0].set_ylabel("관측소 평균 타깃 (m)"); ax[4].set_ylabel("관측소 평균 타깃 (m)")
fig.suptitle(f"유역환경 혼자서는 약하다 (비조석 {len(d)}곳)", x=.02, ha="left"); fig.tight_layout(); plt.show()
print({k: round(v, 2) for k, v in rows.items()})
''')
b.md("""
상관이 전부 **약해요(|ρ| ≤ 0.25 안팎)**. 방향은 상식과 맞아요. 산지(표고·경사·산림 높음)일수록 평균 상승이 작고, 도시화율이 높을수록 커요. 하지만 흩어짐이 매우 커서 **유역환경 하나만 보고는 관측소의 반응을 예측하기 어려워요.**

## 3. 강한 비가 올 때 효과가 커지나 (상호작용)

"유역환경은 비가 **올 때** 반응 크기를 조절한다"는 가설을 확인해요.
비조석 행을 **강우 6구간 × 유역환경 3구간(관측소 기준)** 으로 나눠 타깃 평균을 그려요.
""")
b.code(r'''
rb = pd.cut(nt.imerg_rain_6h, [-.01, 0, 1, 5, 15, 40, 1e9], labels=["0", "0~1", "1~5", "5~15", "15~40", "40+"])
envs = [("wsarea", "상류 유역면적"), ("urban_ratio", "도시화율"), ("mean_slope", "평균 경사")]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
tert = {}
for a, (e, lab) in zip(ax, envs):
    tq = pd.qcut(st.loc[st.index.isin(nt.station_id.unique()), e].rank(method="first"), 3, labels=["하", "중", "상"])
    tert[e] = tq
    t = nt.station_id.map(tq)
    mm = nt.groupby([rb, t], observed=True)[TARGET].mean().unstack()
    for q_, c_ in zip(["하", "중", "상"], ["#9ec5f4", "#3987e5", "#0d366b"]):
        a.plot(range(len(mm)), mm[q_], "-o", ms=3, color=c_, label=f"{lab} {q_}")
    a.set_xticks(range(len(mm))); a.set_xticklabels(mm.index, fontsize=8); a.set_xlabel("IMERG 6h 강우 (mm)"); a.legend(fontsize=7); a.set_title(lab)
ax[0].set_ylabel("타깃 평균 (m)")
fig.suptitle("비가 적을 땐 차이가 없고, 강한 비에서 벌어진다 (비조석)", x=.02, ha="left"); fig.tight_layout(); plt.show()
''')
b.md("""
- 비가 15mm 이하일 땐 세 구간이 거의 겹쳐요.
- **40mm가 넘는 강한 비**에서 벌어져요. 상류 유역면적이 큰 관측소일수록 더 오르고, 도시화율은 오히려 반대 방향이에요(도시가 많은 저지대 하천은 이미 넓고 완만해서일 수 있어요).

**전체를 섞은 그림이니, 사상 하나하나에서도 그런지** 확인해요. 강한 비(40mm+) 행이 충분한(구간마다 50행 이상) 사상마다 "상류면적 상위 3분위 평균 − 하위 3분위 평균"을 구해요.
""")
b.code(r'''
heavy = nt[nt.imerg_rain_6h > 40].copy()
heavy["tq"] = heavy.station_id.map(tert["wsarea"])
g = heavy.groupby(["event_block", "tq"], observed=True)[TARGET].agg(["mean", "size"]).unstack()
ok = (g["size"]["상"] >= 50) & (g["size"]["하"] >= 50)
diff = (g["mean"]["상"] - g["mean"]["하"])[ok]
ev_len = nt.groupby("event_block").pos_in_block.max().reindex(diff.index) + 1 if "pos_in_block" in nt else None
w = g["size"][["상", "하"]].sum(axis=1)[ok]                            # 사상별 강한 비 행 수 (섞은 그림에서의 무게)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].bar(range(len(diff)), diff.sort_values(), color=np.where(diff.sort_values() > 0, C_NT, C_ALL))
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xlabel("사상 (차이 순)"); ax[0].set_ylabel("상위 − 하위 3분위 (m)")
ax[0].set_title(f"강한 비 사상 {len(diff)}개 각각: 방향이 제각각")
ax[1].scatter(ev_len, diff, s=w / w.max() * 400 + 10, color=np.where(diff > 0, C_NT, C_ALL), alpha=.7)
for b_ in w.sort_values().index[-4:]:
    ax[1].annotate(f"#{b_}", (ev_len[b_], diff[b_]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax[1].axhline(0, color="k", lw=.5); ax[1].set_xscale("log"); ax[1].set_xlabel("사상 길이 (h, log)"); ax[1].set_ylabel("상위 − 하위 (m)")
ax[1].set_title("원 크기 = 섞은 그림에서의 무게(강한 비 행 수)"); fig.tight_layout(); plt.show()
print(f"섞은 그림: 상위 {heavy[heavy.tq == '상'][TARGET].mean():.2f} m vs 하위 {heavy[heavy.tq == '하'][TARGET].mean():.2f} m")
print(f"사상 길이와 차이의 순위 상관: {diff.corr(ev_len, method='spearman'):.2f}")
print(f"큰 유역이 더 오른 사상: {(diff > 0).mean():.0%} ({(diff > 0).sum()}/{len(diff)}) · 차이 중앙값 {diff.median():+.2f} m")
''')
b.md("""
**섞은 그림과 사상별 그림이 달라요.** 이게 이 장의 핵심이에요.
- 섞은 그림에서는 강한 비에서 큰 유역이 더 올랐어요(상위 0.40m vs 하위 0.31m).
- 그런데 사상 하나하나로 보면 **큰 유역이 더 오른 사상은 절반도 안 되고(19개 중 8개), 차이 중앙값은 거의 0**이에요.
- 오른쪽 그림: 섞은 그림의 차이는 **길고 큰 사상 몇 개**(원이 큰 #125, #9 등)가 만들어요. 이런 사상에서는 큰 유역이 더 올랐고, 짧은 사상에서는 오히려 반대인 경우가 많아요. 사상 길이와 차이의 순위 상관이 **0.56**이에요.

**왜 그럴까요?** 3권에서 본 것처럼 **큰 유역은 늦게 반응해요.** 비가 짧게 오고 그치면 큰 강은 아직 제대로 오르지 않았고, 비가 며칠씩 이어지면 상류의 물이 모두 도착해 크게 올라요.

→ 유역환경의 효과는 **"강한 비 × 유역" 만으로는 부족하고, 비가 얼마나 오래 이어졌나(누적 강우, 사상 경과 시간)와 함께** 봐야 해요. 트리 모델은 이런 3중 상호작용도 잡을 수 있지만, 데이터가 몇몇 큰 사상에 몰려 있어서 **과적합에 주의**해야 해요(6권).
""")

b.md("""
## 4. 관측소를 '움직이는 방식'으로 묶으면

유역환경은 약했어요. 그렇다면 **수위 기록 자체**에서 관측소의 성격을 뽑아 보면 어떨까요?
관측소마다 아래 **행동 지문 9개**를 계산해요.

| 지문 | 뜻 | 예시 |
|---|---|---|
| 변화 지속성 (`acf1`) | dwl_1h의 1시간 자기상관 | 한번 오르면 계속 오르나 |
| 12h·24h 주기 (`acf12`, `acf24`) | 12·24시간 자기상관 | 조석·조작 관측소를 가려냄 |
| 무강우 변동 (`dry_sd_log`) | 비 없을 때 수위 변화의 표준편차(log) | 비 없이도 흔들리나 |
| 강우 반응 (`rain_rho`) | 6h 강우 순위 ↔ 타깃 순위 상관 | 비에 얼마나 반응하나 |
| 타깃 변동 (`tstd_log`) | 타깃 표준편차(log) | 얼마나 크게 오르내리나 |
| 무변화 비율 (`flat_share`) | dwl_1h가 정확히 0인 비율 | 물이 거의 안 움직이나 |
| 하강 타깃 비율 (`neg_share`) | 타깃이 음수인 비율 | 빠지는 시간이 많나 |
| 급상승/급하강 비대칭 (`asym`) | 가장 빠른 상승 ÷ 가장 빠른 하강(log) | 자연 하천은 빨리 오르고 천천히 빠짐 |

⚠️ `rain_rho`, `tstd_log`, `neg_share`는 **타깃을 써요.** 여기선 관측소를 이해하려는 **분석용**이에요. 모델 변수로 쓸 땐 **학습 묶음 안에서만** 계산해야 해요.
""")
b.code(r'''
s = tr.station_id.to_numpy()
x = tr.dwl_1h.to_numpy(float); yv = tr[TARGET].to_numpy(float)
fp = pd.DataFrame(index=st.index)
for L in (1, 6, 12, 24):
    fp[f"acf{L}"] = station_corr(tr, x, shift_lookup(tr, key, "dwl_1h", -L))
dry = ((tr.imerg_rain_24h < .5) & (tr.sfc_rain_24h < .5)).to_numpy()
fp["dry_sd_log"] = np.log10(pd.Series(np.where(dry, x, np.nan)).groupby(s).std() + 1e-3)
rain6 = np.maximum(tr.imerg_rain_6h, tr.sfc_rain_6h)
fp["rain_rho"] = station_corr(tr, rain6.groupby(tr.station_id).rank(pct=True).to_numpy(), tr[TARGET].groupby(tr.station_id).rank(pct=True).to_numpy())
fp["tstd_log"] = np.log10(pd.Series(yv).groupby(s).std() + 1e-3)
fp["flat_share"] = pd.Series((x == 0).astype(float)).groupby(s).mean()
fp["neg_share"] = pd.Series((yv < 0).astype(float)).groupby(s).mean()
q99 = pd.Series(x).groupby(s).quantile(.99); q01 = pd.Series(x).groupby(s).quantile(.01).abs()
fp["asym"] = np.log10((q99 + .01) / (q01 + .01))
BEHAV = ["acf1", "acf12", "acf24", "dry_sd_log", "rain_rho", "tstd_log", "flat_share", "neg_share", "asym"]
fp["group"] = st.group
fp[BEHAV].describe().round(2)
''')
b.md("""
### 4-1. 군집 — 몇 개로 묶는 게 좋을까
지문 9개를 순위 기반으로 정규분포 모양으로 바꾼 뒤(값의 크기 차이·치우침 제거), 3~8개로 묶어 보고 **실루엣 점수**로 비교해요. 해석하기 좋은 4~7개 중 점수가 가장 높은 개수를 골라요.
""")
b.code(r'''
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

Xb = fp[BEHAV].fillna(fp[BEHAV].median())
Q = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit_transform(Xb)
sil, labs = {}, {}
for k in range(3, 9):
    km = KMeans(k, n_init=30, random_state=0).fit(Q)
    sil[k], labs[k] = silhouette_score(Q, km.labels_), km.labels_
K = max(range(4, 8), key=lambda k: sil[k])
lab = pd.Series(labs[K], index=fp.index)
order = fp.groupby(lab)["tstd_log"].median().sort_values().index          # 타깃 변동이 작은 유형부터 C1, C2 …
lab = lab.map({o: i + 1 for i, o in enumerate(order)})
fp["type"] = "C" + lab.astype(str)
print("실루엣:", {k: round(v, 3) for k, v in sil.items()}, "→ 선택", K)

P = PCA(2, random_state=0).fit_transform(Q)
fig, ax = plt.subplots(1, K, figsize=(3.2 * K, 3.2), sharex=True, sharey=True)
for i, a in enumerate(ax, start=1):
    mk = (lab == i).to_numpy()
    a.scatter(P[~mk, 0], P[~mk, 1], s=6, color="#d9d8d3"); a.scatter(P[mk, 0], P[mk, 1], s=10, color=C_NT)
    tmk = mk & (fp.group == "조석").to_numpy()
    a.scatter(P[tmk, 0], P[tmk, 1], s=26, facecolor="none", edgecolor=C_TIDAL, lw=1.2)
    a.set_title(f"C{i} · {mk.sum()}곳", fontsize=9)
fig.suptitle("관측소 행동 지문의 군집 (파랑 = 해당 유형, 주황 테두리 = 조석)", x=.02, ha="left"); fig.tight_layout(); plt.show()
''')
b.md("""
실루엣이 **0.1~0.2 수준으로 낮아요.** 유형 사이 경계가 뚜렷하지 않다는 뜻이에요. 관측소는 딱 떨어지는 몇 종류라기보다 **연속적인 스펙트럼**에 가까워요. 그래도 유형을 나누면 "대략 어떤 하천들이 있는지"를 이해하는 데 도움이 돼요.

### 4-2. 유형별 지문 — 각 유형은 어떤 하천인가
""")
b.code(r'''
prof = fp.groupby("type")[BEHAV].median()
prof["관측소 수"] = fp.type.value_counts()
prof["조석 수"] = fp[fp.group == "조석"].type.value_counts().reindex(prof.index).fillna(0).astype(int)
prof["약한 조석·24h 수"] = fp[fp.group.isin(["약한 조석", "24h 주기"])].type.value_counts().reindex(prof.index).fillna(0).astype(int)
prof["상류면적 중앙값"] = st.wsarea.groupby(fp.type).median()
prof["경사 중앙값"] = st.mean_slope.groupby(fp.type).median()
Zp = (prof[BEHAV] - fp[BEHAV].median()) / (fp[BEHAV].quantile(.75) - fp[BEHAV].quantile(.25)).replace(0, 1)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("bw", ["#184f95", "#6da7ec", "#f0efec", "#f09a8f", "#b3261e"])
fig, ax = plt.subplots(figsize=(12, .6 * K + 1.5))
im = ax.imshow(Zp.clip(-2.5, 2.5).to_numpy(), cmap=cm, vmin=-2.5, vmax=2.5, aspect="auto")
for (i, j), v in np.ndenumerate(prof[BEHAV].to_numpy()):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(BEHAV))); ax.set_xticklabels(BEHAV, rotation=25, ha="right"); ax.set_yticks(range(K)); ax.set_yticklabels(prof.index)
ax.grid(False); fig.colorbar(im, shrink=.8, label="전체 대비 (빨강 = 높음)"); ax.set_title("유형별 지문 중앙값 (숫자 = 원값)"); plt.show()
prof[["관측소 수", "조석 수", "약한 조석·24h 수", "상류면적 중앙값", "경사 중앙값"]]
''')
b.md("""
표를 읽고 유형마다 한 줄 설명을 붙여 보면 이래요(숫자는 위 표 기준).
- **타깃 변동이 가장 작은 유형**: 물이 거의 안 움직이고(무변화 비율 높음) 비에도 약하게 반응하는 곳
- **중간 유형들**: 비에 반응하지만 크기와 지속성이 다른 곳
- **타깃 변동이 가장 큰 유형**: 비 없이도 흔들리고 12·24h 주기가 있는 곳 → **조석·약한 조석·24h 조작 관측소가 모두 여기**에 모여요(별책과 일치)

### 4-3. 유형별 '움직임 서명' — 유형 안의 관측소들이 정말 비슷하게 움직이나
유형마다 **소속 관측소 전부**의 자기상관 곡선과 강우 교차상관 곡선을 그려요(선 = 중앙값, 띠 = 25~75%).
""")
b.code(r'''
acf = pd.DataFrame({L: station_corr(tr, x, shift_lookup(tr, key, "dwl_1h", -L)) for L in range(1, 27)})
xc = pd.DataFrame({L: station_corr(tr, shift_lookup(tr, key, "sfc_rain_1h", -L) if L else tr.sfc_rain_1h.to_numpy(float), x) for L in range(0, 13)})
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for i, t in enumerate(prof.index):
    ids = fp.index[fp.type == t]
    for a, M in [(ax[0], acf), (ax[1], xc)]:
        Mi = M.loc[M.index.intersection(ids)]
        a.fill_between(Mi.columns, Mi.quantile(.25), Mi.quantile(.75), color=CAT[i], alpha=.12)
        a.plot(Mi.columns, Mi.median(), "-", color=CAT[i], lw=2, label=f"{t} ({len(Mi)}곳)")
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xlabel("시차 (h)"); ax[0].set_ylabel("dwl_1h 자기상관"); ax[0].legend(fontsize=8); ax[0].set_title("유형별 리듬")
ax[1].axhline(0, color="k", lw=.5); ax[1].set_xlabel("비를 몇 시간 전에서 (L)"); ax[1].set_ylabel("corr(비(t−L), dwl_1h(t))"); ax[1].set_title("유형별 강우 반응")
fig.tight_layout(); plt.show()
''')
b.md("""
띠가 좁을수록 유형 안의 관측소들이 비슷하게 움직인다는 뜻이에요. 유형끼리 곡선이 분명히 달라요. **반응 세기**(오른쪽 곡선 높이)와 **리듬**(왼쪽 12·24h 봉우리)이 유형을 가르는 핵심이에요.

## 5. 유역환경으로 유형을 맞힐 수 있나

유형은 **수위 기록**으로 만들었어요. 그러면 **유역환경 7개만** 가지고 그 유형을 맞힐 수 있을까요? 맞힐 수 있다면 유역환경만으로 충분하고, 못 맞힌다면 수위 기록에서 뽑은 지문이 추가 정보를 준다는 뜻이에요.
""")
b.code(r'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
Xe = np.log1p(st[ENV].clip(lower=0))
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
acc = cross_val_score(rf, Xe, fp.type, cv=cv, scoring="accuracy").mean()
bacc = cross_val_score(rf, Xe, fp.type, cv=cv, scoring="balanced_accuracy").mean()
p = fp.type.value_counts(normalize=True)
print(f"유역환경만으로 유형 맞히기: 정확도 {acc:.0%} (가장 많은 유형만 찍으면 {p.max():.0%}) · 균형 정확도 {bacc:.0%} (무작위 {1 / K:.0%})")

# 같은 소권역 값을 공유하는 관측소끼리는 같은 유형인가
combo = st[ENV].round(4).astype(str).agg("|".join, axis=1)
same = tot = 0
for _, g in fp.assign(k=combo).groupby("k"):
    t = g.type.to_numpy()
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            tot += 1; same += t[i] == t[j]
print(f"유역환경이 똑같은 관측소 쌍 {tot}개 중 같은 유형: {same / tot:.0%} (무작위 쌍이라면 {(p ** 2).sum():.0%})")
''')
b.md("""
- 유역환경만으로 맞히면 정확도가 **"가장 많은 유형만 찍기"보다 높지만, 절반 안팎**이에요.
- 유역환경 값이 **완전히 똑같은** 관측소 쌍도 **절반 정도만** 같은 유형이에요.
  - **예시**: 같은 소권역에 있는 관측소 두 곳이라도, 하나는 댐 바로 아래라 물이 안 움직이고, 다른 하나는 지류 합류점이라 크게 출렁일 수 있어요. 유역환경 표에는 이 차이가 안 담겨요.

→ 관측소를 구별하는 정보는 **유역환경보다 수위 기록(행동 지문)에 더 많아요.** 모델에 관측소별 지문(평소 변동폭, 반응 세기, 주기 등)을 **학습 묶음 안에서** 계산해 넣는 것을 권해요.

## 6. 누수 점검과 정리

### 누수 점검표

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 1장 유역환경 분포·PCA | 시간 불변 입력 | 누수 없음 |
| 2장 관측소 평균 타깃 | 타깃 사용 | 설명용. 모델 변수로 쓰면 학습 묶음 안에서 계산 |
| 3장 상호작용 | 입력(과거 강우·유역) × 타깃 | 설명용 |
| 4장 지문 중 `rain_rho`, `tstd_log`, `neg_share` | **타깃 사용** | 분석용. 모델 변수로 쓰면 **학습 묶음 안에서만** |
| 4장 지문 중 나머지 6개 | 관측소의 train 기간 입력값 | 관측소 특성. train에서 계산해 test에 적용 |
| 5장 유역환경 → 유형 분류 | 관측소 단위 교차검증 | 분석용 |

### 정리

| 발견 | 전체에 해당? | 모델링 시사점 |
|---|---|---|
| 유역환경 7개 = 산지↔저지·규모 축, 3개 성분으로 대부분 설명 | 465곳 | 신경망 등에는 주성분 3개 |
| 유역환경과 평균 상승의 상관은 약함(|ρ| ≤ 0.25 안팎) | 비조석 관측소 전부 | 단독 변수로는 약함 |
| 섞어 보면 강한 비(40mm+)에서 큰 유역이 더 오름 | ❌ 사상별로는 19개 중 8개만. 길고 큰 사상이 만든 차이 | 강우 × 유역 × 비 지속시간(누적 강우·경과 시간)으로, 과적합 주의 |
| 행동 지문 군집은 경계가 흐릿함(실루엣 0.1~0.2) | 465곳 | 유형 라벨보다 **지문 값 자체**를 변수로 |
| 유역환경으로 유형 절반 정도만 맞힘 | 5-fold 교차검증 | 수위 기록 기반 관측소 특성 추가 |
""")
b.write("nb4_station_types_and_basin.ipynb")

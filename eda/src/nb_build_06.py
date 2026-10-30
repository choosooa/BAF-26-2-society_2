"""6권: train/test 비교, 검증 전략, 오차 위치."""
from nb_common import Book

b = Book()
b.md("""
# 🎯 6권 · 검증 전략과 오차의 위치 — "점수를 믿으려면, 그리고 어디를 고쳐야 하나"

> 시리즈: 1권 구조·품질 → 2권 타깃과 강우 → 3권 수위 변화·시계열 → 4권 관측소 유형·유역환경 → 5권 연결망 → **6권 검증 전략·오차 위치** · 별책 조석

### 한 줄 요약
**train과 test는 호우 사건만 다르고 분포는 사실상 같아요. 그런데 행을 무작위로 섞어 검증하면 이웃 행끼리 답이 새서 점수가 약 12% 좋게 보여요. 반드시 사상 단위로 나눠 검증해야 해요. 기준 모델의 오차는 '비가 오며 물이 오르는 중'인 행과 몇 개의 큰 호우에 몰려 있어요.**

| 장 | 질문 |
|---|---|
| 1 | train과 test는 어떻게 나뉘었나 |
| 2 | 둘의 분포가 다른가 (분포 비교, 적대적 검증) |
| 3 | 검증은 어떻게 나눠야 하나 (무작위 vs 사상 단위) |
| 4 | 기준 모델은 어디서 틀리나 (국면별) |
| 5 | 어떤 호우에서 틀리나 (사상별) |
| 6 | 미래 강우 탓은 얼마나 되나 (진단용 실험) |
| 7 | 누수 점검과 시리즈 전체 정리 |

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **KFold** | 데이터를 5묶음으로 나눠 하나씩 검증에 쓰는 방법 | 행을 무작위로 섞어 나눔 |
| **GroupKFold** | 같은 그룹(여기선 사상)은 통째로 같은 묶음에 넣는 방법 | 한 호우는 학습이든 검증이든 한쪽에만 |
| **OOF 예측** | 각 행을, 그 행을 학습에 쓰지 않은 모델이 예측한 값 | 교차검증으로 만든 '정직한' 예측 |
| **적대적 검증** | train과 test를 구분하는 분류기를 만들어 보는 것 | 구분이 안 되면(AUC ≈ 0.5) 둘이 비슷하다 |
| **AUC** | 분류기가 두 집단을 얼마나 잘 가르나(0.5 = 동전 던지기, 1 = 완벽) | |
| **KS 통계량** | 두 분포가 얼마나 다른지(0 = 같음, 1 = 전혀 다름) | |
""")
b.setup()
b.code(r'''
cols = ["station_id", "station_code", "hour", "event_block", "pos_in_block", TARGET] + FEATURES + DWL
tr = add_flags(pd.read_parquet(CACHE / "train_sorted.parquet", columns=cols))
te = add_flags(pd.read_parquet(CACHE / "test_sorted.parquet", columns=[c for c in cols if c != TARGET]))
print(f"train {len(tr):,}행 · 사상 {tr.event_block.nunique()} / test {len(te):,}행 · 사상 {te.event_block.nunique()}")
''')

b.md("""
## 1. train과 test는 어떻게 나뉘었나

1권에서 봤듯이 **관측소는 같고 호우 사건(사상)만 달라요**. test 사상은 시간축 전체에 흩어져 있어요. 즉 이 대회는 "**처음 보는 호우**에서 같은 관측소들이 어떻게 반응할지" 맞히는 문제예요.
""")
b.code(r'''
ev = pd.concat([tr.assign(split="train"), te.assign(split="test")]).groupby(["event_block", "split"]).agg(
    h0=("hour", "min"), hours=("hour", "nunique"), n_st=("station_id", "nunique")).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
for sp, col in [("train", C_NT), ("test", C_TIDAL)]:
    e = ev[ev.split == sp]
    ax[0].scatter(e.h0, e.hours, s=14, color=col, label=f"{sp} {len(e)}개")
    ax[1].hist(e.n_st, bins=np.linspace(380, 466, 30), color=col, alpha=.6, label=sp)
ax[0].set_yscale("log"); ax[0].set_xlabel("합성 시각의 시작 위치 (h)"); ax[0].set_ylabel("사상 길이 (h, log)"); ax[0].legend(); ax[0].set_title("test 사상은 시간축 전체에 흩어져 있다")
ax[1].set_xlabel("사상에 등장한 관측소 수"); ax[1].legend(); ax[1].set_title("사상마다 관측소는 거의 전부"); fig.tight_layout(); plt.show()
''')

b.md("""
## 2. 둘의 분포가 다른가

### 2-1. 변수별 분포 비교 (KS 통계량)
""")
b.code(r'''
from scipy.stats import ks_2samp
feats = FEATURES + DWL
ks = pd.Series({c: ks_2samp(tr[c].dropna().sample(150_000, random_state=0), te[c].dropna()).statistic for c in feats}).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.barh(range(len(ks)), ks.values, color=C_NT); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks.index, fontsize=7); ax.invert_yaxis()
ax.axvline(.1, color=C_TIDAL, ls="--"); ax.text(.101, len(ks) - 1, "0.1 (흔히 '차이 있음'으로 보는 수준)", color=C_TIDAL, fontsize=8)
ax.set_xlabel("KS 통계량 (train vs test)"); ax.set_title(f"모든 변수가 0.1 아래 (최대 {ks.max():.3f}, {ks.idxmax()})"); plt.show()
''')
b.md("""
모든 변수의 차이가 작아요. test가 조금 더 습한 편이에요(강우 변수가 위쪽에 있어요).

### 2-2. 적대적 검증 — 분류기가 train과 test를 구분할 수 있나?
train에서 15만 행, test에서 15만 행을 뽑아 "이 행은 test인가?"를 맞히는 분류기를 만들어요. **사상 단위로 교차검증**해요(사상마다 고유한 비 패턴이 있어서, 행 단위로 섞으면 "사상 알아맞히기"가 돼 버려요).
""")
b.code(r'''
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
N = 150_000
a_tr, a_te = tr[feats + ["event_block"]].sample(N, random_state=0), te[feats + ["event_block"]].sample(min(N, len(te)), random_state=0)
X = pd.concat([a_tr, a_te], ignore_index=True); lab = np.r_[np.zeros(len(a_tr)), np.ones(len(a_te))]
oof = np.zeros(len(X)); fold_auc = []
for tri, vai in GroupKFold(5).split(X, lab, groups=X.event_block):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.1, max_leaf_nodes=31, random_state=0)
    oof[vai] = clf.fit(X.iloc[tri][feats], lab[tri]).predict_proba(X.iloc[vai][feats])[:, 1]
    fold_auc.append(roc_auc_score(lab[vai], oof[vai]) if len(np.unique(lab[vai])) == 2 else np.nan)
print(f"적대적 검증 AUC (사상 단위 5-fold): {roc_auc_score(lab, oof):.3f} · fold별 {np.round(fold_auc, 3)}")
''')
b.md("""
AUC가 **0.5 근처(이하)** 예요. 분류기가 train과 test를 **동전 던지기보다 낫게 구분하지 못해요.** 분포 차이는 무시해도 되는 수준이에요.
→ 교차검증 점수가 test 점수를 잘 대변할 거라고 기대할 수 있어요. **단, 검증을 제대로 나눴을 때만요**(다음 장).

## 3. 검증은 어떻게 나눠야 하나

### 3-1. 왜 무작위로 섞으면 안 되나 — 직접 확인
1권에서 봤듯이 **바로 옆 시각의 행끼리는 타깃이 보는 미래가 5시간 겹쳐요.** 그래서 14시 행의 답과 15시 행의 답은 거의 비슷해요.

아주 단순한 '예측기'로 확인해 볼게요.
> **"검증 행의 답 = 같은 관측소 1시간 전 행의 답"** (그 행이 학습 쪽에 있을 때만 베껴 쓰고, 없으면 학습 평균)

이건 실제로는 **쓸 수 없는** 정보예요(1시간 전 행의 답은 지금부터 5시간 뒤까지를 봐야 알 수 있으니까). 그런데 무작위로 섞으면 이런 베끼기가 가능해져요.
""")
b.code(r'''
from sklearn.model_selection import KFold
key = keyed(tr); y = tr[TARGET].to_numpy(float)
prev_y = shift_lookup(tr, key, TARGET, -1)                       # 같은 관측소 1시간 전 행의 타깃 (베끼기용)
res = {}
for nm, sp in [("무작위 KFold", KFold(5, shuffle=True, random_state=0)), ("사상 GroupKFold", GroupKFold(5))]:
    fold = np.zeros(len(tr), int)
    for f, (_, vai) in enumerate(sp.split(tr, groups=tr.event_block)):
        fold[vai] = f
    prev_fold = shift_lookup(tr.assign(f=fold), key, "f", -1)
    usable = np.isfinite(prev_y) & (prev_fold != fold)            # 1시간 전 행이 '학습 쪽'에 있나
    pred = np.where(usable, prev_y, y.mean())
    res[nm] = {"베끼기가 가능한 검증 행": usable.mean(), "RMSE": np.sqrt(np.mean((y - pred) ** 2))}
pd.DataFrame(res).T.round(3)
''')
b.md("""
- 무작위로 섞으면 검증 행의 **약 74%** 가 이웃 행의 답을 베낄 수 있어요. 아무것도 학습하지 않은 베끼기만으로 RMSE **0.241**이 나와요. 제대로 검증한 HGB 모델(0.226)과 거의 같은 수준이에요!
- 사상 단위로 나누면 베낄 수 있는 행이 **0%** 예요. 같은 호우는 통째로 한쪽에만 있으니까요.

진짜 모델(HGB)도 이런 베끼기와 비슷한 효과를 받아요. 원본 `src/s05_baseline.py`에서 같은 모델을 두 방식으로 검증한 결과예요.
""")
b.code(r'''
bl = pd.read_csv(RESULTS / "s05_baseline.csv")
pv = bl.pivot(index="method", columns="split", values="rmse").dropna()
sd = bl.pivot(index="method", columns="split", values="fold_std").loc[pv.index]
pv["낙관 정도"] = (1 - pv["무작위 KFold"] / pv["GroupKFold(사상)"]).map(lambda v: f"{v:.1%}")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
xx = np.arange(len(pv))
ax[0].bar(xx - .2, pv["GroupKFold(사상)"], .4, color=C_NT, label="사상 GroupKFold"); ax[0].bar(xx + .2, pv["무작위 KFold"], .4, color=C_ALL, label="무작위 KFold")
ax[0].set_xticks(xx); ax[0].set_xticklabels(pv.index, fontsize=7, rotation=15); ax[0].set_ylabel("RMSE"); ax[0].legend(); ax[0].set_title("무작위는 점수를 좋게 보여준다")
ax[1].bar(xx - .2, sd["GroupKFold(사상)"], .4, color=C_NT); ax[1].bar(xx + .2, sd["무작위 KFold"], .4, color=C_ALL)
ax[1].set_xticks(xx); ax[1].set_xticklabels(pv.index, fontsize=7, rotation=15); ax[1].set_ylabel("fold 간 표준편차"); ax[1].set_title("…그리고 흔들림을 1/14로 숨긴다")
fig.tight_layout(); plt.show()
pv.round(4)
''')
b.md("""
- 무작위 KFold는 HGB 모델에서 RMSE를 **약 11~12% 좋게** 보여줘요(0.198 vs 0.226).
- fold 간 흔들림도 **1/14**로 작게 보여서, 점수를 과하게 믿게 만들어요. 실제로는 사상에 따라 점수가 꽤 흔들려요(5장).

→ **검증은 사상 단위 GroupKFold로 고정**해요. 사상이 122개뿐이라 fold마다 흔들림이 크니, **여러 번 반복(repeated)** 해서 평균을 보는 걸 권해요.

## 4. 기준 모델은 어디서 틀리나 — 국면별

이제 제대로 나눈 검증(사상 GroupKFold)으로 만든 기준 모델의 OOF 예측(`cache/oof_s09.parquet`의 `oof_base`)으로 **오차가 사는 곳**을 찾아요.
행을 **입력만으로** 국면을 나눠요(타깃을 쓰지 않으니 test에도 똑같이 적용할 수 있어요).
- 비: 최근 6시간 강우(두 원천 중 하나라도) 0.5mm 이상
- 상승 중 / 하강 중 / 반응 없음: 최근 1시간 수위 변화가 +1cm 초과 / −1cm 미만 / 그 사이
""")
b.code(r'''
oofd = pd.read_parquet(CACHE / "oof_s09.parquet")[["station_code", "hour", "oof_base"]]
m = tr.merge(oofd, on=["station_code", "hour"])
m["e2"] = (m[TARGET] - m.oof_base) ** 2
wet = (m.imerg_rain_6h >= .5) | (m.sfc_rain_6h >= .5); x = m.dwl_1h
m["국면"] = np.select([m.station_id.isin(TIDAL + WEAK_TIDAL), m.station_id.isin(OPERATED_24H + STUCK),
                      ~wet & (x.abs() <= .01), ~wet & (x < -.01), ~wet & (x > .01),
                      wet & (x.abs() <= .01), wet & (x > .01), wet & (x < -.01)],
                     ["조석(강·약)", "24h 주기·고정", "무강우·정지", "무강우·하강", "무강우·상승",
                      "비·아직 반응 없음", "비·상승 중", "비·하강 중"], "기타")
tab = m.groupby("국면").agg(행=("e2", "size"), 오차합=("e2", "sum"), RMSE=("e2", lambda e: np.sqrt(e.mean())),
                          편향=(TARGET, "mean"))
tab["편향"] = (m[TARGET] - m.oof_base).groupby(m["국면"]).mean()
tab["행 비중"], tab["오차 비중"] = tab.행 / tab.행.sum(), tab.오차합 / tab.오차합.sum()
tab = tab.sort_values("오차 비중", ascending=False)
fig, ax = plt.subplots(figsize=(10, 3.8))
yy = np.arange(len(tab))
ax.barh(yy + .2, tab["행 비중"] * 100, .4, color=C_ALL, label="행 비중"); ax.barh(yy - .2, tab["오차 비중"] * 100, .4, color=C_TIDAL, label="오차 비중")
ax.set_yticks(yy); ax.set_yticklabels(tab.index); ax.invert_yaxis(); ax.set_xlabel("%"); ax.legend(); ax.set_title("'비·상승 중'은 행 16%인데 오차의 절반"); plt.show()
tab[["행 비중", "오차 비중", "RMSE", "편향"]].round(3)
''')
b.md("""
- **"비·상승 중"** 이 행의 약 16%인데 오차의 **약 절반**이에요.
- 무강우 구간은 행은 많은데 오차는 몇 %뿐이에요. **이미 거의 다 맞히고 있어요.**
- 편향(실제 − 예측의 평균)은 모든 국면에서 **거의 0**이에요. 한쪽으로 치우쳐 틀리는 게 아니라, **"얼마나 더 오를지"의 불확실성**에서 오차가 나와요.

### 4-1. 더 잘게 — 강우 × 수위 변화 격자
비조석 관측소만, 강우 7칸 × 최근 수위 변화 8칸으로 나눠 **평균 잔차(빨강 = 모델이 덜 예측)** 와 **오차 비중**을 봐요.
""")
b.code(r'''
from matplotlib.colors import LinearSegmentedColormap
d = m[~m.station_id.isin(SPECIAL)]
rain = np.maximum(d.imerg_rain_6h, d.sfc_rain_6h)
rb = pd.cut(rain, [-.01, .5, 2, 5, 10, 20, 50, 1e4], labels=["<0.5", "0.5-2", "2-5", "5-10", "10-20", "20-50", "50+"])
xb = pd.cut(d.dwl_1h, [-1e3, -.1, -.03, -.01, .01, .03, .1, .3, 1e3], labels=["<-0.1", "-0.1~-0.03", "-0.03~-0.01", "±0.01", "0.01~0.03", "0.03~0.1", "0.1~0.3", ">0.3"])
r = d[TARGET] - d.oof_base
bias = r.groupby([xb, rb], observed=False).mean().unstack()
sse = (r ** 2).groupby([xb, rb], observed=False).sum().unstack() / (r ** 2).sum()
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
cm_div = LinearSegmentedColormap.from_list("rb", ["#184f95", "#6da7ec", "#f0efec", "#f09a8f", "#b3261e"])
cm_seq = LinearSegmentedColormap.from_list("s", ["#fcfcfb", "#9ec5f4", "#3987e5", "#0d366b"])
for a, M, cmap, vmin, vmax, fmt, ttl in [(ax[0], bias, cm_div, -.4, .4, "{:+.2f}", "평균 잔차 (빨강 = 과소예측)"),
                                         (ax[1], sse * 100, cm_seq, 0, None, "{:.1f}", "오차 비중 (%)")]:
    im = a.imshow(M.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    for (i, j), v in np.ndenumerate(M.to_numpy(float)):
        if np.isfinite(v): a.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7)
    a.set_xticks(range(7)); a.set_xticklabels(rb.cat.categories); a.set_yticks(range(8)); a.set_yticklabels(xb.cat.categories)
    a.set_xlabel("최근 6h 강우 (mm)"); a.set_ylabel("최근 1h 수위 변화 (m)"); a.set_title(ttl); a.grid(False); fig.colorbar(im, ax=a, shrink=.8)
fig.tight_layout(); plt.show()
''')
b.md("""
- 오차는 **두 곳**에 몰려 있어요.
  1. **오른쪽 위**(강한 비 + 이미 오르는 중): "얼마나 더 오를까"를 모르는 오차
  2. **가운데 줄(±0.01) × 비 2~50mm**: 비는 오는데 아직 반응이 없는 행이에요. 칸마다 5~7%씩 쌓여요. "**언제** 오르기 시작할까"를 모르는 오차예요.
- 체계적으로 **덜 예측(빨강)** 하는 곳은 주로 **"1시간에 30cm 넘게 급상승"** 줄이에요. 급상승이 얼마나 더 이어질지를 모델이 짧게 잡아요.

→ 개선은 **상승 국면**과 **반응 시작 시점**에 집중해요. 지체시간(3권)과 유역 크기를 이용한 '곧 반응할 것' 신호가 두 번째 오차 덩어리를 노려요. 예: 상승 국면 가중치, 상승 전용 모델, 상승 가속도(`dwl_1h − dwl_3h/3`) 같은 변수.

## 5. 어떤 호우에서 틀리나 — 사상별
""")
b.code(r'''
e = m.groupby("event_block").agg(오차합=("e2", "sum"), 행=("e2", "size"), rmse=("e2", lambda v: np.sqrt(v.mean())),
                                 hours=("pos_in_block", "max"), peak=("imerg_rain_1h", "max"))
e["hours"] += 1
e["오차 비중"], e["행 비중"] = e.오차합 / e.오차합.sum(), e.행 / e.행.sum()
srt = e.sort_values("오차 비중", ascending=False)
cs, cr = srt["오차 비중"].cumsum().to_numpy(), srt["행 비중"].cumsum().to_numpy()
n_half = int((cs < .5).sum() + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(np.arange(1, len(cs) + 1), cs * 100, color=C_TIDAL, label="오차 누적"); ax[0].plot(np.arange(1, len(cr) + 1), cr * 100, "--", color=C_ALL, label="행 누적")
ax[0].axvline(n_half, color="k", lw=.6, ls=":"); ax[0].set_xlabel("사상 (오차 큰 순)"); ax[0].set_ylabel("누적 (%)"); ax[0].legend()
ax[0].set_title(f"{len(e)}개 사상 중 {n_half}개가 오차의 절반")
ax[1].scatter(e.hours, e.rmse, s=e["행 비중"] * 3000 + 5, color=C_NT, alpha=.6)
for bk in srt.index[:5]:
    ax[1].annotate(f"#{bk}", (e.hours[bk], e.rmse[bk]), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax[1].set_xscale("log"); ax[1].set_xlabel("사상 길이 (h, log)"); ax[1].set_ylabel("사상 RMSE"); ax[1].set_title("길고 강한 호우일수록 어렵다 (원 = 행 비중)")
fig.tight_layout(); plt.show()
print(f"상위 5개 사상: 행 {srt['행 비중'].head(5).sum():.0%} → 오차 {srt['오차 비중'].head(5).sum():.0%}")
print(f"사상 RMSE와의 순위 상관: 길이 {e.rmse.corr(e.hours, method='spearman'):.2f} · 최대 1h 강우(관측소 최대) {e.rmse.corr(e.peak, method='spearman'):.2f}")

# test 사상도 입력만으로 같은 잣대(길이·최대 강우)로 비교
et = te.groupby("event_block").agg(hours=("pos_in_block", "max"), peak=("imerg_rain_1h", "max")); et["hours"] += 1
print(f"test 사상 {len(et)}개 중 train 상위 10% 길이를 넘는 사상: {(et.hours > e.hours.quantile(.9)).sum()}개 · 상위 10% 최대 강우를 넘는 사상: {(et.peak > e.peak.quantile(.9)).sum()}개")
''')
b.md("""
- **몇 개의 큰 호우가 오차의 절반**을 만들어요. 가장 긴 사상(155시간) 하나만 해도 오차의 약 20%예요.
- 사상 RMSE는 **길이·강도**와 함께 커져요.
- test에도 train 상위권에 드는 길고 강한 사상이 몇 개 있어요(입력만으로 비교).

→ 리더보드 점수는 **소수의 큰 호우에서 갈려요.** 그래서
1. 반복 GroupKFold로 **사상별 RMSE 분포**를 같이 봐요.
2. **상위 5개 사상을 뺀 점수**도 봐서, 큰 호우 몇 개에 과적합하지 않았는지 확인해요(4권에서 본 "섞은 그림 vs 사상별 그림" 문제와 같은 뿌리예요).

## 6. 미래 강우 탓은 얼마나 되나 — 진단용 실험

남은 오차 중 **"앞으로 6시간 동안 올 비를 몰라서"** 생기는 몫은 얼마일까요? 원본 `src/s10_events_residuals.py`에서 **미래 6시간 강우를 일부러 넣은 '오라클' 모델**과 기준 모델을 비교했어요.

⚠️ **이 실험은 미래 정보를 일부러 쓴 진단이에요. 모델 변수로 쓰자는 뜻이 절대 아니에요.** "현재 정보로 더 줄일 수 있는 오차가 얼마나 남았나"를 가늠하려는 것뿐이에요.
""")
b.code(r'''
import json
fr = json.load(open(RESULTS / "s10_events_residuals.json", encoding="utf-8"))["future_rain"]
o = fr["oracle"]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar([0, 1], [o["rmse_base"], o["rmse_oracle"]], color=[C_ALL, C_NT], width=.5)
for i, v in enumerate([o["rmse_base"], o["rmse_oracle"]]):
    ax.text(i, v + .003, f"{v:.3f}", ha="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["기준 모델\n(현재 입력)", "오라클\n(+ 미래 6h 강우, 진단용)"]); ax.set_ylabel("RMSE")
ax.set_title(f"미래 비를 완벽히 알아도 MSE는 {o['mse_reduction']:.0%}만 준다"); plt.show()
print(f"(미래 6행이 모두 있는 {fr['rows_with_full_future']:.0%}의 행에서 비교) 비조석 MSE 감소 {o['nontidal_mse_reduction']:.0%}")
''')
b.md("""
미래 비를 **완벽히** 알아도 오차(MSE)는 약 **30%** 만 줄어요. 거꾸로 말하면 남은 오차의 **약 70%는 미래 비 탓이 아니에요.** 현재 입력을 아직 덜 활용했거나, 원래 예측할 수 없는 잡음이에요.
→ **현재 정보로 더 줄일 여지가 크다**는 뜻이에요. 1~5권에서 나온 파생변수 후보들이 그 여지를 노리는 거예요.

## 7. 누수 점검과 시리즈 전체 정리

### 누수 점검표 (6권)

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 2장 KS·적대적 검증 | 입력값만 | 누수 없음 |
| 3-1 베끼기 실험 | **1시간 전 행의 타깃**(= 미래 5시간을 본 값) | 누수가 **어떻게 생기는지 보여주려는** 실험. 모델에 쓰면 안 됨 |
| 3장 기준선 비교 | 관측소 통계는 fold 안에서 계산 | 누수 없음 |
| 4장 국면 정의 | **입력만**으로 정의 | test에도 적용 가능 |
| 5장 사상 길이 | 사상 전체 길이(미래 포함) | 분석용. **모델 변수로 쓰면 누수**(쓸 수 있는 건 '시작 후 경과 시간'뿐) |
| 6장 오라클 | **미래 강우를 일부러 사용** | 진단용. 모델 변수 금지 |

### 시리즈 전체 정리 — 모델링 체크리스트

| 분류 | 할 일 | 근거 |
|---|---|---|
| **검증** | 사상 단위 (반복) GroupKFold. 전체·비조석·상위 5사상 제외 RMSE를 같이 봄 | 6권 3·5장 |
| **누수 방지** | 시각 t 이후 정보 금지(test의 뒤 시각 행, 사상 길이·끝, 사상 전체 순위). 타깃을 쓰는 관측소 통계는 fold 안에서만 | 1권 5장, 별책 4-6, 6권 |
| **전처리** | ST0027 학습 제외·예측 0. 수위는 차분. 강우 `log1p`(선형 모델) | 1권 |
| **핵심 변수** | `dwl_1h`, `dwl_3h`, 짧은 창 강우, 레이더 하나 | 2·3권 |
| **파생 변수** | 상승 속도·가속도, 사상 경과 시간, 사상 누적 강우, 유역면적 × 강우 | 3·4권 |
| **관측소 특성** | 행동 지문(자기상관, 무강우 변동, 반응 세기, 12·24h 주기) — fold 안에서 | 4권, 별책 |
| **조석** | 주기 세기(자기상관 값) + 위상 차분(`wl_t − wl_t−6h`, `−12h`). 연결망 변수는 제외 | 별책, 5권 |
| **연결망 (규정 확인 후)** | 안정적인 선행 관계만, 또는 주변 여러 관측소 요약 | 5권 |
| **개선의 초점** | "비·상승 중" 국면, 급상승 지속, 큰 호우 | 6권 4·5장 |
| **모델** | 트리 계열(비선형·상호작용), 큰 비 구간 가중치·분위수 실험 | 2권 |
""")
b.write("nb6_validation_and_errors.ipynb")

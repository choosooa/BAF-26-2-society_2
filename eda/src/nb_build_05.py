"""5권: 관측소 연결망(상류→하류 선행)."""
from nb_common import Book

b = Book()
b.md("""
# 🕸️ 5권 · 연결망 — "상류의 물은 하류에 먼저 신호를 보낸다"

> 시리즈: 1권 구조·품질 → 2권 타깃과 강우 → 3권 수위 변화·시계열 → 4권 관측소 유형·유역환경 → **5권 연결망** → 6권 검증 전략·오차 위치 · 별책 조석

### 한 줄 요약
**관측소 좌표가 없어도, 수위 변화의 '시차 상관'만으로 "누가 누구보다 먼저 움직이나"를 찾을 수 있어요. 일반 관측소의 절반 이상에 1~2시간 먼저 움직이는 관측소가 있고, 뒤따르는 쪽이 대체로 더 큰 유역(하류)이에요. 다만 '누가 선행인가'는 호우마다 꽤 바뀌어요. 사상을 반씩 나누면 같은 선행이 나오는 곳은 약 1/3이에요. 이 정보를 넣으면 모델 오차가 약 2~3% 줄어요. 다만 조석 관측소에서는 오히려 나빠지고, test에서 다른 관측소의 같은 시각 값을 쓰는 게 규정상 괜찮은지 확인이 필요해요.**

| 장 | 질문 |
|---|---|
| 1 | 아이디어 — 상류가 먼저 움직이면 시차 상관에 봉우리가 생긴다 (가짜 데이터) |
| 2 | 함정 — 호우 때는 모두 같이 움직인다 → 공통 신호 빼기 |
| 3 | 465곳 전부에서 선행 관측소 찾기 |
| 4 | 찾은 관계가 진짜인가 — 방향(하류가 더 큰가)은 ✅, 안정성(사상을 나눠도 같은가)은 약 1/3만 |
| 5 | 모델에 도움이 되나 — 관측소 전부·사상 전부로 확인 |
| 6 | 누수·규정 점검과 정리 |
""")
b.setup()
b.code(r'''
cols = ["station_id", "station_code", "hour", "event_block", TARGET, "dwl_1h", "wsarea", "stream_order"] + ENV[:1]
tr = add_flags(pd.read_parquet(CACHE / "train_sorted.parquet", columns=list(dict.fromkeys(cols))))
cats = sorted(tr.station_id.unique()); code_of = {s: i for i, s in enumerate(cats)}
print(f"train {len(tr):,}행 · 관측소 {len(cats)}곳 · 서로 다른 시각 {tr.hour.nunique():,}개")
''')

b.md("""
## 1. 아이디어 — 가짜 데이터로 먼저

같은 강에서 **상류 A**와 **하류 B**가 있다고 해요. A에서 물이 불면, 그 물이 흘러 내려가 **2시간 뒤 B**에서도 불어요.
그러면 "A의 지금 변화"와 "B의 **L시간 뒤** 변화"의 상관을 L = 0, 1, 2 … 로 재면 **L = 2에서 봉우리**가 생겨요.
""")
b.code(r'''
rng = np.random.default_rng(0)
n = 400
rain = np.where(rng.random(n) < .06, rng.gamma(2, 1, n), 0)
A = np.convolve(rain, [.2, .5, .3])[:n] + rng.normal(0, .05, n)          # 상류: 비에 빨리 반응
B = np.r_[np.zeros(2), A[:-2]] * .9 + rng.normal(0, .05, n)             # 하류: A를 2시간 늦게 따라감
lags = range(0, 7)
cc = [np.corrcoef(A[:n - L], B[L:])[0, 1] for L in lags]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(A[:80], color=C_TIDAL, label="상류 A"); ax[0].plot(B[:80], color=C_NT, label="하류 B"); ax[0].legend(); ax[0].set_xlabel("시간 (h)")
ax[0].set_title("B는 A를 2시간 늦게 따라간다")
ax[1].bar(list(lags), cc, color=[C_TIDAL if L == 2 else C_ALL for L in lags]); ax[1].set_xlabel("L (h)"); ax[1].set_ylabel("corr(A(t), B(t+L))")
ax[1].set_title("시차 상관: L = 2에서 봉우리"); fig.tight_layout(); plt.show()
''')

b.md("""
## 2. 함정 — 호우 때는 모두 같이 움직인다

실제 데이터에서 그냥 시차 상관을 재면 문제가 생겨요. **큰 호우가 오면 전국 하천이 거의 동시에 불어나요.** 그러면 강이 서로 연결되지 않은 관측소끼리도 상관이 높게 나와요.

**예시**: 서울 하천과 부산 하천은 연결돼 있지 않지만, 전국적인 장마 때는 둘 다 올라서 상관이 높아요.

그래서 두 단계로 **공통 신호를 빼요**.
1. **관측소마다 크기 맞추기**: 조석처럼 크게 움직이는 곳이 상관을 좌우하지 않도록, 관측소마다 `dwl_1h`를 그 관측소의 전형적인 변동폭(MAD)으로 나눠요.
2. **시각마다 전체 중앙값 빼기**: 같은 시각에 모든 관측소가 같이 오른 부분(공통 호우 신호)을 빼요. 남은 건 "그 관측소만의 움직임"이에요.
""")
b.code(r'''
hours = np.sort(tr.hour.unique()); hidx = pd.Series(np.arange(len(hours)), index=hours)
code = tr.station_id.map(code_of).to_numpy()
x = tr.dwl_1h.to_numpy(float)
g = pd.Series(x).groupby(code)
mad = g.apply(lambda v: np.median(np.abs(v - np.median(v)))) * 1.4826
scale = np.maximum(np.maximum(mad, .5 * g.std()), .005)                 # 관측소별 전형적 변동폭 (너무 작으면 하한)
Zraw = np.full((len(hours), len(cats)), np.nan)
Zraw[hidx.reindex(tr.hour.to_numpy()).to_numpy(), code] = np.clip(x / scale.reindex(code).to_numpy(), -8, 8)
Z = Zraw - np.nanmedian(Zraw, axis=1, keepdims=True)                     # 시각마다 전 관측소 중앙값(공통 호우 신호) 빼기

def masked_corr(A_, B_, min_n=150):
    """열(관측소)끼리 모든 쌍의 상관. NaN은 빼고 계산. 결과 C[a, b] = corr(A_의 a열, B_의 b열)"""
    MA, MB = ~np.isnan(A_), ~np.isnan(B_)
    A0, B0 = np.where(MA, A_, 0.), np.where(MB, B_, 0.)
    fa, fb = MA.astype(float), MB.astype(float)
    n_ = fa.T @ fb; sa, sb = A0.T @ fb, fa.T @ B0
    saa, sbb, sab = (A0 ** 2).T @ fb, fa.T @ (B0 ** 2), A0.T @ B0
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (sab - sa * sb / n_) / np.sqrt((saa - sa ** 2 / n_) * (sbb - sb ** 2 / n_))
    C[n_ < min_n] = np.nan
    return C

C0_raw, C0 = masked_corr(Zraw, Zraw), masked_corr(Z, Z)
iu = np.triu_indices(len(cats), 1)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(C0_raw[iu][np.isfinite(C0_raw[iu])], bins=80, histtype="step", lw=1.6, color=C_ALL, label=f"공통 신호 그대로 (중앙값 {np.nanmedian(C0_raw[iu]):.2f})")
ax.hist(C0[iu][np.isfinite(C0[iu])], bins=80, histtype="step", lw=1.6, color=C_NT, label=f"공통 신호 뺀 뒤 (중앙값 {np.nanmedian(C0[iu]):.2f})")
ax.set_xlabel("관측소 쌍의 같은 시각 상관"); ax.set_ylabel("쌍 수"); ax.legend(); ax.set_title("공통 신호를 빼면 '아무나 비슷한' 상관이 사라진다"); plt.show()
''')
b.md("""
공통 신호를 빼기 전엔 대부분의 관측소 쌍이 양의 상관을 보여요(전국이 같이 움직여서). 뺀 뒤엔 **0 근처로 모이고, 진짜 연결된 소수의 쌍만** 높게 남아요.

## 3. 465곳 전부에서 선행 관측소 찾기

모든 관측소 쌍(465 × 465)에 대해 시차 L = 0~8시간 상관을 계산해요. **같은 사상 안에서만** 짝을 지어요(사상 사이 공백을 건너뛰지 않도록).

**선행 관측소 판정 규칙** (관측소 B마다):
- 다른 관측소 A 중에서 "A(t)와 B(t+L)의 상관"(L ≥ 1)이 가장 큰 곳을 찾아요.
- 그 상관이 **같은 시각 상관보다 0.05 이상 커야** 해요(동시에 움직이는 게 아니라 **먼저** 움직여야 하니까).
- 상관이 **0.5 이상**이면 "뚜렷한 선행 관측소"라고 불러요.
- 조석·약한 조석·24h 주기·고정 관측소는 제외해요(비와 무관한 주기가 있어서 가짜 선행이 생겨요).
""")
b.code(r'''
MAXLAG = 8
SPECIAL_CODES = np.array(sorted(code_of[s] for s in SPECIAL))

def lagged_corrs(Z_, row_ok=None):
    if row_ok is not None:
        Z_ = np.where(row_ok[:, None], Z_, np.nan)
    Cs = {}
    for L in range(0, MAXLAG + 1):
        i = np.arange(len(hours) - L)
        ok = hours[i + L] - hours[i] == L                  # 정확히 L시간 떨어진 시각끼리만 (= 같은 사상 안)
        Cs[L] = masked_corr(Z_[i[ok]], Z_[i[ok] + L])       # Cs[L][a, b] = corr(a(t), b(t+L))
    return Cs

def find_leaders(Cs, exclude):
    n_ = len(cats)
    stack = np.stack([Cs[L] for L in range(1, MAXLAG + 1)])            # (시차, 선행 후보, 후행)
    stack = np.where(np.isnan(stack), -np.inf, stack)
    best, lag = stack.max(axis=0), stack.argmax(axis=0) + 1
    best[np.arange(n_), np.arange(n_)] = -np.inf; best[exclude, :] = -np.inf
    gain = best - np.nan_to_num(Cs[0], nan=0.)
    cand = np.where(gain > .05, best, -np.inf)                          # 동시 상관보다 확실히 커야 '선행'
    lead, val = cand.argmax(axis=0), cand.max(axis=0)
    out = pd.DataFrame({"leader": [cats[i] for i in lead], "corr": val, "lag": lag[lead, np.arange(n_)]}, index=cats)
    out.loc[~np.isfinite(out["corr"]) | out.index.isin(SPECIAL), ["leader", "corr", "lag"]] = np.nan
    return out

Cs = lagged_corrs(Z)
L = find_leaders(Cs, SPECIAL_CODES)
considered = L.index[~L.index.isin(SPECIAL)]
strong = L[L["corr"] >= .5]
print(f"대상 관측소 {len(considered)}곳 중 뚜렷한 선행 관측소가 있는 곳: {len(strong)}곳 ({len(strong) / len(considered):.0%})")
vc = strong.lag.value_counts(normalize=True).sort_index()
print("선행 간격 분포:", {f"{int(k)}h": f"{v:.0%}" for k, v in vc.items()})
''')
b.code(r'''
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].bar(vc.index, vc.values * 100, color=C_NT); ax[0].set_xlabel("선행 간격 (h)"); ax[0].set_ylabel("관측소 비율 (%)")
ax[0].set_title(f"뚜렷한 선행이 있는 {len(strong)}곳")
ex = strong[strong.lag >= 2].sort_values("corr", ascending=False).iloc[0]
fol, lead, lg = ex.name, ex.leader, int(ex.lag)
blk = tr[tr.station_id == fol].groupby("event_block").size().idxmax()
for sid, col, sh, lab in [(lead, C_TIDAL, lg, f"{lead} (선행, {lg}h 늦춰 그림)"), (fol, C_NT, 0, f"{fol} (후행)")]:
    s = tr[(tr.station_id == sid) & (tr.event_block == blk)].sort_values("hour")
    ax[1].plot(s.hour - s.hour.min() + sh, s.dwl_1h, color=col, lw=1.3, label=lab)
ax[1].set_xlabel(f"사상 #{blk} 경과 시간 (h)"); ax[1].set_ylabel("dwl_1h (m)"); ax[1].legend(fontsize=8)
ax[1].set_title(f"예: 선행 관측소를 {lg}시간 늦추면 겹친다 (상관 {ex['corr']:.2f})"); fig.tight_layout(); plt.show()
''')
b.md("""
- 일반 관측소의 **절반 이상**에 뚜렷한 선행 관측소가 있어요.
- 선행 간격은 **1시간이 대부분, 그다음 2시간**이에요. 먼 상류보다 **바로 위 관측소**가 주로 잡혀요.
- 오른쪽 예시처럼 선행 관측소의 움직임을 몇 시간 늦추면 후행 관측소와 거의 겹쳐요.

## 4. 찾은 관계가 진짜인가

좌표가 없으니 "진짜 상류–하류인지" 직접 확인할 수는 없어요. 대신 두 가지로 간접 확인해요.

### 4-1. 방향 — 뒤따르는 쪽이 더 큰 유역(하류)인가? (선행 쌍 전부)
물은 작은 유역(상류)에서 큰 유역(하류)으로 흘러요. 진짜 상류–하류라면 **뒤따르는 관측소의 상류면적이 더 커야** 해요. 무작위로 짝지었다면 50%일 거예요.
유역환경이 **똑같은** 쌍(같은 소권역)은 비교할 수 없으니 빼요.
""")
b.code(r'''
env = tr.groupby("station_id")[["wsarea", "stream_order"] + ENV[:1]].first()
pairs = strong.assign(ws_f=env.wsarea.reindex(strong.index).to_numpy(), ws_l=env.wsarea.reindex(strong.leader).to_numpy(),
                      so_f=env.stream_order.reindex(strong.index).to_numpy(), so_l=env.stream_order.reindex(strong.leader).to_numpy())
diff_env = ~np.isclose(pairs.ws_f, pairs.ws_l)
p_area = (pairs.ws_f > pairs.ws_l)[diff_env].mean()
p_order = (pairs.so_f > pairs.so_l)[diff_env & ~np.isclose(pairs.so_f, pairs.so_l)].mean()
print(f"유역이 다른 선행 쌍 {int(diff_env.sum())}개 중 뒤따르는 쪽 상류면적이 더 큰 비율: {p_area:.0%} (무작위 50%)")
print(f"하천차수가 다른 쌍 중 뒤따르는 쪽 차수가 더 큰 비율: {p_order:.0%} (무작위 50%)")
from scipy.stats import binomtest
print(f"이항검정 p값 (면적): {binomtest(int((pairs.ws_f > pairs.ws_l)[diff_env].sum()), int(diff_env.sum()), .5).pvalue:.1e}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(np.log10(pairs.ws_l[diff_env]), np.log10(pairs.ws_f[diff_env]), s=14, color=C_NT, alpha=.7)
lim = [np.log10(env.wsarea.min()), np.log10(env.wsarea.max())]; ax.plot(lim, lim, ":", color=C_ALL)
ax.set_xlabel("선행 관측소 상류면적 (log10)"); ax.set_ylabel("후행 관측소 상류면적 (log10)")
ax.set_title("대각선 위 = 뒤따르는 쪽이 더 큰 유역"); plt.show()
''')
b.md("""
뒤따르는 쪽이 더 큰 유역인 비율이 **약 2/3**로, 무작위(50%)보다 뚜렷하게 높아요. **좌표 없이도 상류 → 하류 방향이 복원**된다는 뜻이에요. (100%가 아닌 건 합류점, 지류끼리의 관계, 비의 이동 방향 같은 다른 이유도 섞이기 때문이에요.)

### 4-2. 안정성 — 사상을 반으로 나눠 따로 찾아도 같은 선행 관측소가 나오나?
train 사상을 짝수·홀수 번호로 반씩 나누고, 각각에서 따로 선행 관측소를 찾아요. 진짜 관계라면 **두 절반에서 같은 관측소**가 나와야 해요.
""")
b.code(r'''
blk_of_hour = tr.groupby("hour").event_block.first().reindex(hours).to_numpy()
half = {}
for nm, keep in [("짝수 사상", blk_of_hour % 2 == 0), ("홀수 사상", blk_of_hour % 2 == 1)]:
    half[nm] = find_leaders(lagged_corrs(Z, keep), SPECIAL_CODES)
h1, h2 = half["짝수 사상"], half["홀수 사상"]
both = strong.index[(h1.reindex(strong.index)["corr"] >= .4) & (h2.reindex(strong.index)["corr"] >= .4)]
same = (h1.loc[both, "leader"] == h2.loc[both, "leader"])
print(f"전체에서 뚜렷한 선행이 있던 {len(strong)}곳 중, 두 절반 모두에서 선행(상관 ≥ 0.4)이 잡힌 곳: {len(both)}곳")
print(f"그중 두 절반의 선행 관측소가 같은 곳: {same.mean():.0%}")
print(f"같은 곳일 때 선행 간격도 같은 비율: {(h1.loc[both[same], 'lag'] == h2.loc[both[same], 'lag']).mean():.0%}")
print(f"(참고) 무작위로 고른다면 같을 확률: 약 {1 / len(considered):.1%}")

# 다를 때: 두 선행 관측소가 사실상 같은 신호를 내는 '쌍둥이'인가?
C0 = Cs[0]
dif = both[~same.values]
twin = np.array([C0[code_of[h1.loc[s_, "leader"]], code_of[h2.loc[s_, "leader"]]] for s_ in dif])
print(f"다른 곳 {len(dif)}곳: 두 선행 관측소끼리 같은 시각 상관 중앙값 {np.nanmedian(twin):.2f} (0.5 넘는 경우 {np.mean(twin > .5):.0%})")

# 안정(두 절반 같음) vs 불안정 선행 쌍의 성질 비교
stable_ids = both[same.values]
pp = pairs[diff_env].assign(안정=lambda q: q.index.isin(stable_ids))
cmp = pp.groupby("안정").apply(lambda q: pd.Series({"쌍 수": len(q), "후행이 더 큰 유역": (q.ws_f > q.ws_l).mean(),
                                                   "상관 중앙값": q["corr"].median(), "1시간 간격 비율": (q.lag == 1).mean()}))
cmp.index = cmp.index.map({True: "안정 (두 절반 같음)", False: "불안정"})
cmp.round(2)
''')
b.md("""
**결과가 예상보다 불안정해요.** 솔직하게 정리하면 이래요.
- 두 절반에서 **같은 선행 관측소가 나온 곳은 약 1/3**이에요. 무작위(0.2%)보다는 훨씬 높지만, "대부분"은 아니에요.
- 다른 관측소가 나온 경우, 두 관측소는 **쌍둥이가 아니에요**(같은 시각 상관 중앙값 약 0.3). 이름만 다르고 사실상 같은 관측소인 경우가 아니라는 뜻이에요.
- **두 절반에서 같은 선행이 나온 관계(안정)** 는 상관이 더 강하고, 1시간 간격이 더 많고, 뒤따르는 쪽이 더 큰 유역인 비율도 조금 더 높아요(위 표).

**해석**: 연결망에는 두 층이 섞여 있어요.
1. **고정된 하천 구조**(바로 위 관측소 → 바로 아래 관측소): 어떤 호우에서든 같게 나와요. 안정적인 약 1/3이 여기에 가까워요.
2. **호우마다 달라지는 관계**: 비구름이 어느 방향으로 이동했는지에 따라, 이번 호우에서 "먼저 젖은" 관측소가 바뀌어요.
   - **예시**: 서쪽에서 들어온 비는 서쪽 관측소를 먼저, 남쪽에서 올라온 비는 남쪽 관측소를 먼저 불려요.

→ 모델에 쓸 때는 **안정적인 선행 관계만 고르거나**, 한 관측소 대신 **주변 여러 관측소의 요약**을 쓰는 편이 새 호우(test)에서 덜 흔들릴 거예요.

## 5. 모델에 도움이 되나

원본 `src/s09_network.py`에서 두 모델을 **사상 단위 5-fold 교차검증**으로 비교해 예측값을 저장해 뒀어요(`cache/oof_s09.parquet`).
- **기준 모델**: HGB, 원변수 + 수위 차분 + 관측소 타깃 통계
- **연결망 모델**: 기준 + 선행 관측소의 같은 시각 `dwl_1h`·`dwl_3h`·강우, 선행 간격·상관, 같은 시각 전체 관측소 평균 강우·상승 비율
- ⚠️ 선행 관측소는 **fold마다 학습 사상만으로 다시 찾았어요**(검증 사상의 정보가 새지 않게).

이번엔 전체 점수뿐 아니라 **관측소마다, 사상마다** 좋아졌는지 봐요.
""")
b.code(r'''
oof = pd.read_parquet(CACHE / "oof_s09.parquet")
m = tr[["station_id", "station_code", "hour", "event_block", TARGET, "group"]].merge(oof, on=["station_code", "hour"])
m["e_base"], m["e_net"] = (m[TARGET] - m.oof_base) ** 2, (m[TARGET] - m.oof_net) ** 2
rmse = lambda e: float(np.sqrt(e.mean()))
seg = {"전체": m, "비조석": m[~m.station_id.isin(TIDAL + STUCK)], "조석": m[m.station_id.isin(TIDAL)],
       "비조석·큰 상승(>0.5m)": m[~m.station_id.isin(TIDAL + STUCK) & (m[TARGET] > .5)]}
t = pd.DataFrame({k: {"기준 RMSE": rmse(d.e_base), "연결망 RMSE": rmse(d.e_net)} for k, d in seg.items()}).T
t["변화"] = (t["연결망 RMSE"] / t["기준 RMSE"] - 1).map(lambda v: f"{v:+.1%}")
t.round(4)
''')
b.code(r'''
per_st = m.groupby("station_id").agg(b=("e_base", "mean"), n=("e_net", "mean"))
per_st["ratio"] = np.sqrt(per_st.n / per_st.b)
per_st["has_leader"] = per_st.index.isin(strong.index)
per_ev = m[~m.station_id.isin(TIDAL + STUCK)].groupby("event_block").agg(b=("e_base", "mean"), n=("e_net", "mean"), rows=("e_base", "size"))
per_ev["ratio"] = np.sqrt(per_ev.n / per_ev.b)

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
gen = per_st[~per_st.index.isin(SPECIAL)]
for flag, col, lab in [(True, C_NT, "뚜렷한 선행 있음"), (False, C_ALL, "없음")]:
    v = gen[gen.has_leader == flag].ratio
    ax[0].hist(v.clip(.8, 1.2), bins=np.linspace(.8, 1.2, 41), color=col, alpha=.6, label=f"{lab} ({len(v)}곳, 개선 {(v < 1).mean():.0%})")
ax[0].axvline(1, color="k", lw=.6); ax[0].set_xlabel("관측소별 RMSE 비율 (연결망 ÷ 기준, 1보다 작으면 개선)"); ax[0].set_ylabel("관측소 수"); ax[0].legend(fontsize=8)
ax[0].set_title("일반 관측소 각각")
ax[1].scatter(per_ev.rows, per_ev.ratio, s=14, color=np.where(per_ev.ratio < 1, C_NT, C_TIDAL))
ax[1].axhline(1, color="k", lw=.6); ax[1].set_xscale("log"); ax[1].set_xlabel("사상의 행 수 (log)"); ax[1].set_ylabel("RMSE 비율")
ax[1].set_title(f"비조석 사상 {len(per_ev)}개 각각 (파랑 = 개선)"); fig.tight_layout(); plt.show()
print(f"개선된 일반 관측소: 선행 있음 {(gen[gen.has_leader].ratio < 1).mean():.0%} · 없음 {(gen[~gen.has_leader].ratio < 1).mean():.0%}")
print(f"개선된 비조석 사상: {(per_ev.ratio < 1).mean():.0%} · 행 가중 {(per_ev.rows * (per_ev.ratio < 1)).sum() / per_ev.rows.sum():.0%}")
print(f"조석 15곳 중 나빠진 곳: {(per_st.loc[TIDAL].ratio > 1).sum()}곳")
''')
b.md("""
- 전체 RMSE는 약 **2.4%**, 비조석은 약 **2.9%**, 비조석 큰 상승은 약 **3.4%** 줄어요. 조석은 오히려 **나빠져요.**
- **관측소별**: 선행 관측소가 있는 곳이 없는 곳보다 **개선된 비율이 높아요.** 연결망 변수가 그 관측소에 실제로 쓸모 있는 정보를 준다는 뜻이에요.
- **사상별**: 대부분 사상에서 개선돼요(위 출력 비율). 몇몇 사상에서만 좋아진 게 아니에요.
- **조석**: 조석은 비와 무관한 주기라서, 다른 관측소 정보가 오히려 방해가 돼요. → 연결망 변수는 **조석 관측소에는 빼는** 게 좋아요.

**얼마나 멀리 내다보나?** 선행 간격이 1~2시간이라, 6시간 예측창의 **앞부분만** 보강해요. 뒷부분은 여전히 강우 정보가 담당해요.

## 6. 누수·규정 점검과 정리

### 누수 점검표

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 2~3장 공통 신호 제거·시차 상관 | train 전체 시각의 수위 변화(입력값) | **연결망 찾기는 train에서만.** 모델용은 fold마다 학습 사상으로만 다시 찾음(5장 결과가 그렇게 만든 것) |
| 4-2 안정성 점검 | 사상을 나눠 각각 계산 | 누수 없음 |
| 5장 연결망 변수 | 선행 관측소의 **같은 시각 t** 값 | 미래는 아님. 하지만 ⚠️ **test에서 다른 관측소 행(같은 시각)을 가져와야** 해요 |

⚠️ **규정 확인 필요**: 연결망 변수는 test 안의 **다른 관측소, 같은 시각** 행을 읽어야 만들어져요. 시각 t 이후 정보는 아니지만, "한 행은 그 행의 입력만으로 예측한다"는 해석이면 문제가 될 수 있어요. **대회 규정을 확인한 뒤** 쓰기로 해요. 또 1권에서 본 것처럼 test에는 같은 관측소의 **뒤 시각** 행도 있으니, 다른 관측소 값을 가져올 때 **반드시 같은 시각(또는 과거)만** 가져와야 해요.

### 정리

| 발견 | 전체에 해당? | 모델링 시사점 |
|---|---|---|
| 공통 호우 신호를 빼야 진짜 연결이 보임 | 모든 관측소 쌍 | 연결망 계산 전 공통 신호 제거 |
| 일반 관측소 절반 이상에 뚜렷한 선행 관측소, 간격은 주로 1~2h | 442곳 전부 검사 | 선행 관측소의 같은 시각 변화 |
| 뒤따르는 쪽이 더 큰 유역 약 2/3 | 유역이 다른 선행 쌍 전부 | 상류→하류 구조가 데이터에 있음 |
| ❌ 사상을 반씩 나누면 같은 선행 관측소는 약 1/3 | 두 절반 모두에서 잡힌 곳 전부 | 안정적 관계만 쓰거나 주변 여러 관측소 요약 사용 |
| 연결망 변수로 RMSE 2~3% 감소, 조석은 악화 | 관측소별·사상별로 대부분 개선 | **규정 확인 후** 사용, 조석 관측소는 제외 |
""")
b.write("nb5_station_network.ipynb")

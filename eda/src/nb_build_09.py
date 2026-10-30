"""9권: 도달시간과 강우 창 (선행연구 A1·A2·A3·A4·A5·A6·A9·B3·C5, 가볍게 C7·C8·C9·A16)."""
from nb_common import Book, RESEARCH_HELPERS

b = Book()
b.md("""
# ⏳ 9권 · 도달시간과 강우 창 — "이미 내린 비는 언제 도착하나"

> 시리즈: 7권 구간 → 8권 수위 국면 → **9권 도달시간·강우 창** → 10권 조절 효과·판정
> 근거: 선행연구 md의 A3·A4·A5·A6·A9(도달시간과 강우 창), B3(겹치지 않는 창), C5(초단기), A1(강우 문턱), A2(선행 습윤), 가볍게 C7·C9·C8·A16. 계획서 4장

### 한 줄 요약
**지체시간이 길수록 '가장 잘 맞는 강우 창'이 뒤로 가는 건 맞아요(A6, 관측소 62%). 하지만 6~24시간 전 비(`r6_24`)가 주도하는 곳은 거의 없고, 도달시간은 면적만으로도 Tc_index만큼 설명돼요. 경사 효과(A9)와 선행 습윤(A2)은 이 데이터에서 보이지 않아요. 기준 모델은 이 창 정보들을 이미 거의 다 쓰고 있어요.**

| 장 | 질문 |
|---|---|
| 1 | 가짜 데이터: 소유역·대유역 반응과 '강우 창' |
| 2 | A4·A5·A9 도달시간 지표 vs 실제 지체시간 |
| 3 | A6·A3·B3 강우 창 이동 + 새 후보 '도착 예정 비' |
| 4 | C5 초단기 반응 |
| 5 | A1 강우 문턱 |
| 6 | A2 선행 습윤 |
| 7 | (가볍게) C7·C9 IMERG 강도별, C8 레이더 단위, A16 p95−평균 |
| 8 | 잔차 점검 모음, 누수 점검과 정리 |

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **도달시간·지체시간 L** | 비가 온 뒤 수위가 가장 크게 반응하기까지 걸리는 시간 | L=3이면 3시간 전 비가 지금 수위를 가장 많이 움직임 |
| **Tc_index** | `wsarea^0.46 × mean_slope^−0.385` (Kirpich·Hack 공식 조합). 큰 유역·완만한 경사일수록 큼 | **순위로만** 써요(경사가 하도 경사가 아니라서 시간으로 못 바꿈) |
| **겹치지 않는 창** | `r0_1`=지난 1h, `r1_6`=1~6h 전, `r6_24`=6~24h 전 지상강수 | 누적 6h = r0_1 + r1_6 |
| **도착 예정 비** | 지체 L인 관측소에서, 앞으로 1~6시간 사이에 도착할 과거 비의 합 = k=L−6…L−1시간 전 비 | L=8 → 2~7시간 전 비 |
| **문턱(hinge)** | 강우가 이만큼 넘어야 수위가 반응하기 시작하는 지점 | 초기손실 |
| **선행 습윤** | 이번 비 이전에 이미 온 비(`r6_24`) | 땅이 젖어 있으면 같은 비에도 더 오름 |
""")
b.setup(RESEARCH_HELPERS)
b.code(r'''
tr = load_resid(add_flags(pd.read_parquet(CACHE / "train_sorted.parquet")))
tr["phase"] = phase_of(tr)
reg = pd.read_csv(RESULTS / "nb07_regimes.csv", index_col=0)
tr["regime"] = tr.station_id.map(reg.regime)
key = keyed(tr)
tr["r0_1"], tr["r1_6"], tr["r6_24"], tr["r0_6"] = tr.sfc_rain_1h, tr.sfc_rain_6h - tr.sfc_rain_1h, tr.sfc_rain_24h - tr.sfc_rain_6h, tr.sfc_rain_6h
tr["i0_1"], tr["i1_6"], tr["i6_24"] = tr.imerg_rain_1h, tr.imerg_rain_6h - tr.imerg_rain_1h, tr.imerg_rain_24h - tr.imerg_rain_6h
gen_ids = reg.index[reg.group == "일반"]
print(f"train {len(tr):,}행 · 일반 관측소 {len(gen_ids)}곳")
''')

b.md("""
## 1. 가짜 데이터: 소유역·대유역 반응과 '강우 창'

같은 비를 소유역(빨리·뾰족)과 대유역(늦게·넓게)에 떨어뜨려 봐요. 지금 시각 t에서 **앞으로 6시간 안에 수위를 올릴 비**가 어느 과거 창에 들어 있는지가 유역마다 달라요.
- 소유역: 이미 대부분 도착 → 과거 비로는 앞날을 잘 모름(앞으로 올 비가 중요)
- 대유역: 6~24시간 전 비가 이제 도착 중 → `r6_24`가 중요
""")
b.code(r'''
t = np.arange(48); rain = np.zeros(48); rain[5:9] = [5, 15, 10, 3]
uh = lambda L, w: np.exp(-((np.arange(30) - L) / w) ** 2)
small = np.convolve(rain, uh(1, 1) / uh(1, 1).sum(), "full")[:48]; big = np.convolve(rain, uh(12, 5) / uh(12, 5).sum(), "full")[:48]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.2))
for a, y, ttl in zip(ax, [small, big], ["소유역 L≈1h", "대유역 L≈12h"]):
    a.plot(t, y, lw=2, color=C_NT); a.axvline(10, color="k", ls="--"); a.axvspan(11, 16, color=C_RAIN, alpha=.2)
    a2 = a.twinx(); a2.bar(t, rain, color=C_RAIN, alpha=.5); a2.invert_yaxis(); a2.set_yticks([]); a.set_title(f"{ttl}: 지금(점선) 뒤 6시간(노란 띠)")
plt.tight_layout(); plt.show()
''')

b.md("""
## 2. A4·A5·A9 도달시간 지표 vs 실제 지체시간

> **md A4·A5·A9** · 가설: Tc ∝ wsarea^0.46·slope^−0.385 — 면적이 크면 늦고, 경사가 급하면 빠르다 · 확인: 관측소 × 사상마다 지체 L(0~12h, `sfc_rain_1h(t−L)`과 `dwl_1h(t)` 상관 최대)을 구해 중앙값. 면적만 / Tc_index / 하천차수의 순위상관 비교, 면적 고정 시 경사 부분상관, 사상 짝·홀 반분 재현
> 사상 조건: 그 관측소 행 24개 이상 & 지상 강우 합 5mm 이상. 이런 사상이 3개 미만이면 기존 추정(3권)을 쓰고 표시.
""")
b.code(r'''
tr["se"] = tr.station_id + "|" + tr.event_block.astype(str)
cnt = tr.groupby("se").agg(n=("dwl_1h", "size"), rain=("sfc_rain_1h", "sum"))
okse = cnt.index[(cnt.n >= 24) & (cnt.rain >= 5)]
sub = tr[tr.se.isin(okse)]
skey = keyed(sub)
C = {}
for L in range(13):
    x = lookup_past(sub, skey, "sfc_rain_1h", L)
    C[L] = station_corr(sub, x, sub.dwl_1h.to_numpy(), min_n=12, by="se")
C = pd.DataFrame(C)
Lse = C.idxmax(axis=1).dropna().astype(int).to_frame("L")
Lse["station_id"] = [s.split("|")[0] for s in Lse.index]; Lse["ev"] = [int(s.split("|")[1]) for s in Lse.index]
agg = Lse.groupby("station_id").L.agg(["median", "size"])
old = pd.read_csv(RESULTS / "s04_station_summary.csv", index_col=0)["lag_sfc_rain_1h"]
lag = pd.DataFrame(index=reg.index)
lag["L"] = agg["median"].where(agg["size"] >= 3); lag["how"] = np.where(lag.L.notna(), "사상 중앙값", "기존 추정")
lag["L"] = lag.L.fillna(old)
env = tr.groupby("station_id")[["wsarea", "mean_slope", "stream_order"]].first()
lag = lag.join(env); lag["Tc_index"] = lag.wsarea ** .46 * lag.mean_slope ** -.385
g = lag.loc[gen_ids]
print("추정 방식:", g.how.value_counts().to_dict(), "· L 분포:", g.L.describe()[["25%", "50%", "75%"]].to_dict())
for c in ["wsarea", "Tc_index", "stream_order"]:
    print(f"  L ↔ {c}: 순위상관 {stats.spearmanr(g.L, g[c])[0]:.2f}")
rk = g[["L", "wsarea", "mean_slope"]].rank()
res = lambda a, b_: a - np.polyval(np.polyfit(b_, a, 1), b_)
pc = np.corrcoef(res(rk.L, rk.wsarea), res(rk.mean_slope, rk.wsarea))[0, 1]
print(f"  면적 고정 시 경사의 부분 순위상관 {pc:.2f} (가설: 음수)")
h = Lse[Lse.station_id.isin(gen_ids)].assign(p=lambda d: d.ev % 2).groupby(["station_id", "p"]).L.median().unstack().dropna()
print(f"  사상 짝·홀 반분 재현: 두 절반 L의 순위상관 {stats.spearmanr(h[0], h[1])[0]:.2f} ({len(h)}곳), 차이 ≤1h {((h[0]-h[1]).abs() <= 1).mean():.0%}")
lag.to_csv(RESULTS / "nb09_station_lag.csv")
''')

b.md("""
## 3. A6·A3·B3 강우 창 이동 + 도착 예정 비

> **md A6** · 가설: Tc가 약 2h 미만이면 `r0_1`, 2~8h면 `r1_6`, 8h 초과면 `r6_24`가 주도 · 확인: 비조석·비 오는 행에서 Tc_index 5분위와 실제 L 구간마다 타깃과 세 창의 순위상관, 관측소마다 가장 강한 창이 md 예상과 맞는 비율(판정 기준 0.4)
> **md B3** · 누적 창보다 겹치지 않는 창이 공선성이 작다 · IMERG 창은 **수치만** 나란히 적어요(결정 #6).
""")
b.code(r'''
wr = tr[tr.nontidal & (tr.sfc_rain_24h > 0)].copy()
wr["Lst"] = wr.station_id.map(lag.L); wr["tcq"] = wr.station_id.map(pd.qcut(lag.Tc_index.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]))
wr["Lbin"] = pd.cut(wr.Lst, [-1, 1, 3, 6, 13], labels=["0~1h", "2~3h", "4~6h", "7~12h"])
W = ["r0_1", "r1_6", "r6_24"]; I = ["i0_1", "i1_6", "i6_24"]
def rho_tab(by):
    return wr.groupby(by, observed=True).apply(lambda d: pd.Series({w: stats.spearmanr(d[w], d[TARGET])[0] for w in W + I} | {"행": len(d)}))
display(rho_tab("tcq").round(3)); display(rho_tab("Lbin").round(3))
best = {}
for s, d in wr[wr.station_id.isin(gen_ids)].groupby("station_id"):
    if len(d) >= 100: best[s] = W[int(np.nanargmax([stats.spearmanr(d[w], d[TARGET])[0] for w in W]))]
best = pd.Series(best); Ls = lag.L.loc[best.index]
expect = np.where(Ls < 2, "r0_1", np.where(Ls <= 8, "r1_6", "r6_24"))
share = (best == expect).mean()
print(pd.crosstab(pd.cut(Ls, [-1, 1, 3, 6, 13]), best))
print(f"가장 강한 창이 md 예상(L<2→r0_1, 2~8→r1_6, >8→r6_24)과 맞는 관측소 {share:.0%} ({len(best)}곳) → 판정 {verdict(1, 1, share)}")
def vif(cols):
    cm = np.corrcoef(np.log1p(wr[cols].clip(lower=0).to_numpy()).T); return pd.Series(np.diag(np.linalg.inv(cm)), index=cols).round(2)
print("VIF 누적 창:", vif(["sfc_rain_1h", "sfc_rain_6h", "sfc_rain_24h"]).to_dict(), "· 겹치지 않는 창:", vif(W).to_dict())
''')
b.code(r'''
# 도착 예정 비: 같은 관측소 이전 행의 1h 강우를 k=max(0,L−6)…L−1 만큼 더함 (과거만)
Lr = tr.station_id.map(lag.L).fillna(0).round().astype(int).to_numpy()   # 지체를 못 구한 관측소(고정 등)는 0
R = np.column_stack([lookup_past(tr, key, "sfc_rain_1h", k) for k in range(12)])
arr = np.zeros(len(tr))
for k in range(12):
    inwin = (k >= np.maximum(0, Lr - 6)) & (k <= Lr - 1)
    arr = arr + np.where(inwin, R[:, k], 0.0)
arr[np.isnan(R[:, :12]).any(1) & (Lr > 0)] = np.nan
tr["arrive"] = arr
m = tr.nontidal & (tr.sfc_rain_24h > 0)
print(f"도착 예정 비 ↔ 타깃 순위상관(비조석 비 오는 행) {stats.spearmanr(tr.arrive[m], tr[TARGET][m], nan_policy='omit')[0]:.3f} · r0_6은 {stats.spearmanr(tr.r0_6[m], tr[TARGET][m])[0]:.3f}")
''')

b.md("""
## 4. C5 초단기 반응

> **md C5** · 가설: 도달시간이 매우 짧은 소유역은 앞으로 올 비가 타깃을 좌우해서 과거 입력으로 설명 못 하는 오차가 크다 · 확인: 사상 단위 L = 0h인 일반 관측소(7권 플래그보다 좁힘)의 오차, 타깃과 레이더·`r0_1` vs `r6_24` 상관의 관측소별 비교
""")
b.code(r'''
fast = lag.index[(lag.L == 0) & lag.index.isin(gen_ids) & (lag.how == "사상 중앙값")]
tr["fast0"] = tr.station_id.isin(fast)
G = tr[tr.group == "일반"]
print(f"사상 단위 L=0 관측소 {len(fast)}곳 (7권 초단기 의심 {len(reg.index[reg.regime=='초단기 의심'])}곳과 겹침 {len(set(fast) & set(reg.index[reg.regime=='초단기 의심']))}곳)")
print(G.groupby("fast0").agg(행=("resid", "size"), RMSE=("resid", lambda v: np.sqrt((v**2).mean())), 타깃_표준편차=(TARGET, "std")).round(4))
Gf = G[G.fast0 & (G.sfc_rain_24h > 0)]
rr = group_spearman(Gf, Gf.radar_reflectivity_mean, Gf[TARGET]); r1 = group_spearman(Gf, Gf.r0_1, Gf[TARGET]); r24 = group_spearman(Gf, Gf.r6_24, Gf[TARGET])
sh_c5 = ((np.maximum(rr, r1) > r24).dropna()).mean()
print(f"L=0 관측소 중 (레이더 또는 r0_1 상관) > r6_24 상관인 곳 {sh_c5:.0%} → 판정 {verdict(1, 1, sh_c5)}")
sse_all = (G.resid ** 2).sum(); ph_sh = G[G.fast0].groupby("phase").resid.apply(lambda v: (v**2).sum() / sse_all)
print("L=0 관측소의 국면별 오차 몫(일반 전체 대비):", ph_sh.round(3).to_dict())
''')

b.md("""
## 5. A1 강우 문턱

> **md A1** · 가설: 강우가 초기손실을 넘은 뒤에야 반응(문턱 + 볼록), 도시·습윤이면 문턱이 낮다 · 확인: log1p(`r0_6`) 칸별 평균 타깃에 두 직선을 맞춰 꺾이는 점 추정. 도시화·산림·선행 습윤 3분위별 이동, 사상 짝·홀 반분 재현
""")
b.code(r'''
nr = tr[tr.nontidal & (tr.r0_6 > 0)].copy()
def hinge(d, q=30):
    x = np.log1p(d.r0_6); k = pd.qcut(x.rank(method="first"), q, labels=False)
    bx, by = x.groupby(k).mean().to_numpy(), d[TARGET].groupby(k).mean().to_numpy()
    best = (np.inf, None)
    for c in bx[3:-3]:
        X = np.column_stack([np.ones_like(bx), bx, np.maximum(bx - c, 0)]); r = by - X @ np.linalg.lstsq(X, by, rcond=None)[0]
        best = min(best, ((r ** 2).sum(), c), key=lambda z: z[0])
    return np.expm1(best[1])
out = {"전체": hinge(nr), "짝 사상": hinge(nr[nr.event_block % 2 == 0]), "홀 사상": hinge(nr[nr.event_block % 2 == 1])}
for c, lab in [("urban_ratio", "도시화"), ("forest_ratio", "산림")]:
    tq = pd.qcut(nr.station_id.map(tr.groupby("station_id")[c].first()).rank(method="dense"), 3, labels=["하", "중", "상"])
    for q in ["하", "상"]: out[f"{lab} {q}"] = hinge(nr[tq == q])
aq = pd.qcut(nr.r6_24.rank(method="first"), 3, labels=["하", "중", "상"])
for q in ["하", "상"]: out[f"선행 습윤 {q}"] = hinge(nr[aq == q])
print("꺾이는 점(6h 강우, mm):", {k: round(v, 1) for k, v in out.items()})
''')

b.md("""
## 6. A2 선행 습윤

> **md A2** · 가설: 선행 강우가 많을수록(땅이 젖어) 같은 비에 더 오른다 · 확인: `r0_6` 칸 × 사상 경과 시간 칸 안에서 타깃을 중심화(칸 평균을 뺌)한 뒤 `r6_24`와의 순위상관 — 관측소·사상별 부호. 사상 경과 시간을 통제하는 이유는 4권에서 "긴 사상이 만든 차이"를 봤기 때문이에요. 24h 창까지만 있어서 md의 5일 선행강우는 볼 수 없어요.
""")
b.code(r'''
wa = tr[tr.nontidal & (tr.r0_6 >= 1)].copy()
wa["cb"] = pd.cut(wa.r0_6, [1, 5, 15, 40, 1e4]).astype(str) + "|" + pd.cut(wa.pos_in_block, [-1, 5, 11, 23, 1e4]).astype(str)
wa["y_c"] = wa[TARGET] - wa.groupby("cb")[TARGET].transform("mean")
ov = np.sign(stats.spearmanr(wa.r6_24, wa.y_c)[0])
rs = group_spearman(wa, wa.r6_24, wa.y_c, "station_id", 50); re_ = group_spearman(wa, wa.r6_24, wa.y_c, "event_block", 200)
(ss, ns), (se, ne) = share_same_sign(rs[rs.index.isin(gen_ids)], 1), share_same_sign(re_, 1)
print(f"전체 부호 {ov:+.0f} (ρ {stats.spearmanr(wa.r6_24, wa.y_c)[0]:.3f}) · 일반 관측소 {ss:.0%} ({ns}곳) · 사상 {se:.0%} ({ne}개) → 판정 {verdict(ov, 1, min(ss, se))}")
big = wa[wa.r0_6 >= 5]; aq = pd.qcut(big.r6_24.rank(method="first"), 3, labels=["하", "중", "상"])
print("r0_6 ≥ 5mm에서 선행 습윤 3분위별 타깃÷강우 중앙값:", (big[TARGET] / big.r0_6).groupby(aq, observed=True).median().round(4).to_dict())
''')

b.md("""
## 7. (가볍게) C7·C9 IMERG 강도별 · C8 레이더 단위 · A16 p95−평균

- **C7·C9**: 24h 창에서 IMERG ÷ 지상 비율의 중앙값을 지상 강도별로. 오차 상위 5개 사상도 따로. (수치만, 시각 해석 없음)
- **C8**: 레이더 값의 범위·음수 여부·모양. "dB 척도라고 가정하면" Marshall–Palmer로 환산한 강우와 지상 1h 크기 비. 순위상관은 변환해도 그대로라 트리 모델엔 영향이 없어요.
- **A16**: p95 − 평균(국지 대류 지표)과 타깃·잔차
""")
b.code(r'''
c7 = tr[tr.sfc_rain_24h >= 1]
kb = pd.cut(c7.sfc_rain_24h, [1, 5, 20, 50, 100, 1e4])
ratio = (c7.imerg_rain_24h / c7.sfc_rain_24h).groupby(kb, observed=True).median()
print("C7 IMERG÷지상 (24h) 중앙값, 지상 강도별:", ratio.round(2).to_dict())
eq = pd.qcut(c7.station_id.map(tr.groupby("station_id").mean_elevation.first()).rank(method="dense"), 3, labels=["저", "중", "고"])
print("   표고 3분위별:", (c7.imerg_rain_24h / c7.sfc_rain_24h).groupby(eq, observed=True).median().round(2).to_dict())
top5 = tr.groupby("event_block").resid.apply(lambda v: (v**2).sum()).nlargest(5).index
print("C9 오차 상위 5사상 비율 중앙값 {:.2f} vs 나머지 {:.2f}".format((c7.imerg_rain_24h / c7.sfc_rain_24h)[c7.event_block.isin(top5)].median(), (c7.imerg_rain_24h / c7.sfc_rain_24h)[~c7.event_block.isin(top5)].median()))
z = tr.radar_reflectivity_mean
print(f"C8 레이더 평균: 최소 {z.min():.1f} · 최대 {z.max():.1f} · 음수 {(z < 0).mean():.1%} · 왜도 {stats.skew(z):.2f}")
both = tr[(tr.sfc_rain_1h > 0) & (z > 0)]
R_mp = (10 ** (both.radar_reflectivity_mean / 10) / 200) ** (1 / 1.6)
print(f"   dB 가정 환산 강우 ÷ 지상 1h 중앙값 {np.median(R_mp / both.sfc_rain_1h):.2f} · 환산값 Pearson {np.corrcoef(R_mp, both.sfc_rain_1h)[0,1]:.2f} vs 원값 {np.corrcoef(both.radar_reflectivity_mean, both.sfc_rain_1h)[0,1]:.2f}")
tr["p95m"] = tr.radar_reflectivity_p95 - tr.radar_reflectivity_mean
print(f"A16 p95−평균 ↔ 타깃 순위상관(비조석) {stats.spearmanr(tr.p95m[tr.nontidal], tr[TARGET][tr.nontidal])[0]:.3f}")
''')

b.md("""
## 8. 잔차 점검 모음, 누수 점검과 정리
""")
b.code(r'''
rows = [resid_check("r0_1", tr.r0_1, tr, tr.phase == "비·아직 반응 없음"),
        resid_check("r1_6", tr.r1_6, tr, tr.phase == "비·아직 반응 없음"),
        resid_check("r6_24", tr.r6_24, tr, tr.phase == "비·아직 반응 없음"),
        resid_check("도착 예정 비 (비·아직 반응 없음)", tr.arrive, tr, tr.phase == "비·아직 반응 없음"),
        resid_check("도착 예정 비 (비·상승 중)", tr.arrive, tr, tr.phase == "비·상승 중"),
        resid_check("지체 L (관측소 특성)", tr.station_id.map(lag.L), tr, tr.phase == "비·아직 반응 없음"),
        resid_check("Tc_index", tr.station_id.map(lag.Tc_index), tr, tr.phase == "비·아직 반응 없음"),
        resid_check("선행 습윤 r6_24 (비·상승 중)", tr.r6_24, tr, tr.phase == "비·상승 중"),
        resid_check("p95−평균", tr.p95m, tr, tr.phase.str.startswith("비"))]
RC = pd.DataFrame(rows); display(RC)
import json
json.dump({"lag_share_expected_window": float(share), "c5_share": float(sh_c5), "a2_station": float(ss), "a2_event": float(se),
           "hinge": {k: float(v) for k, v in out.items()}, "imerg_ratio": {str(k): float(v) for k, v in ratio.items()}, "resid": RC.to_dict("records")},
          open(RESULTS / "nb09_summary.json", "w"), ensure_ascii=False, indent=1, default=float)
''')
b.md("""
### 누수 점검표

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 2장 지체 L | 과거 강우(t−L) ↔ 현재 수위 변화, 사상 안 전체 | 관측소 특성 **B등급**(train에서 계산) |
| 3장 창·도착 예정 비 | 이전 행의 1h 강우만 | **A등급**(L은 B등급) |
| 3·4장 창별 상관, 5·6장 | 입력 ↔ 타깃 | 설명용 |
| 7장 IMERG 비율 | 같은 행 24h 누적끼리 | 수치만, 시각 해석 없음 |
| 8장 잔차 | 사상 단위 OOF | 누수 없음 |

### 정리

| md | 발견 | 판정 | 모델링 시사점 |
|---|---|---|---|
| A4·A5 도달시간 | 사상 단위 지체 L 중앙값 3h(442곳 모두 사상 중앙값으로 추정). L ↔ 면적 ρ 0.55, Tc_index 0.51, 하천차수 0.45. 짝·홀 반분 재현 ρ 0.55(차이 1h 이내 59%) | ✅ 채택(큰 유역일수록 늦음). 단 Tc_index는 면적보다 나을 게 없음 | 도달시간 지표는 **면적(log)** 으로 충분. L은 관측소 특성(B등급) 후보 |
| A9 경사 | 면적 고정 시 경사의 부분 순위상관 **+0.13**(가설은 음수) | ❌ 기각 | `mean_slope`로 도달시간을 보정하지 않음 |
| A6·A3 강우 창 이동 | 가장 강한 창: L 0~1h `r0_1`(0.44) → 4~6h `r1_6`(0.39) → 7~12h `r1_6`(0.31, `r6_24` 0.17로 상승). 관측소별 md 예상과 일치 **62%** | 🔶 부분(방향은 맞고, `r6_24` 주도 구간은 거의 없음) | 지체 × 창 상호작용은 유효하나 트리가 이미 잡음(창별 잔차 몫 −0.2~−0.4%) |
| B3 겹치지 않는 창 | VIF 누적 최대 2.79 → 겹치지 않는 창 최대 1.42 | ✅ 채택 | 선형·신경망에는 겹치지 않는 창. 트리는 선택 |
| 새 후보 도착 예정 비 | 타깃 순위상관 0.395(r0_6 0.383). 잔차: 비·아직 반응 없음 −1.5%, 비·상승 중 상위 5사상 제외 +1.6%지만 사상 일관성 44% | 약한 후보 | 모델링에서 시험은 하되 우선순위 낮음 |
| C5 초단기 | 사상 단위 L=0 관측소 33곳(7권 플래그 108곳보다 훨씬 좁음, 겹침 18). 이곳들은 100% 레이더·r0_1 상관 > r6_24 상관, RMSE 0.246(나머지 0.218) | ✅ 채택(방향). 오차 몫은 일반 전체의 약 9% | 7권 '초단기 의심' 플래그를 **사상 단위 L=0 33곳**으로 교체 권고 |
| A1 강우 문턱 | 6h 강우 기준 꺾이는 점 **약 15mm**(짝 11 / 홀 18mm). 도시화·산림·선행 습윤 3분위별 13~16mm로 이동 없음 | 🔶 부분(문턱은 있음, 토지이용·습윤별 이동은 ❌) | 문턱 이동 변수 불필요 |
| A2 선행 습윤 | 같은 현재 강우·같은 사상 경과 시간 안에서 `r6_24`가 많을수록 타깃이 **작음**(ρ −0.15), 일반 관측소 24%·사상 25%만 양(+) | ❌ 기각(24h 창 기준) | 6~24h 전 비가 많다 = 이미 첨두를 지났을 가능성. + 단조 제약 금지 |
| C7 IMERG 강도별 | IMERG÷지상 24h: 지상 1~5mm 2.8 → 100mm 초과 **0.62**. 표고 3분위별 1.27~1.32로 차이 없음 | ✅ 채택(강할수록 IMERG가 상대적으로 작음). 표고 효과 없음 | 강한 비에서는 지상강수를 주 변수로 |
| C9 극한 사상 | 오차 상위 5사상 비율 1.18 vs 나머지 1.35 | 🔶 부분(수치 하나) | – |
| C8 레이더 단위 | 음수 없음, 0~55.9, 거의 대칭(왜도 0.38). dB 가정 환산 강우 ÷ 지상 1h 중앙값 0.68, Pearson 0.36 → 0.45로 상승 | 🔶 "0에서 잘린 dB류 척도"와 일치(공식 단위 아님) | 트리 모델엔 변환 불필요. 선형 모델이면 환산값 |
| A16 p95−평균 | 타깃 ρ 0.05, 잔차 −0.35% | ❌ 기각 | – |

- 잔차 점검의 설명 가능 몫이 대부분 음수 → 기준 모델(HGB)이 이 강우·도달시간 정보를 이미 씀. 상위 5사상을 빼면 +0.0~1.9%로 조금 양수라, 큰 사상 몇 개가 칸 평균을 흔드는 효과가 커요.

### 3단계 점검
- 도달시간 지표: 면적(log) 권장, 실제 L은 관측소 특성 후보 / Tc_index 불채택
- 도착 예정 비: 약한 후보로 등록
- C5 플래그를 사상 단위 L=0 33곳으로 교체
""")
b.write("nb9_travel_time_and_rain_windows.ipynb")

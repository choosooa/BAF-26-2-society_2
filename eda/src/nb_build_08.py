"""8권: 수위 국면과 타깃 (선행연구 T1·T2·T3·A12)."""
from nb_common import Book, RESEARCH_HELPERS

b = Book()
b.md("""
# 📈 8권 · 수위 국면과 타깃 — "지금 어느 단계인가"가 얼마나 알려 주나

> 시리즈: 7권 구간 나누기 → **8권 수위 국면** → 9권 도달시간·강우 창 → 10권 조절 효과·판정
> 근거: 선행연구 md의 T1(하강·상승 가지)·T2(가속도)·T3(사상 단계)·A12(감수곡선), 계획서 3장

### 한 줄 요약
**"떨어지는 중이면 타깃도 음수"(T1)는 중앙값 기준으로 맞아요. 평균이 +로 보였던 건 비 오는 중 재상승과 댐 의심 관측소 때문이에요. "사상 초기냐 첨두 후냐"(T3)는 사상 99%에서 성립해요. 반면 "가속 중이면 더 오른다"(T2)와 감수 법칙(A12)은 이 데이터에서 성립하지 않아요. 그리고 네 가지 모두 기준 모델이 이미 거의 다 쓰고 있어요.**

| 장 | 질문 |
|---|---|
| 1 | 가짜 수문곡선으로 본 국면과 타깃 |
| 2 | T1 하강 가지 — 떨어지는 중이면 타깃도 음수인가 (기존 결과와 부딪히는 곳) |
| 3 | T2 가속도 — 가속 중이면 더 오르나 |
| 4 | T3 사상 단계 — 24h 변화 × 1h 변화 |
| 5 | A12 감수 — 빠지는 속도에 규칙이 있나 |
| 6 | 잔차 점검 모음 |
| 7 | 누수 점검과 정리 |

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **상승부·첨두·감수부** | 수문곡선이 오르는 구간 · 가장 높은 순간 · 빠지는 구간 | |
| **가속도** `acc` | `dwl_1h − (wl_t−1 − wl_t−3)/2` = 최근 1시간 속도 − 그 전 2시간 평균 속도 | 앞 2시간 시간당 2cm, 지금 10cm → acc = +8cm, 가속 중 |
| **하강 가지 기울기 β** | 떨어지는 중인 행에서 "타깃 ÷ 1시간 변화" | β=0.8이면 지금 10cm 빠지면 타깃 −8cm |
| **감쇠비** | `dwl_1h / dwl_3h` (빠지는 중일 때) | 1/3이면 일정 속도, 작을수록 빠지는 속도가 줄어드는 중 |
| **감수 지수 b** | 수위가 높을수록 빨리 빠지는 정도(log 기울기) | |
| **남은 오차(잔차)** | 타깃 − 기준 모델(6권 HGB) 예측 | |
""")
b.setup(RESEARCH_HELPERS)
b.code(r'''
tr = load_resid(add_flags(pd.read_parquet(CACHE / "train_sorted.parquet")))
tr["phase"] = phase_of(tr)
reg = pd.read_csv(RESULTS / "nb07_regimes.csv", index_col=0)
tr["regime"] = tr.station_id.map(reg.regime)
tr["acc"] = tr.dwl_1h - (tr.dwl_3h - tr.dwl_1h) / 2          # 과거 값만
tr["rain6"] = np.maximum(tr.sfc_rain_6h, tr.imerg_rain_6h)
nt = tr[tr.nontidal].copy()
print(f"train {len(tr):,}행 · 비조석 {len(nt):,}행")
''')

b.md("""
## 1. 가짜 수문곡선으로 본 국면과 타깃

하나의 수문곡선에서 시각마다 타깃(앞으로 6시간 최고 − 지금)을 계산해 봐요.
- **상승부**: 타깃이 커요. 특히 **가속 중**(초반)일수록.
- **첨두 직전**: 오르고는 있지만 곧 멈춰서 타깃이 작아져요(감속 중).
- **감수부**: 계속 빠지면 앞으로 6시간 최고는 **바로 다음 시각** 값이에요. 그래서 타깃 ≈ 다음 1시간 하강폭(음수). md T1이 말하는 "하강 가지"예요.
- 타깃의 하한은 `WL(t+1) − WL(t)`예요(정의상). 이건 미래 값이라 **그림 설명에만** 써요.
""")
b.code(r'''
t = np.arange(60); h = 3 * (t / 15) ** 2 * np.exp(-(t / 15 - 1) * 2)      # 가짜 수문곡선
tg = np.array([h[i + 1:i + 7].max() - h[i] if i + 6 < len(h) else np.nan for i in range(len(h))])
d1 = np.r_[np.nan, np.diff(h)]; acc = d1 - np.r_[[np.nan] * 3, (h[2:-1] - h[:-3]) / 2]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
ax[0].plot(t, h, color=C_NT, lw=2); ax[0].set_title("가짜 수문곡선"); ax[0].set_xlabel("시간 (h)"); ax[0].set_ylabel("수위 (m)")
ax[1].plot(t, tg, color=C_TIDAL, lw=2); ax[1].axhline(0, color="k", lw=.8); ax[1].set_title("타깃: 상승 초반 최대 → 감수부 음수"); ax[1].set_xlabel("시간 (h)")
ax[2].scatter(d1, tg, c=t, cmap="viridis", s=18); ax[2].set_xlabel("1시간 변화 dwl_1h"); ax[2].set_ylabel("타깃"); ax[2].set_title("같은 속도라도 단계(색=시간)에 따라 타깃이 다름")
plt.tight_layout(); plt.show()
''')

b.md("""
## 2. T1 하강 가지 — 떨어지는 중이면 타깃도 음수인가

> **md T1** · 가설: 하강 중(dWL1 < 0)이면 y ≈ β·dWL1, β 0.7~1 (음수), 상승 중이면 기울기가 더 크다 · 기존 결과와 부딪힘: 비조석 `dwl_1h` 최하위 칸 평균 +0.016m (별책 6-1)
> 확인: 평균과 **중앙값**을 함께, 이유 후보 3개(① 1cm 계단 ② 비 오는 중 하강 뒤 재상승 ③ 조작)를 하나씩 · 관측소별 β가 0.7~1인 비율 · 판정: 관측소 70% 이상에서 β > 0이면 방향 채택

**예시**: 1시간에 10cm 빠진 관측소가 계속 빠지면 타깃은 약 −8cm. 그런데 비가 계속 오는 중이라 2시간 뒤 다시 30cm 오르면 타깃은 +20cm가 돼요. 평균은 이런 재상승에 끌려 올라갈 수 있어요.
""")
b.code(r'''
fall = nt[nt.dwl_1h < -.01]
def summ(s, lab):
    y = s[TARGET]
    return {"무리": lab, "행": len(s), "평균": y.mean(), "중앙값": y.median(), "음수(<−1cm)": (y < -.01).mean(),
            "0(±1cm)": (y.abs() <= .01).mean(), "양수(>1cm)": (y > .01).mean()}
rows = [summ(fall, "하강 중 전체"), summ(fall[fall.dwl_1h <= -.10], "① 큰 하강(≥10cm)만"),
        summ(fall[fall.rain6 >= .5], "② 과거 6h 비 있음"), summ(fall[fall.rain6 < .5], "② 과거 6h 비 없음")]
for r_ in ["일반(나머지)", "초단기 의심", "댐 방류 의심", "보 관리수위 의심"]:
    rows.append(summ(fall[fall.regime == r_], f"③ {r_}"))
display(pd.DataFrame(rows).set_index("무리").round(3))
''')
b.code(r'''
# 관측소별 기울기 β: 하강 중(|dwl_1h| ≥ 2cm) 행의 타깃 ÷ dwl_1h 중앙값 (= 원점 지나는 중앙값 기울기의 근사)
f2 = fall[fall.dwl_1h <= -.02]
beta = (f2[TARGET] / f2.dwl_1h).groupby(f2.station_id).agg(["median", "size"])
beta = beta[beta["size"] >= 50]["median"]
gen_ids = reg.index[reg.group == "일반"]
bg = beta[beta.index.isin(gen_ids)]
rise = nt[nt.dwl_1h > .01]; r2 = rise[rise.dwl_1h >= .02]
beta_up = (r2[TARGET] / r2.dwl_1h).groupby(r2.station_id).agg(["median", "size"]); beta_up = beta_up[beta_up["size"] >= 50]["median"]
print(f"하강 가지 β(일반 관측소 {len(bg)}곳): 중앙값 {bg.median():.2f} · β>0 {(bg > 0).mean():.0%} · md 예상 0.7~1 안 {bg.between(.7, 1).mean():.0%}")
print(f"상승 가지 β(같은 방식): 중앙값 {beta_up[beta_up.index.isin(gen_ids)].median():.2f}")
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].hist(bg.clip(-2, 3), bins=60, color=C_NT); ax[0].axvspan(.7, 1, color=C_RAIN, alpha=.3, label="md 예상 0.7~1")
ax[0].set_xlabel("하강 가지 β (관측소별)"); ax[0].set_ylabel("관측소 수"); ax[0].legend(); ax[0].set_title("일반 관측소 전부의 하강 가지 기울기")
bins_ = np.r_[-1, -.3, -.1, -.05, -.02, -.01, .01, .02, .05, .1, .3, 1]
k = pd.cut(nt.dwl_1h, bins_)
agg = nt.groupby(k, observed=True)[TARGET].agg(["mean", "median"])
x = [iv.mid for iv in agg.index]
ax[1].plot(x, agg["mean"], "o-", color=C_NT, label="평균"); ax[1].plot(x, agg["median"], "s--", color=C_TIDAL, label="중앙값")
ax[1].axhline(0, color="k", lw=.8); symlog_x(ax[1], .02); ax[1].legend(); ax[1].set_xlabel("dwl_1h 칸"); ax[1].set_ylabel("타깃"); ax[1].set_title("비조석: 평균과 중앙값")
plt.tight_layout(); plt.show()
''')

b.md("""
## 3. T2 가속도 — 가속 중이면 더 오르나

> **md T2** · 가설: 상승이 가속 중이면 첨두 전이라 y가 크다 · 확인: **1시간 변화가 같은 칸 안에서** 가속도 3분위별 타깃 비교(1시간 변화와 겹치는 효과 제거) · 관측소·사상별 부호 · 판정 기준 0.4
""")
b.code(r'''
up = nt[nt.dwl_1h > .01].copy()
up["d1bin"] = pd.cut(up.dwl_1h, [.01, .03, .1, .3, 5])
up["acc3"] = up.groupby("d1bin", observed=True).acc.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["감속", "중간", "가속"]))
display(up.pivot_table(index="d1bin", columns="acc3", values=TARGET, aggfunc="mean", observed=True).round(3))
# 1시간 변화 칸 평균을 뺀 타깃으로 관측소·사상별 순위상관(= dwl_1h를 통제한 가속도 효과)
up["y_c"] = up[TARGET] - up.groupby("d1bin", observed=True)[TARGET].transform("mean")
rs = group_spearman(up, up.acc, up.y_c, "station_id", 50); re_ = group_spearman(up, up.acc, up.y_c, "event_block", 200)
ov = np.sign(stats.spearmanr(up.acc, up.y_c)[0])
(ss, ns), (se, ne) = share_same_sign(rs[rs.index.isin(gen_ids)], 1), share_same_sign(re_, 1)
print(f"전체 부호 {ov:+.0f} · 일반 관측소 {ss:.0%} ({ns}곳) · 사상 {se:.0%} ({ne}개) → 판정 {verdict(ov, 1, min(ss, se))}")
''')

b.md("""
## 4. T3 사상 단계 — 24h 변화 × 1h 변화

> **md T3** · 가설: 24h 상승폭은 큰데 지금 하강 중이면 첨두 이후라 y가 작고, 24h 상승폭이 작은데 지금 상승 중이면 사상 초기라 y가 크다 · 확인: 격자 평균, 사상별로 "초기 칸 > 첨두 후 칸"인 비율 · 판정 기준 0.4
""")
b.code(r'''
nt["b24"] = pd.qcut(nt.dwl_24h.rank(method="first"), 8, labels=False)
nt["b1"] = pd.cut(nt.dwl_1h, [-99, -.10, -.01, .01, .10, .30, 99], labels=["≤−10", "−10~−1", "±1", "+1~+10", "+10~+30", ">+30"])
hm = nt.pivot_table(index="b24", columns="b1", values=TARGET, aggfunc="mean", observed=True)
q24 = nt.groupby("b24").dwl_24h.median().round(2)
fig, ax = plt.subplots(figsize=(8, 4.2))
im = ax.imshow(hm.to_numpy(), cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto", origin="lower")
ax.set_xticks(range(hm.shape[1]), hm.columns); ax.set_yticks(range(8), [f"{v:+.2f}" for v in q24])
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]): ax.text(j, i, f"{hm.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_xlabel("1시간 변화 (cm)"); ax.set_ylabel("24시간 변화 (칸 중앙값, m)"); ax.set_title("비조석 평균 타깃: 사상 단계 지도"); plt.colorbar(im); plt.show()
early = (nt.b24 <= 2) & (nt.dwl_1h > .01); post = (nt.b24 >= 6) & (nt.dwl_1h < -.01)
ev = pd.DataFrame({"early": nt[early].groupby("event_block")[TARGET].mean(), "post": nt[post].groupby("event_block")[TARGET].mean(),
                   "n_e": nt[early].groupby("event_block").size(), "n_p": nt[post].groupby("event_block").size()}).dropna()
ev = ev[(ev.n_e >= 30) & (ev.n_p >= 30)]
sh = (ev.early > ev.post).mean()
print(f"초기 칸 평균 {nt[early][TARGET].mean():.3f} vs 첨두 후 칸 {nt[post][TARGET].mean():.3f} · 사상별 초기>첨두후 {sh:.0%} ({len(ev)}개 사상) → 판정 {verdict(1, 1, sh)}")
nt["cell"] = nt.b24.astype(int) * 10 + nt.b1.cat.codes
''')

b.md("""
## 5. A12 감수 — 빠지는 속도에 규칙이 있나

> **md A12** · 가설: 감수기 하강률은 (수위 − 기저수위)의 거듭제곱(−dQ/dt = aQ^b)을 따른다 · 확인: 무강우(7권 정의) 하강 행에서 관측소별 b를 Theil–Sen으로 추정, b > 0(높을수록 빨리 빠짐)인 비율 · 감쇠비가 "비·하강 중"에서도 잔차를 설명하는지
> ⚠️ 원래 법칙은 유량에 대한 것이라 수위로 쓴 b는 비교용 지표예요. 기저수위 h0는 관측소 train 수위 1% 분위(B등급).
""")
b.code(r'''
dry = (tr.sfc_rain_24h < .5) & (tr.imerg_rain_24h < 1.0)
h0 = tr.groupby("station_id").wl_t.quantile(.01)
rec = tr[dry & (tr.dwl_1h < -.005) & tr.station_id.isin(gen_ids)].copy()
rec["hh"] = rec.wl_t - rec.station_id.map(h0)
rec = rec[rec.hh > .02]
bs = {}
for s, g_ in rec.groupby("station_id"):
    if len(g_) >= 50 and g_.hh.nunique() > 5:
        bs[s] = stats.theilslopes(np.log(-g_.dwl_1h), np.log(g_.hh))[0]
bs = pd.Series(bs)
print(f"감수 지수 b: {len(bs)}곳 · 중앙값 {bs.median():.2f} · b>0 {(bs > 0).mean():.0%} → 판정 {verdict(np.sign(bs.median()), 1, (bs > 0).mean())}")
tr["decay"] = np.where((tr.dwl_1h < -.01) & (tr.dwl_3h <= -.02), tr.dwl_1h / tr.dwl_3h, np.nan)
for ph in ["무강우·하강 중", "비·하강 중"]:
    s = tr[tr.phase == ph]
    print(ph, "타깃 ~ 감쇠비 순위상관", round(stats.spearmanr(s.decay, s[TARGET], nan_policy="omit")[0], 3))
''')

b.md("""
## 6. 잔차 점검 모음

후보 변수가 기준 모델이 **아직 못 쓴 정보**를 담고 있는지 봐요(계획서 0.5). **설명 가능 몫**은 사상 짝·홀 반분으로 잰 값이라, 음수면 신호가 없다는 뜻이에요.
""")
b.code(r'''
def cat_gain(code, e, ev):
    """범주형(격자 칸) 설명 가능 몫(%): 사상 반분."""
    code, e, ev = np.asarray(code), np.asarray(e, float), np.asarray(ev); out = []
    for a in (0, 1):
        A, B = (ev % 2) == a, (ev % 2) != a
        m = pd.Series(e[A]).groupby(code[A]).mean()
        out.append(1 - ((e[B] - pd.Series(code[B]).map(m).fillna(0).to_numpy()) ** 2).sum() / (e[B] ** 2).sum())
    return 100 * np.mean(out)
T = tr.merge(nt[["station_code", "hour", "cell"]], on=["station_code", "hour"], how="left")
rows = [resid_check("가속도 acc", T.acc, T, T.phase == "비·상승 중"),
        resid_check("가속도 acc (1h 30cm↑ 급상승만)", T.acc, T, (T.phase == "비·상승 중") & (T.dwl_1h >= .3)),
        resid_check("감쇠비 (비·하강 중)", T.decay, T, T.phase == "비·하강 중"),
        resid_check("감쇠비 (무강우·하강 중)", T.decay, T, T.phase == "무강우·하강 중")]
m = T.cell.notna().to_numpy()
rows.append({"후보": "사상 단계 격자 칸(비조석 전체)", "행": int(m.sum()), "설명 가능 몫 %": round(cat_gain(T.cell[m].astype(int), T.resid[m], T.event_block[m]), 2)})
RC = pd.DataFrame(rows); display(RC)
bias = T[m & (T.dwl_1h >= .3)].resid.mean()
print(f"1h 30cm 이상 급상승 행의 평균 잔차 {bias:+.3f} m (6권에서 본 과소예측)")
import json
json.dump({"beta_down_median": float(bg.median()), "beta_down_pos": float((bg > 0).mean()), "beta_in_07_1": float(bg.between(.7, 1).mean()),
           "acc_station_share": float(ss), "acc_event_share": float(se), "stage_event_share": float(sh), "recession_b_median": float(bs.median()),
           "recession_b_pos": float((bs > 0).mean()), "resid": RC.to_dict("records")}, open(RESULTS / "nb08_summary.json", "w"), ensure_ascii=False, indent=1, default=float)
''')
b.md("""
## 7. 누수 점검과 정리

### 누수 점검표

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 1장 타깃 하한 | 미래 값(가짜 데이터) | 설명용. 실제 데이터에선 계산 안 함 |
| 2~4장 dwl·가속도·단계 격자 | 과거 수위 차분만 | **A등급**(dwl_24h 칸 경계는 train 분위 → B등급) |
| 2장 β, 5장 b | 타깃·수위로 관측소별 계산 | 분석용. β는 타깃 사용 → 모델 변수면 C등급 |
| 5장 h0(수위 1% 분위), 무강우 | train 수위, 과거 24h 강우 | B등급 / 과거만 |
| 6장 잔차 | 사상 단위 OOF | 누수 없음 |

### 정리

| md | 발견 | 판정 · 전체 확인 | 모델링 시사점 |
|---|---|---|---|
| T1 하강 가지 | 비조석 하강 중 행: 평균 +0.052 but **중앙값 −0.01**, 10cm 이상 큰 하강은 중앙값 −0.09·68%가 음수. 평균을 끌어올린 건 ② 비 오는 중 하강 뒤 재상승(양수 40%)과 ③ **댐 방류 의심**(양수 61%, 중앙값 +0.04). ① 1cm 계단은 원인이 아님 | 관측소별 β > 0: 일반 362곳 중 **78% → ✅ 채택**(방향). 크기는 β 중앙값 0.67, md 예상 0.7~1 안은 39%뿐 → "β≈0.7~1"은 🔶 | **부딪힘 해소**: 기존 +0.016은 평균의 착시. 하강 가지는 음수 쪽이 맞지만 비 오는 중엔 재상승이 섞임. 비 × dwl_1h 상호작용이 중요 |
| T2 가속도 | 1h 변화가 같은 칸 안에서 가속도 효과를 보면 **오히려 음(−)**: 30cm↑ 급상승에서 감속 0.82 > 가속 0.66 | 일반 관측소 42%, 사상 11%만 양(+) → **❌ 기각** | 급상승 중 '가속'은 첨두가 가깝다는 뜻일 수 있음. 잔차: 전체 −1.1%, 상위 5사상 제외 **+1.9%**, 사상 일관성 71%(음의 방향) → 약한 후보, 부호 제약 금지 |
| T3 사상 단계 | 초기 칸(24h 변화 작고 상승 중) 0.18m vs 첨두 후 칸(24h 크고 하강 중) 0.04m | 사상 74개 중 **99%** → **✅ 채택** | 격자 칸의 잔차 설명 몫 −1.0% → 기준 모델이 `dwl_24h`·`dwl_1h`로 이미 씀. 새 변수 불필요 |
| A12 감수 | 관측소별 감수 지수 b 중앙값 0.00, b>0 22% | **❌ 기각**(수위로는 규칙 안 보임) | 감쇠비의 잔차 신호도 없음(−0.3~−0.6%) |

- **1h 30cm 이상 급상승 행의 과소예측(+0.10m)** 은 여기 네 변수로 설명되지 않아요(가속도 잔차 −10.6%). 9권의 도착 예정 비·선행 습윤에서 다시 봐요.
- 설명 가능 몫이 음수 → "기준 모델이 이미 정보를 씀"이라는 뜻이지 "관계가 없다"는 뜻이 아니에요.

### 2단계 점검
- T1: 부딪힘 해소 결론(평균 착시)을 판정 모음에 올릴지
- 파생변수 후보: 가속도는 약한 후보(음의 방향), 단계 격자·감쇠비는 후보에서 제외
""")
b.write("nb8_stage_and_target.ipynb")

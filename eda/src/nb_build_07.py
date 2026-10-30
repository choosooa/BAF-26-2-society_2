"""7권: 구간 나누기 — 비 말고 다른 이유로 움직이는 관측소 (선행연구 C3·C4·C5, B7·C2)."""
from nb_common import Book, RESEARCH_HELPERS

b = Book()
b.md("""
# 🚧 7권 · 비 말고 다른 이유로 움직이는 관측소 — "구간 나누기"

> 시리즈: 1~6권 · 별책 조석 → **7권 구간 나누기** → 8권 수위 국면 → 9권 도달시간·강우 창 → 10권 조절 효과·판정
> 근거: `선행연구/` md의 C3(댐 방류)·C4(보 관리수위)·C5(초단기 반응)·B7·C2(조석), 계획서 `research_eda_plan.md` 2장

### 한 줄 요약
**비 없이 물이 불쑥 오르는 관측소(댐 방류 의심), 비가 와도 수위가 잘 안 움직이는 관측소(보 관리수위 의심), 너무 빨리 반응하는 소하천(초단기 의심)을 입력만으로 골라 '구간 플래그'를 만들어요. 8~10권은 이 구간으로 나눠서 봐요.**

| 장 | 질문 |
|---|---|
| 1 | 세 가지 모양을 가짜 데이터로 익히기 |
| 2 | C3 댐 방류 의심 — 비로 설명 안 되는 상승 |
| 3 | C4 보 관리수위 의심 — 비가 와도 안 움직임 |
| 4 | C5 초단기 반응 의심 — 플래그만 |
| 5 | (가볍게) B7 조석 위상 · C2 홍수와 조석 진폭 |
| 6 | 구간 플래그 표와 구간별 오차 |
| 7 | 누수 점검과 정리 |

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **무강우 행** | 과거 24시간 지상강수 < 0.5mm 그리고 IMERG < 1mm | 하루 종일 맑았던 시각 |
| **마른 상승 비율** | 무강우 행 중 1시간에 5cm 이상 오른 비율 | 비가 없는데 물이 불쑥 → 방류 의심 |
| **계단 시작** | 직전 2시간 평평(±1cm)하다가 1시간에 5cm 이상 오름 | 수문을 연 순간 |
| **비 올 때 정지 비율** | 6시간 강우 10mm 이상인데 1시간 변화가 ±1cm 이하인 비율 | 보가 수위를 붙잡고 있음 |
| **상류 유입 우세** | `wsarea / cat_area` | 클수록 물 대부분이 멀리 상류에서 옴 |
| **남은 오차(잔차)** | 타깃 − 기준 모델(6권 HGB) 예측 | +면 모델이 덜 예측 |
| **의심** | 좌표가 없어 실제 댐·보는 확인 못 해요. 모양으로만 판단한다는 뜻이에요 | |
""")
b.setup(RESEARCH_HELPERS)
b.code(r'''
tr = load_resid(add_flags(pd.read_parquet(CACHE / "train_sorted.parquet")))
tr["phase"] = phase_of(tr)
key = keyed(tr)
print(f"train {len(tr):,}행 · 관측소 465곳 · 기준 모델 RMSE {np.sqrt((tr.resid**2).mean()):.4f} (6권 0.2264와 같아야 함)")
''')

b.md("""
## 1. 세 가지 모양을 가짜 데이터로 익히기

실제 데이터를 보기 전에, 우리가 찾을 **세 가지 모양**을 가짜로 만들어 봐요.
- **보통 하천**: 비가 오면 오르고 천천히 빠져요.
- **방류 계단**: 비가 없는데 어느 순간 **계단처럼 확** 올랐다가 한동안 유지돼요(댐 수문을 연 경우).
- **관리수위**: 비가 와도 수위가 **거의 평평**해요. 가끔 수문을 열면 뚝 떨어져요(보).
""")
b.code(r'''
t = np.arange(72); rain = np.zeros(72); rain[10:16] = [2, 8, 15, 10, 4, 1]
natural = np.convolve(rain, np.exp(-np.arange(24) / 5) * .02, mode="full")[:72]          # 비 → 부드러운 상승과 감수
dam = np.where((t >= 40) & (t < 55), .8, 0.) + np.random.default_rng(1).normal(0, .005, 72)  # 비 없이 계단 상승
weir = np.full(72, 1.0); weir[30:] -= .4; weir += np.random.default_rng(2).normal(0, .003, 72)  # 평평하다가 수문 개방으로 뚝
fig, ax = plt.subplots(1, 3, figsize=(14, 3.2), sharex=True)
for a, y, ttl, c in zip(ax, [natural, dam, weir], ["보통 하천: 비 → 상승 → 감수", "방류 계단: 비 없이 불쑥", "관리수위: 비가 와도 평평"], [C_NT, C_TIDAL, C_PURPLE]):
    a.plot(t, y, color=c, lw=2); a.set_title(ttl); a.set_xlabel("시간 (h)")
    a2 = a.twinx(); a2.bar(t, rain, color=C_RAIN, alpha=.4); a2.invert_yaxis(); a2.set_yticks([])
ax[0].set_ylabel("수위 (m)"); plt.tight_layout(); plt.show()
''')
b.md("""
가짜 데이터에서 본 것처럼, **비(노란 막대)와 수위 모양이 어긋나는 방식**으로 세 종류를 구별할 수 있어요. 이제 이 모양을 숫자(지표)로 바꿔 465곳 전부에 적용해요.
""")

b.md("""
## 2. C3 댐 방류 의심 — "비로 설명 안 되는 상승"

> **md C3** · 가설: 댐 방류는 국지 강우와 무관한 계단형 급상승을 만든다 · 확인: 입력만으로 만든 지표로 관측소를 줄 세우고, 이미 아는 조작 관측소가 위에 오는지, 떨어져 나온 무리가 있는지, 기준 모델이 그 무리를 덜 예측하는지 본다

지표 두 개를 **입력만으로** 계산해요(타깃 안 씀 → 모델 변수로 써도 되는 B등급).
1. **마른 상승 비율**: 무강우 행 중 `dwl_1h ≥ +5cm` 비율 (무강우 행 50개 이상 관측소만)
2. **계단 시작 비율**: 전체 행 중 "직전 2시간 평평 + 1시간 5cm 이상 상승" 비율. 직전 2시간 변화는 `wl_t−1 − wl_t−3 = dwl_3h − dwl_1h`로 계산해요(과거 값만).

**지표가 제대로 작동하나?** 별책에서 이미 찾은 **24시간 조작 관측소 2곳(ST0194, ST0200)** 은 사람이 수위를 움직이는 곳이에요. 좋은 지표라면 이 두 곳이 일반 관측소 분포의 **위쪽 끝**에 와야 해요.
""")
b.code(r'''
dry = (tr.sfc_rain_24h < .5) & (tr.imerg_rain_24h < 1.0)                  # 과거 24시간만 봄
flat2 = (tr.dwl_3h - tr.dwl_1h).abs() <= .0101                           # 직전 2시간 평평(±1cm, 부동소수 여유)
rain6 = np.maximum(tr.sfc_rain_6h, tr.imerg_rain_6h)
g = tr.assign(dry=dry, dryrise=dry & (tr.dwl_1h >= .0499), stepup=flat2 & (tr.dwl_1h >= .0499),
              stepdn=flat2 & (tr.dwl_1h <= -.0499), wet=rain6 >= 10, wetflat=(rain6 >= 10) & (tr.dwl_1h.abs() <= .0101))
st = g.groupby("station_id").agg(n=("dry", "size"), n_dry=("dry", "sum"), dryrise=("dryrise", "sum"), stepup=("stepup", "mean"),
        stepdn=("stepdn", "mean"), n_wet=("wet", "sum"), wetflat=("wetflat", "sum"), group=("group", "first"),
        wsarea=("wsarea", "first"), cat_area=("cat_area", "first"), mean_slope=("mean_slope", "first"))
st["dryrise_rate"] = np.where(st.n_dry >= 50, st.dryrise / st.n_dry.clip(1), np.nan)
st["wetflat_rate"] = np.where(st.n_wet >= 50, st.wetflat / st.n_wet.clip(1), np.nan)
st["up_ratio"] = st.wsarea / st.cat_area
gen = st[st.group == "일반"]
print(f"무강우 행 비율 {dry.mean():.1%}")
for c, lab in [("dryrise_rate", "마른 상승 비율"), ("stepup", "계단 시작 비율")]:
    pct = {s: f"{(gen[c] < st.loc[s, c]).mean():.1%}" for s in OPERATED_24H}
    print(f"{lab}: 일반 중앙값 {gen[c].median():.4f} · 95% 지점 {gen[c].quantile(.95):.4f} · 조작 관측소의 일반 분포 내 위치 {pct}")
''')
b.code(r'''
DAM_CUT = gen.dryrise_rate.quantile(.95)          # 잠정 문턱: 일반 관측소 95% 지점 (1단계 점검에서 확정)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].hist(gen.dryrise_rate.dropna(), bins=60, color=C_NT)
for s, c in zip(OPERATED_24H, [C_PINK, C_PURPLE]): ax[0].axvline(st.loc[s, "dryrise_rate"], color=c, lw=2, label=f"{s} (24h 조작)")
ax[0].axvline(DAM_CUT, color="k", ls="--", label="잠정 문턱(일반 95%)"); ax[0].set_yscale("log"); ax[0].legend()
ax[0].set_xlabel("마른 상승 비율"); ax[0].set_ylabel("관측소 수 (log)"); ax[0].set_title("일반 관측소 442곳의 분포와 조작 관측소 위치")
ax[1].scatter(gen.up_ratio, gen.dryrise_rate, s=10, color=C_NT, alpha=.6); ax[1].set_xscale("log")
ax[1].set_xlabel("상류 유입 우세 wsarea/cat_area (log)"); ax[1].set_ylabel("마른 상승 비율"); ax[1].axhline(DAM_CUT, color="k", ls="--")
rho = stats.spearmanr(gen.up_ratio, gen.dryrise_rate, nan_policy="omit")[0]
ax[1].set_title(f"상류 유입이 우세할수록 마른 상승이 잦나 (ρ={rho:.2f})"); plt.tight_layout(); plt.show()
dam_ids = gen.index[gen.dryrise_rate >= DAM_CUT].tolist()
print(f"댐 방류 의심(잠정): {len(dam_ids)}곳 → {dam_ids}")
print(f"의심 관측소 상류 유입 우세 중앙값 {gen.loc[dam_ids, 'up_ratio'].median():.0f} vs 나머지 {gen.drop(dam_ids).up_ratio.median():.0f}")
''')
b.md("""
### 2-1. 이 무리를 모델이 덜 예측하나? (분석용 · 타깃 사용)
의심 관측소의 **무강우 행**과 **비·상승 중 행**에서 남은 오차의 평균(+면 덜 예측)과 RMSE를 나머지 일반 관측소와 비교해요. 또 "비 없이 큰 상승(과거 6h 강우 < 1mm, 타깃 ≥ 0.5m)"이 어디서 나오는지 봐요.
""")
b.code(r'''
tr["dam_sus"] = tr.station_id.isin(dam_ids)
base = tr[tr.group == "일반"]
rows = []
for ph in ["무강우·반응 없음", "무강우·상승 중", "비·상승 중", "비·아직 반응 없음"]:
    for flag, lab in [(True, "댐 의심"), (False, "나머지 일반")]:
        s = base[(base.phase == ph) & (base.dam_sus == flag)]
        rows.append({"국면": ph, "무리": lab, "행": len(s), "평균 잔차 (m)": s.resid.mean(), "RMSE": np.sqrt((s.resid ** 2).mean())})
display(pd.DataFrame(rows).round(4))
bigdry = base[(np.maximum(base.sfc_rain_6h, base.imerg_rain_6h) < 1) & (base[TARGET] >= .5)]
print(f"비 없이 큰 상승: {len(bigdry)}행 · 그중 댐 의심 관측소 몫 {bigdry.dam_sus.mean():.0%} (댐 의심 관측소의 행 비중 {base.dam_sus.mean():.0%})")
''')

b.md("""
## 3. C4 보 관리수위 의심 — "비가 와도 안 움직이는 곳"

> **md C4** · 가설: 보 상류 관측소는 비가 와도 수위가 일정하게 유지되어 상승이 억제된다 · 확인: 비 올 때 정지 비율로 줄 세우고, 떨어진 무리의 타깃 반응과 잔차를 본다

- **비 올 때 정지 비율**: 6h 강우(두 원천 중 큰 값) 10mm 이상 행 중 `|dwl_1h| ≤ 1cm` 비율 (그런 행 50개 이상 관측소만)
- **계단 하강 비율**: 직전 2시간 평평 + 1시간 5cm 이상 하강 (수문 개방 모양)
""")
b.code(r'''
WEIR_CUT = gen.wetflat_rate.quantile(.95)           # 잠정 문턱
fp = pd.read_csv(RESULTS / "s08_station_fingerprints.csv", index_col=0)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].hist(gen.wetflat_rate.dropna(), bins=50, color=C_NT); ax[0].axvline(WEIR_CUT, color="k", ls="--", label="잠정 문턱(일반 95%)")
ax[0].set_xlabel("비 올 때 정지 비율"); ax[0].set_ylabel("관측소 수"); ax[0].legend(); ax[0].set_title("일반 442곳 분포")
ax[1].scatter(gen.wetflat_rate, gen.stepdn, s=10, color=C_NT, alpha=.6)
ax[1].set_xlabel("비 올 때 정지 비율"); ax[1].set_ylabel("계단 하강 비율"); ax[1].axvline(WEIR_CUT, color="k", ls="--")
ax[1].set_title("정지가 잦은 곳에서 계단 하강도 잦나"); plt.tight_layout(); plt.show()
weir_ids = gen.index[gen.wetflat_rate >= WEIR_CUT].tolist()
print(f"보 관리수위 의심(잠정): {len(weir_ids)}곳 → {weir_ids}")
print("정지 비율과 계단 하강의 순위상관", round(stats.spearmanr(gen.wetflat_rate, gen.stepdn, nan_policy='omit')[0], 2))
print("4권 유형 분포(의심 관측소):", fp.loc[weir_ids, "type"].value_counts().to_dict(), "· 전체 일반:", fp.loc[gen.index, "type"].value_counts().to_dict())
tr["weir_sus"] = tr.station_id.isin(weir_ids)
wet = tr[(tr.group == "일반") & (np.maximum(tr.sfc_rain_6h, tr.imerg_rain_6h) >= 10)]
display(wet.groupby("weir_sus").agg(행=(TARGET, "size"), 평균_타깃=(TARGET, "mean"), 평균_잔차=("resid", "mean"), RMSE=("resid", lambda v: np.sqrt((v ** 2).mean()))).round(4))
''')

b.md("""
## 4. C5 초단기 반응 의심 — 플래그만

> **md C5** · 가설: 소유역·급경사는 도달시간이 매우 짧아, 앞으로 올 비(입력에 없음)가 타깃을 좌우한다

여기선 **플래그만** 정해요. 기존 지체시간(3권, train 전체로 추정한 `lag_sfc_rain_1h`)이 **1시간 이하**이면서 상류면적이 일반 관측소 **하위 1/3**인 곳이에요. 9권에서 사상 단위 지체시간으로 다시 확인하고 깊게 봐요.
""")
b.code(r'''
lag = pd.read_csv(RESULTS / "s04_station_summary.csv", index_col=0)["lag_sfc_rain_1h"]
st["lag_sfc"] = lag
area_cut = gen.wsarea.quantile(1 / 3)
fast_ids = st.index[(st.group == "일반") & (st.lag_sfc <= 1) & (st.wsarea <= area_cut)].tolist()
print(f"상류면적 하위 1/3 문턱 {area_cut:.1f} · 초단기 의심 {len(fast_ids)}곳")
print(f"겹침: 댐 의심과 {len(set(fast_ids) & set(dam_ids))}곳, 보 의심과 {len(set(fast_ids) & set(weir_ids))}곳")
''')

b.md("""
## 5. (가볍게) 조석 구간 보강 — B7 위상 변수 · C2 홍수와 조석 진폭

> **md B7** · 가설: 위상 대용 변수 `atan2(dWL3, dWL6)`가 조석 위치를 알려준다 · 확인: 조석 15곳에서 기존 `dwl_6h` 10칸과 **설명 가능 몫**을 비교(수치 하나)
> **md C2** · 가설: 하천 유량이 크면 조석 진폭이 줄어든다 · 확인: 조석 15곳 각각 "과거 13시간 진폭 ↔ 24h 강우" 순위상관의 부호

**위상 변수 예시(가짜)**: 조석을 원 위의 점이라고 생각하면, `dwl_6h`(반 주기 전과의 차이)와 `dwl_3h`(1/4 주기 전과의 차이)는 그 점의 두 좌표 같은 역할을 해요. `atan2`는 두 좌표로 **각도(위상)** 를 구해요.

**진폭**은 과거 13시간 수위(이전 행)에서 직선 추세를 뺀 뒤의 최대−최소예요. 홍수로 전체가 오르는 건 추세로 빠지고, 조석의 오르내림만 남아요.
""")
b.code(r'''
td = tr[tr.station_id.isin(TIDAL)].copy()
td["phi"] = np.arctan2(td.dwl_3h, td.dwl_6h)
# 관측소 안에서 타깃을 표준화해 비교(관측소마다 조차가 달라서)
td["z"] = td.groupby("station_id")[TARGET].transform(lambda v: (v - v.mean()) / v.std())
td["q6"] = td.groupby("station_id").dwl_6h.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
td["qphi"] = td.groupby("station_id").phi.transform(lambda v: pd.qcut(v.rank(method="first"), 12, labels=False))
def eta2(y, k): m = y.groupby(k).transform("mean"); return 1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"타깃(관측소 표준화) 중 칸 평균으로 설명되는 몫: dwl_6h 10칸 {eta2(td.z, [td.station_id, td.q6]):.1%} · 위상 12칸 {eta2(td.z, [td.station_id, td.qphi]):.1%}")
rc = resid_check("조석 위상 φ", td.phi.to_numpy(), td)
print("잔차 점검(조석 15곳):", rc)

# C2: 과거 13시간 진폭 (이전 행에서 가져옴, 10개 이상 있어야 계산)
tk = keyed(td); W = np.column_stack([lookup_past(td, tk, "wl_t", k) for k in range(13)])
tt = -np.arange(13.0); okc = np.isfinite(W).sum(1) >= 10
def detr_range(w):
    m = np.isfinite(w); p = np.polyfit(tt[m], w[m], 1); r = w[m] - np.polyval(p, tt[m]); return r.max() - r.min()
amp = np.full(len(td), np.nan); idx = np.where(okc)[0]
amp[idx] = [detr_range(W[i]) for i in idx]
td["amp"] = amp; td["rain24"] = np.maximum(td.sfc_rain_24h, td.imerg_rain_24h)
r_st = group_spearman(td, td.amp, td.rain24, min_n=50)
sh, n = share_same_sign(r_st, -1)
print(f"C2: 비가 많을수록 진폭이 작아지는(ρ<0) 조석 관측소 {sh:.0%} ({n}곳 중) · ρ 중앙값 {r_st.median():.2f}")
''')

b.md("""
## 6. 구간 플래그 표와 구간별 오차

지금까지 만든 구간을 별책의 기존 구간(조석·약한 조석·24h 조작·고정)과 합쳐요. 한 관측소는 **하나의 구간**에만 들어가게 우선순위를 둬요: 기존 구간 → 댐 의심 → 보 의심 → 초단기 의심 → 일반.
""")
b.code(r'''
def regime(s):
    g0 = st.loc[s, "group"]
    if g0 != "일반": return g0
    if s in dam_ids: return "댐 방류 의심"
    if s in weir_ids: return "보 관리수위 의심"
    if s in fast_ids: return "초단기 의심"
    return "일반(나머지)"
st["regime"] = [regime(s) for s in st.index]
tr["regime"] = tr.station_id.map(st.regime)
sse = (tr.resid ** 2).sum()
tab = tr.groupby("regime").agg(관측소=("station_id", "nunique"), 행=("resid", "size"), 오차합=("resid", lambda v: (v ** 2).sum()),
                                RMSE=("resid", lambda v: np.sqrt((v ** 2).mean())), 편향=("resid", "mean"))
tab["행 비중"] = tab.행 / len(tr); tab["오차 비중"] = tab.오차합 / sse
display(tab.drop(columns="오차합").sort_values("오차 비중", ascending=False).round(4))
out = st[["group", "regime", "dryrise_rate", "stepup", "stepdn", "wetflat_rate", "up_ratio", "lag_sfc", "wsarea"]]
out.to_csv(RESULTS / "nb07_regimes.csv")
import json
json.dump({"dam_cut": float(DAM_CUT), "weir_cut": float(WEIR_CUT), "area_cut": float(area_cut), "dam_ids": dam_ids, "weir_ids": weir_ids,
           "fast_ids": fast_ids, "regime_table": tab.drop(columns="오차합").round(4).reset_index().to_dict("records")},
          open(RESULTS / "nb07_summary.json", "w"), ensure_ascii=False, indent=1)
print("저장: nb07_regimes.csv, nb07_summary.json")
''')
b.md("""
## 7. 누수 점검과 정리

### 누수 점검표

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 2·3장 지표(마른 상승, 계단, 비 올 때 정지) | 과거 강우·과거 수위 차분만 | **B등급**: train에서 계산해 test에 적용 가능(관측소 동일) |
| 2-1·3장 잔차·타깃 비교 | 타깃 사용 | 분석용. 모델 변수로 쓰면 학습 묶음 안에서만(C등급) |
| 4장 초단기 플래그 | 기존 지체시간(과거 강우 ↔ 현재 수위 변화) | B등급 |
| 5장 위상·진폭 | 이전 행(과거)만 | A등급(위상), 진폭도 과거 13시간만 |
| 문턱(95% 지점) | 일반 관측소 분포에서 정함 | 구간의 **정의**라 가설 판정과 별개. 1단계 점검에서 확정 |

> 🚫 test 안의 뒤 시각 행은 쓰지 않았어요. "앞으로 비가 안 온다"는 조건도 쓰지 않았어요(무강우는 과거 24시간 기준).

### 정리

| md | 발견 | 판정 · 전체 확인 | 모델링 시사점 |
|---|---|---|---|
| C3 댐 방류 | 마른 상승 비율이 일반 95% 지점(6.2%)을 넘는 **23곳**. 이미 아는 조작 관측소 2곳이 일반 분포의 97.7%·99.5% 지점에 와서 지표가 작동함을 확인했어요. 이 23곳은 상류 유입 우세가 3배(중앙값 87 vs 28)이고, "비 없이 큰 상승"의 23%가 행 비중 5%인 이 무리에서 나와요 | 🔶 **부분**: 구간은 뚜렷하지만 기준 모델 오차는 주로 **무강우·반응 없음**(RMSE 0.150 vs 0.075, 평균 잔차 +0.019)에서만 커요. 비·상승 중 편향은 없어요 | `댐 방류 의심` 플래그(B등급)와 `wsarea/cat_area`를 후보로. 오차 비중 7%라 효과는 작을 것 |
| C4 보 관리수위 | 비 올 때 정지 비율 상위 **23곳**. 비가 와도 평균 타깃 0.08m(나머지 0.21m) | ⚠️ 이 무리의 17곳이 4권 **C1 조용한 산지 소하천**이에요. 계단 하강과는 오히려 음의 상관(−0.46). 즉 "보"라기보다 **원래 반응이 작은 하천**을 잡은 것 → **⏸ 보류**(보를 구별할 정보 부족) | 오차 비중 1.9%, 모델이 이미 잘 맞힘(RMSE 0.146). 별도 처리 불필요 |
| C5 초단기 | 기존 지체 ≤ 1h & 소유역이 **113곳**(너무 넓음) | 판정은 9권 | 9권에서 사상 단위 지체로 좁힘 |
| B7 조석 위상 | 위상 12칸이 타깃의 60.5%, `dwl_6h` 10칸이 56.2%를 설명 | 기준 모델 잔차엔 신호 없음(설명 가능 몫 −1.5%, 사상 일관성 38%) → **❌ 기각**(모델이 이미 lag로 위상을 씀) | 추가 불필요 |
| C2 진폭–홍수 | 비가 많을 때 진폭이 작아지는 조석 관측소 40%, ρ 중앙값 0.02 | **❌ 기각** | – |

### 1단계 점검에서 확인할 것
- 댐 의심 문턱(일반 95%, 23곳)과 목록
- 보 의심은 "반응이 작은 하천"으로 해석하고 구간에서 빼거나 이름을 바꿀지
- 초단기 의심 113곳은 9권에서 좁힘
""")
b.write("nb7_regimes.ipynb")

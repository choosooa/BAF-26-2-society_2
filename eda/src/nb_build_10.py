"""10권: 조절 효과와 판정 (선행연구 T5·A10·A11·A13·C6·A7·A14·B9)."""
from nb_common import Book, RESEARCH_HELPERS

b = Book()
b.md("""
# 🎛️ 10권 · 조절 효과와 판정 — "같은 비라도 누가 더 오르나"

> 시리즈: 7권 구간 → 8권 수위 국면 → 9권 도달시간·강우 창 → **10권 조절 효과·판정**
> 근거: 선행연구 md의 T5·A10·A11(수위 높이), A13·C6(도시화), A7·A14(면적·산림), B9(단조 제약). 계획서 5장

### 한 줄 요약
**도시화 가설(A13)은 '중간 강우'에서는 맞아요. 도시 관측소는 1시간 비에 더 민감하고 6~24시간 전 비에는 덜 민감해요(사상 78%). 기존의 반대 결과는 40mm 넘는 극한 비에서만 나타나는데, md 주의사항(극한호우에선 도시 효과가 줄어듦)과 맞아요. 같은 높이라도 오르는 중이면 더 오르고(A11, 100%), 관측소 안에서 수위가 높을수록 순위로는 덜 오르지만 평균으로는 분명하지 않아요(A10). 강우·레이더는 비조석 관측소에서 + 단조 방향이 매우 일관돼요.**

| 장 | 질문 |
|---|---|
| 1 | T5 관측소 안 수위 높이(분위) |
| 2 | A10 수위가 높을수록 같은 비에 덜 오르나 |
| 3 | A11 같은 높이에서 상승 중 vs 하강 중 |
| 4 | A13·C6 도시화 (기존 결과와 부딪히는 곳) |
| 5 | (가볍게) A7 면적, A14 산림 |
| 6 | B9 단조 방향 네 겹 점검 |
| 7 | 누수 점검과 정리 |

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **수위 분위 q** | 그 관측소의 train 수위 중 지금 수위보다 낮은 비율(0~1) | q=0.95 → 평소보다 아주 높은 물 |
| **수위–유량 곡선** | Q = a(h−c)^b. b>1이면 물이 높을수록 같은 유량 증가에 수위가 덜 오름 | 강폭이 넓어지는 효과 |
| **이력** | 같은 수위라도 오를 때와 내릴 때 흐름이 다름 | |
| **단조 제약** | 트리 모델에 "이 변수가 커지면 예측도 커져야 한다"를 강제 | 강우 ↑ → 예측 ↑ |
| **조건부 부호** | 다른 핵심 변수를 같게 맞춘 칸 안에서의 방향 | 같은 1h 수위 변화 안에서 강우 효과 |
""")
b.setup(RESEARCH_HELPERS)
b.code(r'''
tr = load_resid(add_flags(pd.read_parquet(CACHE / "train_sorted.parquet")))
tr["phase"] = phase_of(tr)
reg = pd.read_csv(RESULTS / "nb07_regimes.csv", index_col=0); lag = pd.read_csv(RESULTS / "nb09_station_lag.csv", index_col=0)
tr["regime"] = tr.station_id.map(reg.regime)
tr["r0_1"], tr["r1_6"], tr["r6_24"], tr["r0_6"] = tr.sfc_rain_1h, tr.sfc_rain_6h - tr.sfc_rain_1h, tr.sfc_rain_24h - tr.sfc_rain_6h, tr.sfc_rain_6h
tr["acc"] = tr.dwl_1h - (tr.dwl_3h - tr.dwl_1h) / 2
tr["q"] = tr.groupby("station_id").wl_t.rank(pct=True)          # train 기준 관측소 안 분위 (B등급)
gen_ids = reg.index[reg.group == "일반"]
print(f"train {len(tr):,}행")
''')

b.md("""
## 1. T5 관측소 안 수위 높이

> **md T5** · 절대 수위는 관측소마다 0점이 달라 비교가 안 되고, 관측소 안에서 정규화해야 의미가 생긴다

**예시**: A 관측소 평소 1.0m, B 관측소 평소 85m. 둘 다 "평소보다 아주 높음"이면 q ≈ 0.95로 같아져요.
""")
b.code(r'''
nt = tr[tr.nontidal]
print(f"타깃 순위상관: 절대 수위 wl_t {stats.spearmanr(nt.wl_t, nt[TARGET])[0]:.3f} · 관측소 안 분위 q {stats.spearmanr(nt.q, nt[TARGET])[0]:.3f}")
print("잔차 점검:", resid_check("수위 분위 q", tr.q, tr, tr.phase.str.startswith("비")))
''')

b.md("""
## 2. A10 수위가 높을수록 같은 비에 덜 오르나

> **md A10** · 가설: 수위–유량 곡선이 b>1이면 고수위일수록 같은 강우에 y가 작고, 홍수터에서 급감 · 확인: `r0_6` 칸 × q 5칸 평균 타깃. 첨두 근처 효과와 섞이지 않게 **상승 중 행만** 따로. 관측소·사상별로 (강우 칸 중심화 타깃 ↔ q) 부호 · 기대 방향 **음(−)**
""")
b.code(r'''
w = tr[tr.nontidal & (tr.r0_6 >= 1)].copy()
w["rb"] = pd.cut(w.r0_6, [1, 5, 15, 40, 1e4]); w["qb"] = pd.cut(w.q, [0, .2, .4, .6, .8, 1], include_lowest=True)
display(w.pivot_table(index="rb", columns="qb", values=TARGET, aggfunc="mean", observed=True).round(3))
wu = w[w.dwl_1h > .01]
print("상승 중 행만:"); display(wu.pivot_table(index="rb", columns="qb", values=TARGET, aggfunc="mean", observed=True).round(3))
w["cb"] = w.rb.astype(str) + np.where(w.dwl_1h > .01, "상승", np.where(w.dwl_1h < -.01, "하강", "정지"))
w["y_c"] = w[TARGET] - w.groupby("cb")[TARGET].transform("mean")
ov = np.sign(stats.spearmanr(w.q, w.y_c)[0])
(s1, n1), (s2, n2) = share_same_sign(group_spearman(w, w.q, w.y_c)[lambda r: r.index.isin(gen_ids)], -1), share_same_sign(group_spearman(w, w.q, w.y_c, "event_block", 200), -1)
print(f"A10: 전체 부호 {ov:+.0f} · 음(−)인 일반 관측소 {s1:.0%} ({n1}) · 사상 {s2:.0%} ({n2}) → {verdict(ov, -1, min(s1, s2))}")
w["qr"] = w.q * np.log1p(w.r0_6)
print("잔차 점검:", resid_check("q × log1p(r0_6)", w.qr, w, w.dwl_1h > .01))
''')

b.md("""
## 3. A11 같은 높이에서 상승 중 vs 하강 중

> **md A11** · 가설: 같은 수위라도 상승부와 하강부가 다르다 · 확인: 관측소마다 같은 q 칸 안에서 (상승 중 평균 타깃 − 하강 중 평균 타깃)을 칸별로 구해 평균, 양(+)인 관측소 비율
""")
b.code(r'''
t3 = tr[tr.nontidal & (tr.dwl_1h.abs() > .01)].copy()
t3["qb"] = pd.cut(t3.q, 5, labels=False); t3["up"] = t3.dwl_1h > 0
m = t3.groupby(["station_id", "qb", "up"])[TARGET].agg(["mean", "size"]).query("size >= 10")["mean"].unstack()
dif = (m[True] - m[False]).dropna().groupby(level=0).mean()
s11, n11 = share_same_sign(dif[dif.index.isin(gen_ids)], 1)
print(f"A11: 같은 높이에서 상승 중이 더 큰 일반 관측소 {s11:.0%} ({n11}곳) · 차이 중앙값 {dif.median():.3f} m → {verdict(np.sign(dif.median()), 1, s11)}")
''')

b.md("""
## 4. A13·C6 도시화 — 기존 결과와 부딪히는 곳

> **md A13·C6** · 가설: 도시화율이 높으면 첨두가 크고 빠르다 → 1h 강우 기울기는 크고 `r6_24` 기울기는 작다. md 주의사항: 극한호우에서는 도시 효과가 작아진다 · 기존: 40mm 넘는 비에서 도시 하위 0.40 vs 상위 0.32m
> 확인 순서: ① 기존 결과 재현 ② 중간 강우(6h 5~40mm)에서 도시 3분위별 `r0_1`·`r6_24` 순위상관 ③ 상류면적 3분위 **안에서** 비교 ④ 사상별 재현 · 판정: "도시 상위의 r0_1 상관 > 하위" 인 비율
""")
b.code(r'''
ug = tr[tr.nontidal].copy()
st_env = tr.groupby("station_id")[["urban_ratio", "wsarea"]].first()
ug["ut"] = ug.station_id.map(pd.qcut(st_env.urban_ratio.rank(method="first"), 3, labels=["하", "중", "상"]))
ug["at"] = ug.station_id.map(pd.qcut(st_env.wsarea.rank(method="first"), 3, labels=["소", "중", "대"]))
print("① 6h 40mm 초과, 도시 3분위 평균 타깃:", ug[ug.r0_6 > 40].groupby("ut", observed=True)[TARGET].mean().round(3).to_dict())
mid = ug[ug.r0_6.between(5, 40)]
rows = []
for a in ["소", "중", "대"]:
    for u in ["하", "상"]:
        d = mid[(mid["at"] == a) & (mid.ut == u)]   # .at 은 pandas 예약어라 대괄호로
        rows.append({"면적": a, "도시": u, "행": len(d), "평균 타깃": d[TARGET].mean(), "ρ(r0_1)": stats.spearmanr(d.r0_1, d[TARGET])[0], "ρ(r6_24)": stats.spearmanr(d.r6_24, d[TARGET])[0]})
display(pd.DataFrame(rows).round(3))
ev = []
for e, d in mid.groupby("event_block"):
    hi, lo = d[d.ut == "상"], d[d.ut == "하"]
    if len(hi) >= 100 and len(lo) >= 100:
        ev.append({"e": e, "r1": stats.spearmanr(hi.r0_1, hi[TARGET])[0] - stats.spearmanr(lo.r0_1, lo[TARGET])[0], "mean": hi[TARGET].mean() - lo[TARGET].mean()})
ev = pd.DataFrame(ev).dropna()
s13 = (ev.r1 > 0).mean()
print(f"④ 사상별: 도시 상위의 r0_1 상관이 더 큰 사상 {s13:.0%}, 평균 타깃이 더 큰 사상 {(ev['mean'] > 0).mean():.0%} ({len(ev)}개) → {verdict(np.sign(ev.r1.median()), 1, s13)}")
''')

b.md("""
## 5. (가볍게) A7 면적 · A14 산림

- **A7**: 비유량(면적당 반응)은 큰 유역일수록 작다 → 6h 강우 5mm 이상에서 타깃 ÷ 강우 중앙값을 상류면적 3분위별로
- **A14**: 산림은 유출을 줄인다(방향 모호) → 일반 관측소 단위, 면적(log)·경사를 통제한 산림과 평균 타깃의 부분 순위상관
""")
b.code(r'''
bb = ug[ug.r0_6 >= 5]
a7 = (bb[TARGET] / bb.r0_6).groupby(bb["at"], observed=True).median()
print("A7 타깃÷6h 강우 중앙값, 면적 3분위:", a7.round(4).to_dict())
S = tr[tr.station_id.isin(gen_ids)].groupby("station_id").agg(y=(TARGET, "mean"), forest=("forest_ratio", "first"), area=("wsarea", "first"), slope=("mean_slope", "first")).rank()
Z = np.column_stack([np.ones(len(S)), S.area, S.slope])
res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
pc14 = np.corrcoef(res(S.y.to_numpy()), res(S.forest.to_numpy()))[0, 1]
print(f"A14 산림 부분 순위상관(면적·경사 통제) {pc14:.2f}")
''')

b.md("""
## 6. B9 단조 방향 네 겹 점검

> **md B9** · 방향이 맞는 변수에만 단조 제약을 건다(틀리면 손실이 큼) · 확인: ① 전체 20칸 평균의 순위상관과 역전 칸 수 ② 일반 관측소 부호 일치율 ③ 구간별(조석·비조석·7권 구간) 부호 ④ 조건부: 강우 변수는 `dwl_1h` 칸 안, `dwl_1h`는 강우 칸 안에서 부호 일치율
> 권고: 모두 기준(70%) 통과 → + 제약 후보 / 일부 구간만 → 구간별 / 나머지 → 제약 없음
""")
b.code(r'''
big_ids = lag.index[lag.wsarea >= lag.loc[gen_ids, "wsarea"].quantile(2 / 3)]
tr["r6_24_big"] = np.where(tr.station_id.isin(big_ids), tr.r6_24, np.nan)
VARS = [("sfc_rain_1h", 1), ("r1_6", 1), ("imerg_rain_1h", 1), ("imerg_rain_6h", 1), ("radar_reflectivity_mean", 1), ("radar_reflectivity_p95", 1),
        ("radar_reflectivity_ge20_fraction", 1), ("dwl_1h", 1), ("r6_24_big", 1), ("acc", 1), ("wl_t", 1), ("dwl_6h", 1), ("r6_24", 1)]
tr["d1b"] = pd.cut(tr.dwl_1h, [-99, -.01, .01, .1, 99]); tr["rb"] = pd.cut(np.maximum(tr.sfc_rain_6h, tr.imerg_rain_6h), [-1, .5, 5, 20, 1e4])
def four(v, exp):
    d = tr[tr[v].notna()]; dn = d[d.nontidal]
    k = pd.qcut(dn[v].rank(method="first"), 20, labels=False); bm = dn.groupby(k)[TARGET].mean()
    r1 = stats.spearmanr(np.arange(len(bm)), bm)[0]; drops = int((np.diff(bm.to_numpy()) * exp < 0).sum())
    s2, _ = share_same_sign(group_spearman(d, d[v], d[TARGET])[lambda r: r.index.isin(gen_ids)], exp)
    reg_sign = {g: np.sign(stats.spearmanr(x[v], x[TARGET])[0]) for g, x in d.groupby("regime") if len(x) > 1000}
    cond = "rb" if v in ("dwl_1h", "dwl_6h", "wl_t") else "d1b"   # 가속도는 1h 변화 칸 안에서 (8권과 같은 통제)
    cs = [np.sign(stats.spearmanr(x[v], x[TARGET])[0]) for _, x in dn.groupby(cond, observed=True) if len(x) > 1000 and x[v].nunique() > 1]
    s4 = np.mean(np.array(cs) == exp)
    ok = [r1 * exp > 0, s2 >= .7, s4 >= .7]; regok = all(sg == exp for g, sg in reg_sign.items() if g != "고정")
    rec = "+ 제약 후보" if all(ok) and regok else ("구간별 제약(비조석만)" if all(ok) else "제약 없음")
    return {"변수": v, "① 칸 순위상관": round(r1, 2), "① 역전 칸": drops, "② 관측소 일치": round(s2, 2), "③ 반대 부호 구간": [g for g, sg in reg_sign.items() if sg != exp],
            "④ 조건부 일치": round(s4, 2), "권고": rec}
B9 = pd.DataFrame([four(v, e) for v, e in VARS]); display(B9)
''')
b.code(r'''
import json
json.dump({"a10": [float(s1), float(s2)], "a11": float(s11), "a13_event": float(s13), "a7": {str(k): float(v) for k, v in a7.items()}, "a14": float(pc14),
           "b9": B9.astype(str).to_dict("records")}, open(RESULTS / "nb10_summary.json", "w"), ensure_ascii=False, indent=1)
''')
b.md("""
## 7. 누수 점검과 정리

### 누수 점검표

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 1~3장 수위 분위 q | 관측소 train 수위 분포 기준 | **B등급**. 주의: 여기선 train 전체 순위라 그 행 이후 수위도 분포에 들어감 → 모델에서는 학습 묶음(train) 분포로 test 값을 변환 |
| 4·5장 도시·면적·산림 | 시간 불변 입력 | 누수 없음 |
| 6장 단조 방향 | 입력 ↔ 타깃 방향만 | 설명용. 제약 켜고 끄기 비교는 모델링 단계 |

### 정리

| md | 발견 | 판정 | 모델링 시사점 |
|---|---|---|---|
| T5 수위 정규화 | 타깃 순위상관: 절대 수위 0.003 → 관측소 안 분위 q −0.063. 잔차 점검 −0.3%(상위 5사상 제외 +1.0%, 사상 일관성 76%) | ✅ 채택(정규화해야 방향이 생김) | q는 약한 후보(B등급, train 분포로 변환) |
| A10 수위–유량 곡선 | 관측소 안 순위로는 음(−): 일반 관측소 80%·사상 96%. **그러나 칸 평균 표는 q가 높을수록 오히려 커요**(15~40mm: 0.19 → 0.29). 최상위 칸에서 꺾이는 건 40mm 초과(0.56 → 0.51)뿐 | 🔶 부분(판정 기준으로는 ✅이지만 평균과 순위가 반대 → 전형적 행에서는 덜 오르고, 큰 사상의 고수위 행은 크게 오름) | q × 강우의 잔차: 상위 5사상 제외 **+2.9%**, 사상 일관성 91% → 이번 시리즈에서 가장 나은 잔차 신호. 모델링 1순위 시험 후보 |
| A11 이력 | 같은 q 칸에서 상승 중이 하강 중보다 큰 관측소 **100%**(차이 중앙값 0.18m) | ✅ 채택 | dwl_1h 부호가 이미 담음. 새 변수 불필요 |
| A13·C6 도시화 | ① 40mm 초과: 도시 하 0.63 > 상 0.35m(기존 결과 재현) ② 중간 강우(5~40mm)에서 **면적 3분위 모두** 도시 상위의 ρ(r0_1)가 크고 ρ(r6_24)가 작음 ④ 사상별 78% | ✅ 채택(중간 강우), 극한 비에서는 반대 → **부딪힘 해소**: md 주의사항대로 극한호우에서 도시 효과가 역전·약화 | 도시화 × 1h 강우 상호작용. 도시화에 단조 제약 금지 |
| A7 면적 | 타깃÷강우 중앙값 소 0.0040 / 중 0.0052 / 대 0.0046 | ❌ 기각(비유량 감소 안 보임) | – |
| A14 산림 | 면적·경사 통제 부분 순위상관 −0.19 | 🔶 부분(방향은 가설대로, 약함) | 제약 없음 |
| B9 단조 방향 | 강우·레이더 7개: 관측소 일치 98~100%, 조건부 75~100%. 반대 부호는 24h 주기·고정 관측소뿐. `dwl_1h` + (조석 V자 주의), `dwl_6h`는 조석·댐 의심에서 반대, `wl_t`·가속도는 불일치 | 아래 권고 | 아래 권고 |

**단조 제약 권고(모델링 단계에서 켜고 끄기로 확인)**
- **+ 제약 후보**: `sfc_rain_1h`, `r1_6`, IMERG 1h·6h, 레이더 3종 — 단 **24h 주기·고정 관측소를 빼거나 따로 모델링**할 때
- **비조석에서만**: `dwl_1h`(조석 V자), `dwl_6h`(조석·댐 의심에서 반대)
- **제약 없음**: `wl_t`, 가속도(8권: 1h 변화 통제 시 음), `r6_24`(이 표는 +지만 9권 A2에서 강우·경과 시간 통제 시 음 → 통제 방식에 따라 방향이 바뀜), 유역 변수·도시화
- 판정 모음은 `research_findings.md`에 정리했어요.
""")
b.write("nb10_moderators_and_verdicts.ipynb")

"""3권: 수위 변화·시계열·지체시간."""
from nb_common import Book

b = Book()
b.md("""
# ⏱️ 3권 · 수위 변화와 시간 — "물은 언제, 얼마나 늦게 반응하나?"

> 시리즈: 1권 구조·품질 → 2권 타깃과 강우 → **3권 수위 변화·시계열** → 4권 관측소 유형·유역환경 → 5권 연결망 → 6권 검증 전략·오차 위치 · 별책 조석

### 한 줄 요약
**수위는 절대값이 아니라 '변화량'으로 봐야 해요. 한번 오르기 시작한 물은 1~3시간 더 오르는 경향이 있고(추세 지속), 비가 온 뒤 물이 반응하기까지는 대부분 몇 시간 안이에요. 유역이 클수록 늦어요. 이 규칙들은 일반 관측소 거의 전부에서 성립해요.**

| 장 | 질문 |
|---|---|
| 1 | 수위는 왜 변화량으로 봐야 하나 |
| 2 | 수위 변화는 얼마나 오래 이어지나 (지속성) |
| 3 | 지금 오르는 중이면 앞으로도 오르나 (추세 지속) |
| 4 | 비가 온 뒤 몇 시간 만에 물이 반응하나 (지체시간) |
| 5 | 호우가 진행될수록 타깃은 어떻게 변하나 |
| 6 | 누수 점검과 정리 |

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **dwl_k** | 지금 수위 − k시간 전 수위 | `dwl_3h = +0.5` → 3시간 새 50cm 오름 |
| **자기상관** | 신호를 자기 자신의 몇 시간 뒤와 비교한 상관 | 1시간 뒤 상관 0.7 → 지금 오르면 1시간 뒤에도 오를 가능성이 큼 |
| **교차상관** | 두 신호(비, 수위 변화)를 시차를 두고 비교한 상관 | 비(t−2)와 수위 변화(t)의 상관이 가장 크면 "2시간 늦게 반응" |
| **지체시간** | 비가 온 뒤 수위가 가장 크게 반응하기까지 걸리는 시간 | 소하천 1시간, 대하천 반나절 |
| **수문곡선** | 시간에 따른 수위(또는 유량) 곡선 | 비 → 상승 → 첨두 → 하강 |
""")
b.setup()
b.code(r'''
cols = ["station_id", "station_code", "hour", "timestamp", "event_block", "pos_in_block", "wl_t", TARGET] + DWL + IMERG[:2] + SFC[:2] + ENV
tr = add_flags(pd.read_parquet(CACHE / "train_sorted.parquet", columns=cols))
key = keyed(tr)
gen = tr[tr.group == "일반"]                       # 조석·약한 조석·24h 주기·고정 관측소를 모두 뺀 '일반' 관측소
GEN_IDS = gen.station_id.unique()
print(f"train {len(tr):,}행 · 일반 관측소 {len(GEN_IDS)}곳")
''')

b.md("""
## 1. 수위는 왜 변화량으로 봐야 하나

관측소마다 **수위의 0점(기준면)** 이 달라요. 어떤 곳은 해발 높이, 어떤 곳은 그 자리 수위표 눈금이에요.

**예시**: A 관측소의 평소 수위가 1.2m, B 관측소는 85.0m라고 해 봐요. B가 물이 더 많은 게 아니라 **0점이 다를 뿐**이에요. 그래서 관측소끼리 절대 수위를 비교하면 안 되고, **"얼마나 변했나"** 만 비교할 수 있어요.

465곳 전부의 평소 수위(중앙값)와 변동폭을 그려 봐요.
""")
b.code(r'''
st = tr.groupby("station_id").agg(wl_med=("wl_t", "median"), dwl_sd=("dwl_1h", "std"))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].hist(st.wl_med, bins=80, color=C_ALL); ax[0].set_xlabel("관측소별 수위 중앙값 (m)"); ax[0].set_ylabel("관측소 수")
ax[0].set_title("평소 수위: −5m ~ 140m (0점이 제각각)")
ax[1].scatter(st.wl_med, st.dwl_sd, s=8, color=C_NT, alpha=.6); ax[1].set_yscale("log")
ax[1].set_xlabel("수위 중앙값 (m)"); ax[1].set_ylabel("dwl_1h 표준편차 (m, log)"); ax[1].set_title("평소 수위와 움직임은 무관")
e = np.logspace(np.log10(.015), 1, 20); bins = np.r_[-e[::-1], [-.005, .005], e]
for c, col in zip(["dwl_1h", "dwl_3h", "dwl_6h", "dwl_24h"], CAT):
    h, ee = np.histogram(tr[c].dropna(), bins=bins); ax[2].step(ee[1:], h, where="pre", color=col, label=c)
symlog_x(ax[2], .01); ax[2].set_yscale("log"); ax[2].legend(fontsize=8); ax[2].set_xlabel("수위 변화 (m, symlog)")
ax[2].set_title("변화량: 0 중심, 창이 길수록 넓다")
fig.tight_layout(); plt.show()
print(f"수위 중앙값과 움직임(표준편차)의 순위 상관: {st.wl_med.corr(st.dwl_sd, method='spearman'):.2f}")
''')
b.md("""
- 평소 수위는 관측소마다 −5m에서 140m까지 흩어져 있지만, **이 값은 물이 얼마나 움직이는지와 거의 무관**해요.
- 변화량은 0을 중심으로 모이고, 창이 길수록(1h → 24h) 넓게 퍼져요.

→ 입력으로는 **`dwl_k = wl_t − wl_t−k`** 를 써요. 2권에서 본 것처럼 절대 수위 `wl_t`는 타깃과 상관이 0이었지만 `dwl_1h`는 가장 강한 변수였어요.

### 1-1. 수문곡선 — 비가 오면 물은 이렇게 반응해요
일반 관측소 중 **작은 유역·중간 유역·큰 유역**에서 한 곳씩, 가장 긴 호우 사상에서 가장 크게 오른 곳을 골라 봐요. 비 막대는 위에서 아래로 내려오게 뒤집어 그렸어요.
""")
b.code(r'''
blk = gen.groupby("event_block").size().idxmax()
bb = gen[gen.event_block == blk]
peak = bb.groupby("station_id").agg(rise=(TARGET, "max"), area=("wsarea", "first"))
peak = peak[peak.rise > .5]
peak["t"] = pd.qcut(peak.area.rank(method="first"), 3, labels=["작은", "중간", "큰"])
pick = peak.sort_values("rise").groupby("t").tail(1).sort_values("area")
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for a, (sid, r) in zip(ax, pick.iterrows()):
    s = bb[bb.station_id == sid].sort_values("hour")
    h = s.pos_in_block
    a.plot(h, s.wl_t - s.wl_t.iloc[0], color=C_NT, label="수위 − 시작값")
    a2 = a.twinx(); a2.bar(h, s.sfc_rain_1h, color=C_RAIN, alpha=.6, width=.9); a2.invert_yaxis(); a2.grid(False)
    a2.set_ylim(s.sfc_rain_1h.max() * 3 + 1, 0); a2.set_ylabel("비 (mm)", color=C_RAIN)
    a.set_ylabel("m"); a.set_title(f"{sid} · {r.t} 유역 (상류면적 {r.area:,.0f})", fontsize=9)
ax[-1].set_xlabel(f"사상 #{blk} 시작 후 시간 (h)")
fig.suptitle("작은 유역은 빨리 솟고 빨리 빠지고, 큰 유역은 천천히 오래 오른다", x=.02, ha="left"); fig.tight_layout(); plt.show()
''')
b.md("""
- **작은 유역**: 비가 오면 몇 시간 안에 솟았다가 금방 빠져요.
- **큰 유역**: 상류 곳곳에서 모인 물이 도착할 때까지 **하루 가까이** 천천히 올라요.

세 곳만 본 거라서, 4장에서 **일반 관측소 전부**로 "유역이 클수록 늦다"를 확인할 거예요.
""")

b.md("""
## 2. 수위 변화는 얼마나 오래 이어지나 (지속성)

지금 1시간 새 올랐다면 다음 1시간에도 오를까요? `dwl_1h`의 **자기상관**을 1·2·3·6시간 시차로 **일반 관측소마다** 계산해요.
""")
b.code(r'''
x = tr["dwl_1h"].to_numpy(float)
A = pd.DataFrame({L: station_corr(tr, x, shift_lookup(tr, key, "dwl_1h", -L)) for L in range(1, 13)})
Ag = A.loc[A.index.intersection(GEN_IDS)]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].fill_between(Ag.columns, Ag.quantile(.1), Ag.quantile(.9), color=C_NT, alpha=.2, label="일반 관측소 10~90%")
ax[0].plot(Ag.columns, Ag.median(), "-o", ms=3, color=C_NT, label="중앙값")
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xlabel("시차 (h)"); ax[0].set_ylabel("dwl_1h 자기상관"); ax[0].legend(fontsize=8)
ax[0].set_title("1~3시간이면 대부분 사라진다")
for L, col in [(1, C_NT), (3, C_GREEN), (6, C_RAIN)]:
    ax[1].hist(Ag[L].dropna(), bins=np.linspace(-.4, 1, 57), histtype="step", lw=1.6, color=col, label=f"{L}h (중앙값 {Ag[L].median():.2f})")
ax[1].axvline(0, color="k", lw=.5); ax[1].set_xlabel("관측소별 자기상관"); ax[1].set_ylabel("관측소 수"); ax[1].legend(fontsize=8)
ax[1].set_title(f"일반 관측소 {len(Ag)}곳 각각"); fig.tight_layout(); plt.show()
print(f"1시간 자기상관 > 0 인 일반 관측소: {(Ag[1] > 0).mean():.1%}")
''')
b.md("""
- **거의 모든** 일반 관측소에서 1시간 자기상관이 양수예요. 지금 오르면 다음 시간에도 오를 가능성이 커요.
- 3시간쯤이면 약해지고 6시간이면 거의 0이에요.

→ 수위 변화의 "기억"은 **1~3시간** 정도예요. 그래서 `dwl_1h`, `dwl_3h`가 강한 입력이 돼요.

## 3. 지금 오르는 중이면 앞으로도 오르나 (추세 지속)

2장은 "1시간 변화끼리"의 관계였어요. 이번엔 진짜 관심사인 **타깃(앞으로 6시간 최대 상승)** 과의 관계를 봐요.
전체와 비조석을 나란히 그려요. 별책에서 본 것처럼, 전체의 **V자 왼쪽 가지("빠진 뒤 반등")는 조석이 만든 착시**예요.
""")
b.code(r'''
nt = tr[tr.nontidal]
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, c in zip(ax, ["dwl_1h", "dwl_3h"]):
    for g, d, col in [("전체", tr, C_ALL), ("비조석", nt, C_NT)]:
        dd = d.dropna(subset=[c])
        edges = np.unique(np.quantile(dd[c], np.linspace(0, 1, 41)))
        k = pd.cut(dd[c], edges, include_lowest=True)
        m = dd.groupby(k, observed=True).agg(x=(c, "median"), y=(TARGET, "mean"))
        a.plot(m.x, m.y, "-o", ms=2.5, color=col, label=g)
    symlog_x(a, .01); a.axvline(0, color=C_ALL, lw=.6); a.set_xlabel(f"{c} (m, symlog)")
ax[0].set_ylabel("타깃 평균 (m)"); ax[0].legend()
fig.suptitle("비조석: 오르던 물은 계속 오르고, 빠지던 물은 반등하지 않는다", x=.02, ha="left"); fig.tight_layout(); plt.show()
''')
b.md("""
비조석(파랑)은 **오른쪽으로 갈수록 올라가는 한쪽 가지**예요. 최근에 많이 올랐을수록 앞으로도 많이 올라요.

이게 **일반 관측소 전부**에서 성립하는지 확인해요. 관측소마다 "`dwl_3h` 순위 ↔ 타깃 순위" 상관을 구해요.
""")
b.code(r'''
rho = {}
for c in ["dwl_1h", "dwl_3h"]:
    d = gen.dropna(subset=[c])
    rho[c] = station_corr(d, d[c].groupby(d.station_id).rank(pct=True).to_numpy(), d[TARGET].groupby(d.station_id).rank(pct=True).to_numpy())
rho = pd.DataFrame(rho)
fig, ax = plt.subplots(figsize=(9, 3.2))
for c, col in zip(rho.columns, [C_NT, C_GREEN]):
    ax.hist(rho[c].dropna(), bins=np.linspace(-.3, .9, 61), histtype="step", lw=1.6, color=col, label=f"{c} (중앙값 {rho[c].median():.2f})")
ax.axvline(0, color="k", lw=.5); ax.set_xlabel("관측소별 수위 변화–타깃 순위 상관"); ax.set_ylabel("관측소 수"); ax.legend()
ax.set_title(f"일반 관측소 {rho.dropna().shape[0]}곳: 거의 전부 양수"); plt.show()
print("상관 > 0 인 비율:", (rho > 0).mean().round(3).to_dict())
''')
b.md("""
일반 관측소 거의 전부에서 **"최근 올랐으면 앞으로도 오른다"** 가 성립해요.

## 4. 비가 온 뒤 몇 시간 만에 물이 반응하나 (지체시간)

### 4-1. 방법 — 교차상관
관측소마다 **"L시간 전의 비"와 "지금의 수위 변화(dwl_1h)"** 의 상관을 L = 0~24시간으로 구해요. 상관이 가장 큰 L이 그 관측소의 **지체시간**이에요.

**예시**: L = 2일 때 상관이 가장 크면 "비가 오고 2시간 뒤에 물이 가장 많이 오른다"는 뜻이에요.

모든 값은 **과거의 비**만 써요(비를 L시간 **전**에서 가져와요).
""")
b.code(r'''
dwl = tr["dwl_1h"].to_numpy(float)
XC = {}
for src in ["sfc_rain_1h", "imerg_rain_1h"]:
    XC[src] = pd.DataFrame({L: station_corr(tr, shift_lookup(tr, key, src, -L) if L else tr[src].to_numpy(float), dwl) for L in range(0, 25)})
lag = pd.DataFrame({src: C.loc[C.index.intersection(GEN_IDS)].idxmax(axis=1) for src, C in XC.items()})
area = tr.groupby("station_id").wsarea.first()

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for src, col, lab in [("sfc_rain_1h", C_RAIN, "지상"), ("imerg_rain_1h", C_NT, "IMERG")]:
    C = XC[src].loc[XC[src].index.intersection(GEN_IDS)]
    ax[0].plot(C.columns, C.median(), "-o", ms=3, color=col, label=f"{lab} (중앙값)")
    ax[1].hist(lag[src].clip(upper=12), bins=np.arange(-.5, 13.5), histtype="step", lw=1.8, color=col, label=lab)
ax[0].set_xlabel("비를 몇 시간 전에서 가져왔나 (L)"); ax[0].set_ylabel("corr(비(t−L), dwl_1h(t))"); ax[0].legend(); ax[0].set_title("반응은 몇 시간 안에 최대")
ax[1].set_xlabel("관측소별 지체시간 (h, 12 이상은 12)"); ax[1].set_ylabel("관측소 수"); ax[1].legend(); ax[1].set_title(f"일반 관측소 {len(lag)}곳 각각")
ax[2].scatter(area.reindex(lag.index), lag.sfc_rain_1h + np.random.default_rng(0).uniform(-.2, .2, len(lag)), s=8, alpha=.6, color=C_RAIN)
ax[2].set_xscale("log"); ax[2].set_xlabel("상류 유역면적 (log)"); ax[2].set_ylabel("지체시간 (h, 지상 기준)"); ax[2].set_title("유역이 클수록 늦다")
fig.tight_layout(); plt.show()
for src in lag:
    r = lag[src].corr(area.reindex(lag.index), method="spearman")
    print(f"{src}: 지체시간 중앙값 {lag[src].median():.0f}h · 6시간 이내 {(lag[src] <= 6).mean():.0%} · 유역면적과 순위 상관 {r:.2f}")
''')
b.md("""
- 지체시간 중앙값은 **지상 기준 1시간, IMERG 기준 3시간** 안팎이에요. **대부분 관측소가 6시간 안에** 반응해요.
- 오른쪽 그림처럼 **유역이 클수록 늦어요**(순위 상관 약 0.4~0.5).

→ 예측창이 6시간이니까, **"이미 내렸지만 아직 하천에 도착하지 않은 비"** 가 가장 중요한 신호예요. 과거 1~6시간 강우, 그리고 **유역면적 × 강우** 같은 교차 변수가 유용해요.

### 4-2. 유역 크기별로 나눠 보면 (일반 관측소 전부)
위 산점도를 유역면적 3구간으로 나눠, 구간마다 교차상관 곡선의 **중앙값과 흩어짐**을 봐요.
""")
b.code(r'''
C = XC["sfc_rain_1h"].loc[XC["sfc_rain_1h"].index.intersection(GEN_IDS)]
tert = pd.qcut(area.reindex(C.index).rank(method="first"), 3, labels=["작은 유역", "중간 유역", "큰 유역"])
fig, ax = plt.subplots(figsize=(9, 3.8))
for t, col in zip(tert.cat.categories, [C_RAIN, C_GREEN, C_PURPLE]):
    Ct = C[tert == t]
    Cn = Ct.div(Ct.max(axis=1), axis=0)                        # 관측소마다 최댓값을 1로 맞춰 모양만 비교
    ax.fill_between(Cn.columns, Cn.quantile(.25), Cn.quantile(.75), color=col, alpha=.15)
    ax.plot(Cn.columns, Cn.median(), "-o", ms=3, color=col, label=f"{t} ({len(Ct)}곳, 지체 중앙값 {Ct.idxmax(axis=1).median():.0f}h)")
ax.set_xlabel("L (h)"); ax.set_ylabel("교차상관 (관측소별 최댓값 = 1)"); ax.legend(fontsize=8)
ax.set_title("큰 유역일수록 반응 곡선이 늦게, 넓게 퍼진다 (띠 = 25~75%)"); plt.show()
''')
b.md("""
세 구간 모두 곡선 모양이 일관돼요. 작은 유역은 **짧고 뾰족하게**, 큰 유역은 **늦고 넓게** 반응해요. 몇 곳의 우연이 아니라 **유역 크기에 따른 규칙**이에요.

### 4-3. (참고) 강우 원천끼리의 시차 — 수치만
위성(IMERG)과 지상강수를 시차 k를 두고 비교한 상관이에요. **해석은 싣지 않고 수치만** 적어요(보고서 방침). 이 계산은 train 안에서 지상강수의 **뒤 시각 값**을 가져오므로 **분석용으로만** 쓰고, 모델 입력으로는 쓰지 않아요.
""")
b.code(r'''
d = tr
xi = d["imerg_rain_1h"].to_numpy(float)
curve = {k: station_corr(d, xi, shift_lookup(d, key, "sfc_rain_1h", k)).median() for k in range(-3, 4)}
print("corr(IMERG(t), 지상(t+k)) — 관측소 중앙값:")
print(pd.Series(curve).round(3).to_string())
''')

b.md("""
## 5. 호우가 진행될수록 타깃은 어떻게 변하나

사상(호우 기간)이 시작된 지 몇 시간째인지(`pos_in_block`)에 따라 타깃이 달라지는지 봐요. 비조석 행만 써요.

⚠️ **누수 주의**: "사상 시작 후 경과 시간"은 **과거만 보고** 알 수 있어요(사상은 앞쪽 공백으로 시작이 정해져요). 하지만 **"사상이 언제 끝나는지", "사상 전체 길이"** 는 미래 정보라서 쓰면 안 돼요.
""")
b.code(r'''
p = nt.pos_in_block.clip(upper=48)
m = nt.groupby(p)[TARGET].agg(["mean", "std", "size"]); m = m[m["size"] >= 500]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(m.index, m["mean"], color=C_NT, label="평균"); ax[0].plot(m.index, m["std"], ":", color=C_NT, label="표준편차")
ax[0].set_xlabel("사상 시작 후 경과 시간 (h, 48 이상은 48)"); ax[0].set_ylabel("타깃 (m)"); ax[0].legend(); ax[0].set_title("비조석 전체: 시간이 갈수록 커진다")

# 사상 전부에서 확인: 길이 12시간 이상인 사상마다 '앞 3시간' vs '6~12시간째' 타깃 평균
e = nt.groupby("event_block").pos_in_block.max()
long_ev = e[e >= 12].index
cmp = nt[nt.event_block.isin(long_ev)].assign(ph=lambda z: np.select([z.pos_in_block < 3, z.pos_in_block.between(6, 12)], ["초반", "중반"], "기타"))
cmp = cmp[cmp.ph != "기타"].groupby(["event_block", "ph"])[TARGET].mean().unstack()
ax[1].scatter(cmp["초반"], cmp["중반"], s=14, color=C_NT)
lim = [min(cmp.min().min(), 0), cmp.max().max()]; ax[1].plot(lim, lim, ":", color=C_ALL)
ax[1].set_xlabel("사상 초반(0~2h) 타깃 평균 (m)"); ax[1].set_ylabel("사상 중반(6~12h) 타깃 평균 (m)")
ax[1].set_title(f"12시간 이상 사상 {len(cmp)}개 각각"); fig.tight_layout(); plt.show()
print(f"0h 평균 {m['mean'].iloc[0]:.3f} → 24h 평균 {m['mean'].get(24, np.nan):.3f} m")
print(f"중반이 초반보다 큰 사상: {(cmp['중반'] > cmp['초반']).mean():.0%} ({(cmp['중반'] > cmp['초반']).sum()}/{len(cmp)})")
''')
b.md("""
- 사상이 진행될수록 타깃의 평균과 흩어짐이 모두 커져요. 비가 쌓이고, 물이 불어나는 국면에 들어가니까요.
- 사상 하나하나로 봐도 **대부분** 중반이 초반보다 커요. 하지만 반대인 사상도 있어요(초반에 비가 몰렸다가 그친 경우 등). 그래서 경과 시간 **혼자**로는 부족하고, **사상 시작 후 누적 강우** 같은 변수와 함께 써야 해요.

## 6. 누수 점검과 정리

### 누수 점검표

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 1장 절대 수위·변화량 | 입력값(과거) | 누수 없음 |
| 1-1 수문곡선 예시 | 사상 전체를 그림으로만 봄 | 설명용 |
| 2장 자기상관 | 관측소의 train 기간 전체 수위 변화 | 관측소 특성. 모델에 넣을 땐 **학습 묶음 안에서** 계산 |
| 3장 수위 변화 × 타깃 | 입력은 과거 차분, 타깃은 결과 | 누수 없음 |
| 4-1·4-2 지체시간 | **과거** 강우(L시간 전)와 현재 변화 | 관측소 특성. 학습 묶음 안에서 계산 |
| 4-3 원천 간 시차 | 지상강수의 **뒤 시각** 값을 씀 | 분석용 수치만. 모델 입력 금지 |
| 5장 사상 경과 시간 | 사상 **시작**은 과거로 결정됨 | 사용 가능. 단 사상 **길이·끝**은 미래라 금지 |

### 정리

| 발견 | 전체에 해당? | 모델링 시사점 |
|---|---|---|
| 절대 수위는 관측소마다 0점이 달라 무의미 | 465곳 | 변화량 `dwl_k` 사용 |
| 수위 변화의 기억은 1~3시간 | ✅ 일반 관측소 거의 전부 (1h 자기상관 > 0) | `dwl_1h`, `dwl_3h` 핵심 입력 |
| 최근 올랐으면 앞으로도 오름 (추세 지속) | ✅ 일반 관측소 거의 전부 | 상승 속도·가속도 변수 |
| 지체시간 대부분 6시간 이내, 유역이 클수록 늦음 | ✅ 일반 관측소 전부의 분포, 유역 3구간 모두 | 과거 1~6h 강우, 유역면적 × 강우 교차, 원천별 시간창 |
| 사상이 진행될수록 타깃 평균·분산 증가 | 대부분 사상 (예외 있음) | 사상 경과 시간 + 사상 누적 강우 |
""")
b.write("nb3_water_level_and_time.ipynb")

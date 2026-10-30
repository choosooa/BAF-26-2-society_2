"""1권 노트북(데이터 구조·품질) 생성기. 실행: python src/nb_build_01.py → eda/nb1_data_structure_quality.ipynb"""
import nbformat as nbf
from pathlib import Path

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 📦 1권 · 데이터 구조와 품질 — "이 데이터, 믿고 써도 되나요?"

> 시리즈: **1권 구조·품질** → 2권 타깃과 강우 → 3권 수위 변화·시계열 → 4권 관측소 유형·유역환경 → 5권 연결망 → 6권 검증 전략·오차 위치
> (조석 이야기는 별책 `tidal_analysis.ipynb`)

### 한 줄 요약
**데이터는 깨끗해요. 빠진 값·중복·계산 오류가 없어요. 다만 "한 행"이 무엇인지, 여러 관측소가 같은 비·같은 유역 값을 나눠 쓴다는 점, 그리고 수상한 관측소 몇 곳을 알고 시작해야 해요.**

### 이 노트북에서 하는 일

| 장 | 질문 | 쉽게 말하면 |
|---|---|---|
| 1 | 한 행은 무엇인가 | "한 줄에 뭐가 들어 있고, 뭘 맞혀야 하지?" |
| 2 | 데이터는 어떤 덩어리로 되어 있나 | "관측소 465곳 × 호우 사건 152개 × 시간" |
| 3 | 값은 어디서 왔나 | "비·유역 값은 여러 관측소가 나눠 써요" |
| 4 | 품질 5가지 점검 | 빠진 값, 중복, 말이 안 되는 값, 계산 오류, 튀는 값 |
| 5 | 누수 점검 | "미래 정보가 섞일 수 있는 길은 어디인가" |
| 6 | 정리 | 그래서 전처리는 어떻게 하나 |

이번에도 **예시 몇 개로 끝내지 않고**, 점검마다 **관측소 전체·사상 전체**에서 같은 결과가 나오는지 확인해요.

### 용어 사전

| 용어 | 뜻 | 예시 |
|---|---|---|
| **행** | (관측소, 시각) 하나 | "ST0100의 어느 날 14시" |
| **lag 수위** (`wl_t_minus_3h`) | k시간 **전** 수위 | 지금 14시면 11시 수위 |
| **누적 강우** (`_6h`) | 지금까지 k시간 동안 온 비의 합 | 9~14시에 온 비 |
| **사상** (`event_block`) | 시간이 끊기지 않고 이어진 한 덩어리 = 한 번의 호우 기간 | train 122개, test 30개 |
| **구간 척도** | 0점이 임의로 정해진 값. 차이만 의미가 있어요 | 섭씨 온도, 관측소마다 기준면이 다른 수위 |
| **누수** | 예측할 때는 알 수 없는 정보(미래 등)가 학습·평가에 섞이는 것 | 답안지를 슬쩍 보고 시험 치기 |
""")

md("## 0. 준비")
code(r'''
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = ["DejaVu Sans", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})

EDA = Path.cwd() if (Path.cwd() / "cache").exists() else Path.cwd().parent
CACHE, DATA = EDA / "cache", EDA.parent / "data"
TARGET = "target_maxrise_6h"
WL = ["wl_t", "wl_t_minus_1h", "wl_t_minus_3h", "wl_t_minus_6h", "wl_t_minus_12h", "wl_t_minus_24h"]
IMERG = ["imerg_rain_1h", "imerg_rain_6h", "imerg_rain_24h"]
SFC = ["sfc_rain_1h", "sfc_rain_6h", "sfc_rain_24h"]
RADAR = ["radar_reflectivity_mean", "radar_reflectivity_p95", "radar_reflectivity_ge20_fraction"]
ENV = ["cat_area", "wsarea", "mean_elevation", "mean_slope", "urban_ratio", "forest_ratio", "stream_order"]

# 색 약속: 파랑 = train, 주황 = test / 문제 있는 것, 회색 = 기준선
C_TR, C_TE, C_GRAY, C_OK, C_RAIN = "#2a78d6", "#eb6834", "#9a9890", "#1baf7a", "#eda100"

# train_sorted / test_sorted: 원본을 (관측소, 시각) 순으로 정렬하고
#   station_code(관측소 번호 0~464), hour(합성 시각을 시간 단위 정수로), event_block(사상 번호)을 붙인 캐시 (src/s01_integrity.py)
tr = pd.read_parquet(CACHE / "train_sorted.parquet")
te = pd.read_parquet(CACHE / "test_sorted.parquet")
for d in (tr, te):
    d["station_id"] = d["station_id"].astype(str)
print(f"train {tr.shape} · test {te.shape}")
''')
code(r'''
def keyed(d):
    """(관측소, 시각) → 정수 키 하나. 예: 관측소 12, 시각 345 → 12_000_345"""
    return d["station_code"].to_numpy(np.int64) * 1_000_000 + d["hour"].to_numpy(np.int64)

def shift_lookup(d, key, col, dh):
    """같은 관측소의 (지금 + dh)시간 col 값. 그 행이 없으면 NaN."""
    return pd.Series(d[col].to_numpy(np.float64), index=key).reindex(key + dh).to_numpy()

key_tr, key_te = keyed(tr), keyed(te)
''')

# ---------------- 1
md("""
## 1. 한 행은 무엇인가

대회 데이터의 한 행은 **"어떤 관측소의 어떤 시각"** 하나예요. 실제 한 줄을 뽑아서 칸마다 뜻을 붙여 볼게요.
""")
code(r'''
dic = pd.read_csv(DATA / "data_dictionary.csv")[["column_name", "role", "description", "unit_range"]].set_index("column_name")
ex = tr[(tr.imerg_rain_6h > 5)].iloc[1000]           # 비가 좀 오고 있는 아무 행 하나
one = pd.DataFrame({"값": ex[dic.index.intersection(ex.index)]}).join(dic)
one
''')
md("""
**읽는 법**
- `wl_t`, `wl_t_minus_*`: 지금과 1·3·6·12·24시간 **전** 수위 → **과거**
- `*_rain_1h/6h/24h`: 지금까지 1·6·24시간 동안 온 비의 **누적** → **과거**
- `radar_*`: **지금** 레이더 영상 요약
- 유역환경 7개: 시간에 따라 변하지 않는 관측소 특성
- `target_maxrise_6h`: **앞으로** 6시간 안에 수위가 최대 얼마나 더 오르나 → **미래** (train에만 있음)

즉 **입력은 전부 '지금까지', 타깃만 '앞으로'** 예요. 그림으로 보면 이래요.
""")
code(r'''
fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 2]})
ax = axes[0]
rows = [("수위 (지금·1·3·6·12·24h 전)", "pt", None), ("강우 24시간 누적", (-23, 0), C_RAIN),
        ("강우 6시간 누적", (-5, 0), C_RAIN), ("강우 1시간 · 레이더", (0, 0), C_RAIN),
        ("타깃: 앞으로 6시간 최고 − 지금", (1, 6), C_TE)]
for i, (lab, span, col) in enumerate(rows):
    y = len(rows) - i
    if span == "pt":
        for h in (-24, -12, -6, -3, -1, 0):
            ax.plot(h, y, "o", color=C_TR, ms=8)
    else:
        a, b = span
        ax.add_patch(Rectangle((a - .45, y - .3), b - a + .9, .6, color=col, alpha=.75))
    ax.text(-25.5, y, lab, ha="right", va="center", fontsize=9)
ax.axvline(0, color="k", lw=1, ls="--"); ax.text(0, len(rows) + .8, "지금 (t)", ha="center")
ax.axvspan(-25, 0.5, color=C_TR, alpha=.04); ax.axvspan(0.5, 8, color=C_TE, alpha=.06)
ax.text(-12, .45, "← 과거: 입력", ha="center", color=C_TR); ax.text(4, .45, "미래: 타깃 →", ha="center", color=C_TE)
ax.set_xlim(-25, 8); ax.set_ylim(.2, len(rows) + 1.2); ax.set_yticks([]); ax.set_xlabel("지금으로부터 시간 (h)")
ax.set_title("한 행이 보는 시간창")

ax = axes[1]   # 이웃한 세 행의 타깃 구간이 겹친다
for i, t0 in enumerate([0, 1, 2]):
    y = 3 - i
    ax.add_patch(Rectangle((t0 + .55, y - .3), 5.9, .6, color=C_TE, alpha=.3 + .2 * i))
    ax.plot(t0, y, "o", color=C_TR, ms=8); ax.text(-1.2, y, f"행 t{'+' + str(t0) if t0 else ''}", ha="right", va="center")
ax.add_patch(Rectangle((2.55, .6), 4.9, 2.8, fill=False, ec="k", ls=":", lw=1.2))
ax.text(5, .25, "이웃한 행끼리 타깃 구간이 최대 5시간 겹쳐요", ha="center")
ax.set_xlim(-4, 9); ax.set_ylim(0, 3.6); ax.set_yticks([]); ax.set_xlabel("시간 (h)")
fig.tight_layout(); plt.show()
''')
md("""
아래 그림이 중요해요. **바로 옆 시각의 행끼리는 타깃이 보는 미래가 거의 같아요.** 14시 행의 타깃(15~20시)과 15시 행의 타깃(16~21시)은 5시간이 겹쳐요.
그래서 행을 무작위로 섞어 학습용과 검증용으로 나누면, 검증 행의 답이 사실상 학습 쪽에 들어가 버려요(**누수**). 이 문제는 5장과 6권에서 다시 다뤄요.
""")

# ---------------- 2
md("""
## 2. 데이터는 어떤 덩어리로 되어 있나

### 2-1. 규모
""")
code(r'''
print(f"train {len(tr):,}행 · test {len(te):,}행 · 합 {len(tr) + len(te):,}행")
print(f"관측소: train {tr.station_id.nunique()}곳 · test {te.station_id.nunique()}곳 · 같은 집합? {set(tr.station_id) == set(te.station_id)}")
print(f"NaN: train {int(tr.isna().sum().sum())}개 · test {int(te[[c for c in te.columns if c != 'n_future_rows']].isna().sum().sum())}개")
''')
md("""
### 2-2. 사상(호우 기간)을 복원하기

`timestamp`는 **가짜(합성) 시각**이에요. 실제 날짜는 지워졌지만, **같은 호우 안에서 시간 순서와 간격은 그대로**예요(데이터 사전 설명).
그래서 모든 시각을 줄 세웠을 때 **1시간보다 크게 끊기는 곳**이 호우와 호우 사이 경계예요. 이렇게 나눈 덩어리를 **사상(event_block)** 이라고 불러요.

- 규칙: 시각을 정렬 → 앞 시각과 1시간 넘게 차이 나거나, train↔test가 바뀌면 새 사상
- 아래 셀은 이 규칙으로 처음부터 다시 나눠 보고, 캐시에 저장된 `event_block`과 같은지 확인해요.
""")
code(r'''
# train/test의 서로 다른 시각을 모아 줄 세운다
h = pd.DataFrame({"hour": np.r_[np.sort(tr.hour.unique()), np.sort(te.hour.unique())],
                  "split": ["train"] * tr.hour.nunique() + ["test"] * te.hour.nunique()}).sort_values("hour", ignore_index=True)
new_block = (h.hour.diff() != 1) | (h.split != h.split.shift())      # 1시간 넘게 끊기거나 train/test가 바뀌면 새 사상
h["block"] = new_block.cumsum() - 1
print(f"복원한 사상 수: {h.block.nunique()}개 (train {h[h.split == 'train'].block.nunique()} · test {h[h.split == 'test'].block.nunique()})")

m = tr[["hour", "event_block"]].drop_duplicates().merge(h, on="hour")
print(f"캐시의 event_block과 1:1 대응? {m.groupby('event_block').block.nunique().eq(1).all() and m.groupby('block').event_block.nunique().eq(1).all()}")

bl = h.groupby("block").agg(split=("split", "first"), n_hours=("hour", "size"), h0=("hour", "min"))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for s, c in [("train", C_TR), ("test", C_TE)]:
    d = bl[bl.split == s]
    ax[0].hist(d.n_hours, bins=np.arange(0, 160, 5), color=c, alpha=.6, label=f"{s} ({len(d)}개)")
ax[0].set_xlabel("사상 길이 (시간)"); ax[0].set_ylabel("사상 수"); ax[0].legend(); ax[0].set_title("대부분 짧고, 가끔 며칠짜리")
ax[1].scatter(bl.h0, bl.n_hours, c=np.where(bl.split == "train", C_TR, C_TE), s=14)
ax[1].set_xlabel("합성 시각에서의 시작 위치 (h)"); ax[1].set_ylabel("길이 (h)"); ax[1].set_title("test 사상은 시간축 전체에 흩어져 있다")
fig.tight_layout(); plt.show()
bl.groupby("split").n_hours.describe().round(1)
''')
md("""
- 사상 **152개**(train 122 · test 30), 길이 중앙값은 9~10시간, 가장 긴 건 155시간이에요.
- test 사상이 끝부분에 몰려 있지 않고 **시간축 전체에 흩어져** 있어요. 그러니까 "과거로 학습해 미래를 맞히는" 구조가 아니라 **"다른 호우 사건을 맞히는"** 구조예요.

### 2-3. 사상마다 관측소가 다 들어 있나? (152개 전부)
""")
code(r'''
cov = pd.concat([tr[["event_block", "station_id"]].assign(split="train"), te[["event_block", "station_id"]].assign(split="test")])
per_block = cov.groupby(["event_block", "split"]).station_id.nunique().reset_index()
per_block["share"] = per_block.station_id / 465

fig, ax = plt.subplots(figsize=(10, 3.2))
for s, c in [("train", C_TR), ("test", C_TE)]:
    d = per_block[per_block.split == s]
    ax.hist(d.share * 100, bins=np.linspace(80, 100, 41), color=c, alpha=.6, label=f"{s} 사상 {len(d)}개")
ax.set_xlabel("사상에 등장한 관측소 비율 (%)"); ax.set_ylabel("사상 수"); ax.legend()
ax.set_title("152개 사상 전부: 거의 모든 관측소가 등장한다"); plt.show()
print(f"사상별 관측소 비율: 최소 {per_block.share.min():.1%} · 중앙값 {per_block.share.median():.1%}")
''')
md("""
모든 사상에 관측소가 거의 다 들어 있어요. 데이터는 **관측소 465곳 × 사상 152개 × 시간**으로 된 3차원 표(패널)예요.

→ train과 test는 **관측소는 같고, 호우 사건만 다르다**. 그래서 관측소별 특성(평소 변동폭 같은 것)은 train에서 배워 test에 써먹을 수 있어요.
""")

# ---------------- 3
md("""
## 3. 값은 어디서 왔나 — 여러 관측소가 같은 값을 나눠 쓴다

입력은 원천 5개에서 와요. 그런데 원천마다 **공간 단위**가 달라요.

| 원천 | 변수 | 공간 단위 | 쉽게 말하면 |
|---|---|---|---|
| WAMIS 수위 | 수위·타깃 | **관측소(점)** | 관측소마다 고유 |
| IMERG 위성강우 | `imerg_rain_*` | 약 10km 격자 | 가까운 관측소는 같은 칸 → **같은 비** |
| 지상기상 | `sfc_rain_*` | 격자 | 마찬가지 |
| 레이더 | `radar_*` | 격자 | 마찬가지 |
| DataON 유역 | 유역환경 7개 | 표준소권역 | 같은 소권역이면 **같은 유역 값** |

정말 나눠 쓰는지 **사상 전체**에서 세 봐요.
""")
code(r'''
# (1) 유역환경: 관측소 465곳의 7개 값 조합이 몇 종류인가
env = tr.groupby("station_id")[ENV].first()
combo = env.round(4).astype(str).agg("|".join, axis=1)
sizes = combo.value_counts()
print(f"유역환경 조합: {sizes.size}종 / 관측소 {len(env)}곳 · 다른 관측소와 값을 공유하는 곳 {int(sizes[sizes > 1].sum())}곳 · 한 조합을 쓰는 최대 관측소 {sizes.max()}곳")

# (2) 강우·레이더: 사상마다, 관측소별 시계열이 몇 종류인가 (train 사상 전부)
def unique_series_ratio(d, col):
    out = []
    for b, g in d.groupby("event_block"):
        p = g.pivot_table(index="station_id", columns="hour", values=col)
        p = p.dropna(axis=1, thresh=int(.95 * len(p))).dropna()          # 거의 모든 관측소에 있는 시각만
        if p.shape[1] < 3 or len(p) < 50: continue
        n_uni = p.round(3).astype(str).agg("|".join, axis=1).nunique()
        out.append({"event_block": b, "n_station": len(p), "n_unique": n_uni, "ratio": n_uni / len(p)})
    return pd.DataFrame(out)

share = {col: unique_series_ratio(tr, col) for col in ["imerg_rain_1h", "sfc_rain_1h", "radar_reflectivity_mean"]}
fig, ax = plt.subplots(figsize=(10, 3.2))
for (col, r), c in zip(share.items(), [C_TR, C_RAIN, C_OK]):
    ax.hist(r.ratio * 100, bins=np.linspace(0, 100, 51), color=c, alpha=.6, label=f"{col} (사상 {len(r)}개)")
ax.set_xlabel("서로 다른 시계열 수 ÷ 관측소 수 (%)  ← 낮을수록 많이 공유"); ax.set_ylabel("사상 수"); ax.legend(fontsize=8)
ax.set_title("사상 전부에서: 강우는 관측소들이 나눠 쓴다"); plt.show()
pd.DataFrame({c: r.ratio.describe()[["min", "50%", "max"]] for c, r in share.items()}).round(2)
''')
md("""
- 유역환경은 465곳이 **329가지 조합**만 써요. 여러 관측소가 같은 소권역 값을 공유해요.
- IMERG는 사상마다 서로 다른 시계열이 관측소 수의 약 60% 수준이에요. 가까운 관측소들이 **같은 격자 비**를 받아요.

**그래서 뭐가 중요한가요?**
같은 비를 맞는 이웃 관측소 둘이 있다고 해 봐요. 비·유역 값이 똑같으니, 두 곳을 구별할 정보는 **수위 기록뿐**이에요.
→ 관측소마다 다른 반응(평소 변동폭, 얼마나 빨리 오르나)은 **수위 이력에서 뽑아야** 해요(4권).
""")

# ---------------- 4
md("""
## 4. 품질 5가지 점검

데이터 품질을 볼 때 흔히 쓰는 기준 5가지를 차례로 적용해요.

| 기준 | 질문 |
|---|---|
| 완전성 | 빠진 값이 있나? 빠졌다면 아무렇게나 빠졌나, 특정 상황에서 빠졌나? |
| 유일성 | 같은 행이 두 번 들어 있지 않나? |
| 유효성 | 말이 안 되는 값(음수 강우 등)은 없나? |
| 일관성 | 서로 계산으로 이어진 값들이 맞아떨어지나? |
| 이상값 | 튀는 값은 오류인가, 실제로 일어난 극단인가? |

### 4-1. 완전성 — 빠진 시각이 비와 관련 있나?

NaN은 0개예요. 그런데 **행 자체가 빠진** 경우가 있어요. 어떤 사상이 10시간짜리인데 어떤 관측소는 9행만 있는 식이죠.

걱정되는 시나리오는 이거예요: **"홍수가 나서 수위계가 고장 나거나 값이 잘렸다."** 그렇다면 비가 많이 올수록 빠진 행이 많아야 해요. 관측소×사상 조합 전부를 강우량 10칸으로 나눠 확인해요.
""")
code(r'''
blk_len = pd.concat([tr, te]).groupby("event_block").hour.nunique()           # 사상 길이 (시간)
g = tr.groupby(["event_block", "station_id"]).agg(n=("hour", "size"), rain=("imerg_rain_1h", "mean"))
g["expected"] = g.index.get_level_values(0).map(blk_len)                    # 빠짐없이 있었다면 있어야 할 행 수
g["miss"] = 1 - g.n / g.expected
g["decile"] = pd.qcut(g.rain.rank(method="first"), 10, labels=range(1, 11))
s = g.groupby("decile")[["n", "expected"]].sum(); by = 1 - s.n / s.expected
overall = 1 - g.n.sum() / g.expected.sum()

st_miss = g.groupby("station_id")[["n", "expected"]].sum().pipe(lambda t: 1 - t.n / t.expected)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].bar(by.index.astype(int), by.values * 100, color=C_TR); ax[0].axhline(overall * 100, color=C_GRAY, ls="--")
ax[0].text(10.4, overall * 100, f"전체 {overall:.1%}", ha="right", va="bottom", color=C_GRAY)
ax[0].set_xticks(range(1, 11)); ax[0].set_xlabel("관측소×사상의 평균 강우 10분위 (1 = 가장 적음)")
ax[0].set_ylabel("빠진 시각 비율 (%)"); ax[0].set_title(f"비가 많아도 더 빠지지 않는다 (n = {len(g):,})")
ax[1].hist(st_miss * 100, bins=40, color=C_TR); ax[1].set_xlabel("관측소별 빠진 시각 비율 (%)"); ax[1].set_ylabel("관측소 수")
ax[1].set_title("관측소 465곳 전부의 빠진 비율")
fig.tight_layout(); plt.show()
print(f"관측소별 빠진 비율: 중앙값 {st_miss.median():.1%} · 90% 지점 {st_miss.quantile(.9):.1%} · 최대 {st_miss.max():.1%}")
print(f"빠진 비율 10% 넘는 관측소: {(st_miss > .1).sum()}곳 / {len(st_miss)}곳")
''')
md("""
- 빠진 시각은 전체 약 2.5%이고, **비가 많은 칸이라고 더 빠지지 않아요**(오히려 들쭉날쭉해요).
- 관측소 단위로 보면 **절반 이상은 빠진 시각이 0%** 예요. 그런데 **58곳(12%)** 은 빠진 비율이 10%를 넘고, 많게는 18~22%예요. 이런 관측소는 lag 변수(몇 시간 전 수위)가 없는 행이 더 많아요. 하지만 왼쪽 그림처럼 **비가 많을 때 골라서 빠진 건 아니에요.**

→ **홍수 때 값이 잘려 나간 흔적은 없어요.** 빠진 행을 억지로 채울(대치) 필요는 없고, 이전 행을 참조할 때 "그 행이 없으면 NaN"으로 처리하면 돼요.

### 4-2. 유일성 — 같은 행이 두 번 있나?
""")
code(r'''
for name, d in [("train", tr), ("test", te)]:
    print(f"{name}: 완전 중복 행 {int(d.duplicated(subset=[c for c in d.columns if c != 'row_id']).sum())}개 · "
          f"(관측소, 시각) 중복 {int(d.duplicated(['station_id', 'hour']).sum())}개 · row_id 중복 {int(d.row_id.duplicated().sum())}개")
print("train과 test가 같은 (관측소, 시각)을 공유:", len(set(key_tr) & set(key_te)), "개")
''')
md("""
중복은 전혀 없어요. train과 test가 같은 시각을 공유하지도 않아요.

### 4-3. 유효성 — 말이 안 되는 값이 있나?

상식으로 **반드시 지켜져야 하는 규칙**을 적고, train+test 전체에서 어긴 행을 세요.
""")
code(r'''
both = pd.concat([tr.drop(columns=TARGET), te], ignore_index=True)
eps = 1e-3
rules = {
    "강우 ≥ 0": (both[IMERG + SFC] < 0).any(axis=1),
    "IMERG 1h ≤ 6h ≤ 24h": (both.imerg_rain_1h > both.imerg_rain_6h + eps) | (both.imerg_rain_6h > both.imerg_rain_24h + eps),
    "지상 1h ≤ 6h ≤ 24h": (both.sfc_rain_1h > both.sfc_rain_6h + eps) | (both.sfc_rain_6h > both.sfc_rain_24h + eps),
    "레이더 20 이상 비율 0~1": ~both.radar_reflectivity_ge20_fraction.between(0, 1),
    "레이더 평균 ≤ p95": both.radar_reflectivity_mean > both.radar_reflectivity_p95 + eps,
    "도시 + 산림 ≤ 100%": (both.urban_ratio + both.forest_ratio) > 100 + eps,
    "면적 > 0": (both.cat_area <= 0) | (both.wsarea <= 0),
}
viol = pd.DataFrame({"어긴 행": {k: int(v.sum()) for k, v in rules.items()}, "비율": {k: f"{v.mean():.2%}" for k, v in rules.items()}})
viol
''')
md("""
딱 하나, **"레이더 평균 ≤ p95(상위 5% 값)"** 이 7.5% 어겨져요. 평균이 상위 5% 값보다 클 수가 있나요?

**예시로 이해하기**: 격자 안 화소 100개 중 3개만 비구름(값 30)이고 97개는 0이라고 해 봐요.
- 평균 = 30 × 3 / 100 = **0.9**
- p95(95번째로 작은 값) = **0**
- → 평균 > p95 가 돼요. **비구름이 격자의 5% 미만만 덮으면 자연스럽게 생기는 일**이에요.

정말 그런 경우인지 어긴 행 전부를 확인해요.
""")
code(r'''
v = rules["레이더 평균 ≤ p95"]
bad = both[v]
print(f"어긴 행 중 p95 = 0 인 비율: {(bad.radar_reflectivity_p95 == 0).mean():.1%}")
print(f"어긴 행의 20 이상 화소 비율: 중앙값 {bad.radar_reflectivity_ge20_fraction.median():.3f} · 최대 {bad.radar_reflectivity_ge20_fraction.max():.3f}")
print(f"어긴 행의 레이더 평균: 중앙값 {bad.radar_reflectivity_mean.median():.2f} (전체 행 중앙값 {both.radar_reflectivity_mean.median():.2f})")
''')
md("""
어긴 행의 **99.9%가 p95 = 0**이고, 평균도 작아요. 오류가 아니라 **아주 약하게 조금 걸친 비구름**이에요.

이 밖에 알아 둘 **특성**(오류는 아님)이 두 가지 있어요.
1. **수위 범위가 −5m ~ 140m**: 관측소마다 기준면(0m의 위치)이 달라서예요. 어떤 곳은 해수면 기준, 어떤 곳은 수위표 기준이에요. → **관측소끼리 수위 절대값을 비교하면 안 되고, 변화량만 의미가 있어요.**
2. **지상강수 단위가 공식적으로 미확인**이에요. 값 모양으로 추정해 볼게요(관측소 전부에서).
""")
code(r'''
nz = both[both.sfc_rain_1h > 0]
frac = (nz.sfc_rain_1h.to_numpy() * 2) % 1                    # 0.5로 나눈 나머지 (0 또는 1에 가까우면 0.5의 배수)
is_half = pd.Series(np.isclose(frac, 0, atol=1e-4) | np.isclose(frac, 1, atol=1e-4), index=nz.index)
per_st = is_half.groupby(nz.station_id).mean()
print(f"0이 아닌 지상강수 중 0.5의 배수: 전체 {is_half.mean():.1%}")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
vals = nz.sfc_rain_1h[nz.sfc_rain_1h <= 5]
ax[0].hist(vals, bins=np.arange(0, 5.05, .1), color=C_RAIN); ax[0].set_yscale("log")
ax[0].set_xlabel("sfc_rain_1h (0 초과, 5 이하)"); ax[0].set_ylabel("행 수 (log)"); ax[0].set_title("0.5, 1.0, 1.5 … 에 몰린 계단")
ax[1].hist(per_st * 100, bins=40, color=C_RAIN); ax[1].set_xlabel("관측소별 '0.5의 배수' 비율 (%)"); ax[1].set_ylabel("관측소 수")
ax[1].set_title("대부분 관측소가 같은 계단"); fig.tight_layout(); plt.show()
print(f"관측소별 비율: 최소 {per_st.min():.0%} · 중앙값 {per_st.median():.0%} · 80% 미만인 관측소 {(per_st < .8).sum()}곳 / {len(per_st)}곳")
''')
md("""
값이 **0.5 단위 계단**에 몰려 있어요. 관측소별로 봐도 대부분(중앙값 100%)이 그래요. 다만 **69곳은 80% 미만**이라서, 모든 격자가 같은 우량계는 아닐 수 있어요. 0.5mm마다 한 번씩 기울어지는 **전도형 우량계**의 기록 모양과 같아요. 그래서 단위를 **mm로 추정**해요(공식 확인은 아님).
→ 0.5mm보다 약한 비는 지상강수에 **0으로 찍혀요.** 약한 비는 위성(IMERG)이 더 잘 잡을 수 있어요(2권).

### 4-4. 일관성 — 계산으로 이어진 값들이 맞아떨어지나?

데이터 안에는 **서로 계산 관계가 있는 열**이 있어요. 이게 맞으면 병합·가공이 정확했다는 뜻이고, **이전 행을 가져다 새 변수를 만들어도 안전**해요.
- `wl_t_minus_3h` = 3시간 전 행의 `wl_t` 여야 해요.
- `imerg_rain_6h` = 지금 포함 6개 행의 `imerg_rain_1h` 합이어야 해요.

train과 test 모두, 그리고 **관측소 465곳 각각**에서 확인해요. (이전 행이 있는 경우만 검산할 수 있어요.)
""")
code(r'''
checks = []
for name, d, k_ in [("train", tr, key_tr), ("test", te, key_te)]:
    for k in (1, 3, 6, 12, 24):                                        # lag 수위 검산
        prev = shift_lookup(d, k_, "wl_t", -k)                          # k시간 전 행의 wl_t (과거)
        ok = np.isfinite(prev)
        match = np.abs(prev[ok] - d[f"wl_t_minus_{k}h"].to_numpy()[ok]) <= 1e-3
        per_station = pd.Series(match).groupby(d.station_id.to_numpy()[ok]).mean()
        checks.append((name, f"wl_t_minus_{k}h = {k}시간 전 wl_t", ok.mean(), match.mean(), per_station.min()))
    for src in ("imerg", "sfc"):                                        # 누적 강우 검산
        for win in (6, 24):
            acc = sum(shift_lookup(d, k_, f"{src}_rain_1h", -j) for j in range(win))   # 지금 포함 과거 win개 합
            ok = np.isfinite(acc)
            tgt = d[f"{src}_rain_{win}h"].to_numpy()[ok]
            match = np.abs(acc[ok] - tgt) <= np.maximum(.05, .01 * np.abs(tgt))
            per_station = pd.Series(match).groupby(d.station_id.to_numpy()[ok]).mean()
            checks.append((name, f"{src}_rain_{win}h = 1h {win}개 합", ok.mean(), match.mean(), per_station.min()))
pd.DataFrame(checks, columns=["데이터", "검산", "검산 가능 비율", "일치율", "가장 낮은 관측소의 일치율"]).set_index(["데이터", "검산"]).round(4)
''')
md("""
**모두 100% 일치**하고, 가장 낮은 관측소도 100%예요. 병합과 파생 계산이 정확해요.
(타깃 = 앞으로 6행 최대 − 지금 수위도 100% 맞아요. 이건 별책 `tidal_analysis.ipynb` 1장에서 확인했어요. 미래 행을 쓰는 계산이라 여기서는 train에서만 했어요. 5장 참고)

### 4-5. 이상값 — 튀는 값은 오류인가, 진짜 극단인가?

#### 교과서 방법을 그냥 쓰면 생기는 문제
교과서의 이상값 규칙은 두 가지예요.
- **Tukey 울타리**: 사분위 범위(IQR = Q3 − Q1)의 1.5배를 넘게 벗어나면 이상값
- **강건 z**: 중앙값에서 MAD(중앙값 절대 편차)의 약 3.5배를 넘게 벗어나면 이상값

그런데 **값 대부분이 0에 몰린 변수**(수위 변화, 타깃, 강우)에서는 IQR이 거의 0이 돼요. 그러면 **조금만 움직여도 이상값**으로 잡혀요. 가짜 데이터로 먼저 봐요.
""")
code(r'''
rng = np.random.default_rng(0)
fake = np.where(rng.random(10000) < .7, 0, rng.exponential(.1, 10000))   # 70%는 0, 나머지는 작은 양수 (평범한 하천의 수위 변화 느낌)
q1, q3 = np.quantile(fake, [.25, .75]); iqr = q3 - q1
flag = (fake < q1 - 1.5 * iqr) | (fake > q3 + 1.5 * iqr)
fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(fake[~flag], bins=60, color=C_TR, label="정상으로 판정"); ax.hist(fake[flag], bins=60, color=C_TE, label="이상값으로 판정")
ax.set_yscale("log"); ax.set_xlabel("값"); ax.legend(); ax.set_title(f"0이 많은 분포에 Tukey를 쓰면: {flag.mean():.0%}가 '이상값' (IQR = {iqr:.3f})")
plt.show()
''')
md("""
평범한 값인데도 약 20%가 "이상값"이 돼요. 실제 데이터에서도 똑같은 일이 생겨요.
""")
code(r'''
st_tab = tr.groupby("station_id")[ENV].first()
rows = {}
for c in ["dwl_1h", "dwl_3h", "dwl_24h", TARGET] + IMERG + SFC + RADAR[:2] + ENV:
    x = (st_tab[c] if c in ENV else tr[c]).astype(float)
    logv = c in IMERG + SFC + ["cat_area", "wsarea"]                    # 치우친 변수는 log1p 척도에서 판정
    xs = np.log1p(x) if logv else x
    q1, q3 = xs.quantile([.25, .75]); iqr = q3 - q1
    med = xs.median(); mad = (xs - med).abs().median()
    rows[c] = {"단위": "관측소" if c in ENV else "행", "척도": "log1p" if logv else "원척도",
               "Tukey 이상 비율": ((xs < q1 - 1.5 * iqr) | (xs > q3 + 1.5 * iqr)).mean() if iqr > 0 else np.nan,
               "강건 z>3.5 비율": (0.6745 * (xs - med).abs() / mad > 3.5).mean() if mad > 0 else np.nan,
               "p99": x.quantile(.99), "최대": x.max()}
out = pd.DataFrame(rows).T
out
''')
md("""
수위 변화·타깃은 **12~25%가 "이상값"** 으로 잡혀요. 이 과제에서 가장 중요한 신호(큰 상승)를 지워 버리게 되죠. 그래서 **기계적 기준 대신 물리적으로** 판단했어요.
- 1시간에 100mm 넘는 IMERG 강우: 한국에서 실제로 기록되는 극한 호우 범위 → **진짜 극단**
- 큰 수위 상승: 과제의 핵심 → **진짜 극단, 절대 제거 금지**

물리적으로 **오류**라고 판단한 건 하나, 그리고 **주의할 특성**이 둘이에요. 관측소 전부를 훑어서 찾았어요.

#### (1) 한 번도 안 변하는 관측소 — 465곳 전부 검사
""")
code(r'''
wl_sd = pd.concat([tr, te]).groupby("station_id").wl_t.std()
stuck = wl_sd[wl_sd == 0].index.tolist()
print("수위 표준편차가 정확히 0인 관측소:", stuck)
for s_ in stuck:
    print(f"  {s_}: train {int((tr.station_id == s_).sum())}행 · test {int((te.station_id == s_).sum())}행 · 값 {tr.loc[tr.station_id == s_, 'wl_t'].unique()}")
fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(np.log10(wl_sd.clip(lower=1e-4)), bins=50, color=C_TR)
ax.axvline(-4, color=C_TE, lw=2); ax.text(-3.9, ax.get_ylim()[1] * .8, f"표준편차 0: {', '.join(stuck)}", color=C_TE)
ax.set_xlabel("관측소별 수위 표준편차 (log10 m)"); ax.set_ylabel("관측소 수"); ax.set_title("465곳 중 딱 한 곳만 완전히 멈춰 있다")
plt.show()
''')
md("""
**ST0027**은 train 1,676행과 test 413행 모두 수위가 28.19m로 고정이에요. 계측기 고장으로 보여요. 이 관측소의 타깃은 train에서 전부 0이에요.
→ **학습에서 빼고, test 예측은 0으로 고정**해요.

#### (2) 맑은 날에도 레이더가 켜져 있는 관측소 — 465곳 전부
하루 종일 비가 0인데 레이더 반사도가 높다면, 비구름이 아니라 **산이나 건물에 반사된 신호(지형 클러터)** 일 가능성이 커요.
""")
code(r'''
dry = both[(both.imerg_rain_24h == 0) & (both.sfc_rain_24h == 0)]                  # 24시간 동안 비가 전혀 없는 행
clear = dry.groupby("station_id").radar_reflectivity_mean.median()
print(f"맑은 날 레이더 평균: 전체 중앙값 {dry.radar_reflectivity_mean.median():.2f}")
clutter = clear[clear > 4].sort_values(ascending=False)
print(f"맑은 날 중앙값이 4를 넘는 관측소: {len(clutter)}곳")
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.hist(clear, bins=60, color=C_TR); ax.axvline(4, color=C_TE, ls="--")
ax.text(4.2, ax.get_ylim()[1] * .8, f"4 초과: {len(clutter)}곳", color=C_TE)
ax.set_xlabel("관측소별 '맑은 날' 레이더 반사도 중앙값"); ax.set_ylabel("관측소 수"); ax.set_title("대부분 0 근처, 일부만 맑은 날에도 높다")
plt.show()
clutter.round(2).head(10)
''')
md("""
대부분 관측소는 맑은 날 반사도가 0 근처인데, **21곳**(train+test 기준. train만 보면 18곳)은 맑은 날에도 4를 넘어요. 클러터로 추정해요.
→ 이 관측소들에서는 레이더 값을 그대로 쓰면 "비가 온다"고 착각할 수 있어요. **관측소별 맑은 날 바탕값을 빼는 것**을 검토해요.

#### (3) 수위는 1cm 계단 — 관측소 전부
""")
code(r'''
w = pd.concat([tr, te])
on_cm = np.isclose(w.wl_t * 100, np.round(w.wl_t * 100), atol=2e-3)                  # 값이 1cm 격자 위에 있나
per_st = pd.Series(on_cm).groupby(w.station_id.to_numpy()).mean()
print(f"수위가 1cm 단위인 행: {on_cm.mean():.2%} · 가장 낮은 관측소 {per_st.min():.2%}")
print(f"타깃이 0 또는 ±1cm 인 행: {(tr[TARGET].abs() <= .0101).mean():.0%}")
fig, ax = plt.subplots(figsize=(10, 3))
d = tr.dwl_1h[(tr.dwl_1h.abs() <= .1)]
ax.hist(d, bins=np.arange(-.1, .1001, .0025), color=C_TR); ax.set_yscale("log")
ax.set_xlabel("dwl_1h (±10cm 구간)"); ax.set_ylabel("행 수 (log)"); ax.set_title("수위 변화는 1cm 칸에만 값이 있다 (사이는 비어 있음)")
plt.show()
''')
md("""
수위는 **모든 관측소에서 1cm 단위**로 기록돼요. 그래서 타깃의 **절반 가까이(46%)가 0 또는 ±1cm**예요.
→ 1cm보다 정밀하게 맞히려는 노력은 의미가 없어요. RMSE는 큰 상승이 좌우하니까 **꼬리를 잘 맞히는 데 집중**해요(2권).
""")

# ---------------- 5
md("""
## 5. 누수 점검 — 미래 정보가 섞일 수 있는 길

이 대회에서 누수가 생길 수 있는 길은 두 가지예요.

### 5-1. test 파일 안의 "뒤 시각 행"
test에도 같은 관측소의 **연속된 시각**이 들어 있어요. 14시 행의 타깃은 "15~20시 최고 수위 − 14시 수위"인데, **15~20시 행이 test에 있으면 그 행들의 `wl_t`로 타깃을 거의 그대로 계산할 수 있어요.**

아래 셀은 test에서 **뒤 시각 행이 몇 개나 존재하는지만** 세요. **그 행들의 값은 읽지도, 계산에 쓰지도 않아요.**
""")
code(r'''
# 존재 여부만 센다: 같은 관측소의 (hour + j) 행이 test에 있는가. 값(wl_t)은 사용하지 않는다.
exists = np.column_stack([np.isin(key_te + j, key_te) for j in range(1, 7)])
n_future = exists.sum(axis=1)
dist = pd.Series(n_future).value_counts(normalize=True).sort_index()
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(dist.index, dist.values * 100, color=[C_TE if i == 6 else C_GRAY for i in dist.index])
ax.set_xlabel("test 안에 존재하는 '뒤 1~6시간' 행 수"); ax.set_ylabel("test 행 비율 (%)")
ax.set_title("test 행의 과반은 뒤 6시간이 test 안에 다 있다 → 쓰면 답이 계산됨"); plt.show()
print(f"뒤 6시간이 전부 있음: {(n_future == 6).mean():.0%} · 하나라도 있음: {(n_future >= 1).mean():.0%}")
''')
md("""
test 행의 **61%는 뒤 6시간이 test 안에 전부** 있어요. 이 값들을 쓰면 예측이 아니라 **답 계산**이 돼요.
→ **원칙: 예측 시점 t 이후의 정보는 train이든 test든 어떤 행에서도 쓰지 않는다.**
  - 금지 예: test에서 같은 관측소의 다음 행 `wl_t`를 가져오기, 사상 전체 평균·최대·순위(미래 포함)로 변수 만들기
  - 허용 예: 그 행 안의 과거 값(`wl_t_minus_*`, 누적 강우)과 그 차분, train에서 미리 계산한 관측소 특성

### 5-2. 이웃 행의 타깃 겹침
1장에서 봤듯이 옆 시각의 행끼리는 타깃 구간이 5시간 겹쳐요. 검증할 때 행을 무작위로 섞으면, 검증 행의 "쌍둥이"가 학습에 들어가 점수가 부풀어요.
→ **검증은 사상 단위로 나눠요**(GroupKFold). 얼마나 부풀어 보이는지는 6권에서 실제로 비교해요.
""")

# ---------------- 6
md("""
## 6. 정리

### 발견과 전처리 결정

| 발견 | 전체에 해당? | 전처리 결정 |
|---|---|---|
| NaN·중복 0, train/test 관측소 동일 | ✅ 전 행 | 대치·삭제 없음 |
| 사상 152개, 사상마다 관측소 대부분 등장 | ✅ 152개 전부 | 관측소 특성은 train에서 배워 test에 사용 |
| 빠진 시각 2.5%, 비와 무관 | ✅ 강우 10칸 전부. 관측소 58곳은 10% 넘게 빠짐 | 이전 행이 없으면 NaN으로 두기 |
| 강우·유역 값을 여러 관측소가 공유 | ✅ train 사상 전부 | 관측소 구별 정보는 수위 이력에서 뽑기 |
| lag·누적강우 계산 100% 일치 | ✅ 관측소 전부 | 이전 행 참조 변수 안심하고 만들기 |
| 레이더 평균 > p95 (7.5%) | 99.9%가 약한 에코 | 오류 아님, 그대로 |
| 수위 기준면이 관측소마다 다름 | 설명서 명시 | 절대값 대신 **변화량**(`dwl_k = wl_t − wl_t−k`) 사용 |
| 지상강수 0.5 계단 | 관측소 대부분 (69곳은 80% 미만) | mm로 추정, 약한 비는 IMERG로 보완 |
| ST0027 수위 고정 | 465곳 중 1곳 | 학습 제외, test 예측 0 |
| 맑은 날 레이더가 높은 관측소 21곳 | 관측소별 확인 | 맑은 날 바탕값 차감 검토 |
| 수위 1cm 계단, 타깃 46%가 0·±1cm | ✅ 관측소 전부 | 작은 값보다 꼬리 예측에 집중 |
| 교과서 이상값 규칙이 12~25%를 오판 | 0이 많은 변수 전부 | 물리적 기준으로 판단, 큰 값 제거 금지 |

### 누수 점검표 — 이 노트북에서 미래 정보를 쓴 곳

| 어디 | 미래 정보? | 판단 |
|---|---|---|
| 2장 사상 복원 | 시각의 존재 여부만 사용(값 아님) | 구조를 파악하는 용도. 모델 입력 아님 |
| 3장 공유 시계열 비교 | 사상 안 강우 시계열 전체(입력값) | 구조 파악용. 모델 입력 아님 |
| 4-4 일관성 검산 | **과거 행만** 사용 | 누수 없음 |
| 4-5 관측소별 통계(ST0027, 클러터) | 관측소 전체 기간의 입력값 | 타깃 미사용. 모델에 넣을 땐 train에서만 계산 |
| 5-1 test 뒤 시각 행 | **존재 여부만 셈, 값은 안 씀** | 누수 없음. 모델에서는 절대 사용 금지 |

> 🚫 예측 시점 t 이후의 정보는 어떤 행에서도 쓰지 않는다.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "nb1_data_structure_quality.ipynb"
nbf.write(nb, out); print("written", out, len(C), "cells")

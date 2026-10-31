---
created: 2026-10-31
type: note
topics: [공부]
tags: [KISTI2026, 수위예측, 피처엔지니어링, 팀공유, 유역환경, 이웃관측소]
summary: 수아의 두 실험 아이디어(유역×강우, 사상 전체 상승 신호)에 대해 EDA 판정·기준 모델 잔차 점검 결과·권장 변형·코드 뼈대·누수 체크를 정리한 전달 자료
status: organized
---

# KISTI 수위예측 — 수아 피처 참고자료 (유역×강우, 상승 신호)

> 작성 2026-10-31 · 한동 → 수아
> 근거: `eda/report_final.md`(팀 EDA 최종본), 기준 모델 잔차 분석 [[KISTI 수위예측 - 모델링 전략용 EDA (기준 모델 잔차)]]
> 잔차 = 정답 − 기준 LightGBM(dev OOF 0.2178) 예측. "잔차 상관"은 새 피처가 기준 모델이 놓친 정보를 얼마나 갖는지 미리 보는 값이다. 실제 효과는 CV로 판정한다.

## 0. 먼저 볼 것 — 오차가 어디 있나

![[kmeda_situation.png|520]]
*오차제곱합의 46.5%가 "비는 오는데 아직 정체 중" 행, 36.9%가 "상승 중" 행에서 나온다.*

- **상승 시작 행**(dwl_1h ≤ 0.05이면서 정답 > 0.5)이 행 3.3%에 오차 43.8%다. 새 피처는 이 행을 정체 행과 구별해야 RMSE가 줄어든다.
- 예측은 이미 보정되어 있다(예측 구간별 실제 평균 ≈ 예측). 그래서 같은 정보를 다르게 변환한 피처는 효과가 작다. **기존 72개 피처에 없는 정보**가 필요하다.
- 비교할 때 전체 RMSE와 함께 **상승 시작 행의 오차제곱합**도 같이 보면, 피처가 겨냥한 곳에서 효과가 났는지 알 수 있다.

## 1. 아이디어 ① 유역 × 강우

### EDA가 이미 판정한 것 (report_final 5.3절, 부록 A)

| 조합 | 판정 | 근거 |
|---|---|---|
| 큰 유역 × 강한 비 (log 면적 × 24h 강우) | **기각** | 섞은 그림에서는 보였지만 사상 19개 중 8개에서만 성립 |
| 도시화 × 짧은 비 (urban × 1h 강우) | 채택(중간 강우) | 사상 78%. 단 40mm 초과 극한 비에서는 방향이 반대 |
| **관측소 안 수위 분위 q × 강우** | **1순위** | 관측소 80%·사상 96%. 상위 5사상 제외 잔차 설명 +2.9%, 일관성 91% |
| 면적↑ → 비유량↓ | 기각 | 면적 3분위별 타깃÷강우 0.0040 / 0.0052 / 0.0046 |
| 경사↑ → 빠른 반응 | 기각 | 면적 통제 시 오히려 +0.13(가파를수록 늦음) |
| 선행 습윤↑ → 반응↑ | 기각 | 최근 강우·경과 시간 통제 시 음(ρ −0.15) |
| 면적 → 도달 시간 | 채택 | 관측소 지체 L과 면적 ρ 0.55 |

### 기준 모델 잔차로 본 사전 점검 (비조석)

- `urban_ratio × log1p(sfc_rain_1h)` (중간 강우 행): 잔차 상관 **0.020**
- `log1p(wsarea) × log1p(sfc_rain_24h)`: 잔차 상관 **0.010**
- 둘 다 약하다. 트리 모델은 두 변수를 나눠서도 이 교차를 어느 정도 학습한다.

### 권장 변형 (기대 순)

1. **q × log1p(sfc_rain_6h)**: q = 관측소별 dev `wl_t` 분포에서의 분위. "같은 비라도 이미 물이 높으면 덜 오른다"를 관측소 기준면 문제 없이 표현한다. 유역 × 강우 실험의 첫 후보로 가장 근거가 강하다.
2. **도착 예정 비**: 면적 대신 관측소 지체 L(`eda/cache/results/nb09_station_lag.csv`)을 써서, 이미 내렸지만 아직 도달하지 않은 비를 합한다. Σ sfc_rain_1h(t−k), k = max(0, L−6) … L−1. "큰 유역은 24h 강우가 중요할 것"이라는 가설을 면적보다 직접 표현한다.
3. 도시화 × log1p(sfc_rain_1h): 한다면 단조 제약 없이 한다.

## 2. 아이디어 ② 사상 전체 상승 신호

### 먼저 알아둘 것

- 베이스라인에 이미 동시각 전체 평균이 있다: `xs_mean_{imerg_rain_1h, imerg_rain_6h, sfc_rain_6h, dwl_1h, dwl_3h, dwl_6h}`, `xs_rising_frac`, `dwl_3h_vs_xs`.
- 잔차를 분해하면 **사상 전체가 같은 시각에 함께 틀리는 몫은 오차의 5.6%**뿐이다(무작위 대조 1.6%). 전체 관측소의 분위수·비율을 더 만들어도 다룰 수 있는 몫의 상한이 작다.

![[kmeda_common_error.png|520]]
*함께 틀리는 단위는 사상 전체가 아니라 같은 강우 격자(약 10km)를 쓰는 이웃 관측소다.*

### 그래서 권장하는 방향

| 후보 | 정의 | 잔차 점검 | 판단 |
|---|---|---|---|
| 같은 격자 이웃의 현재 상승 | 같은 IMERG(또는 지상강수) 격자 관측소들의 시점 t `dwl_1h` 평균, 자기 제외 | 상관 0.065~0.072, 사상 50~56%에서 양 | 약함. 빠르게 한 번만 시험 |
| **선행(상류) 관측소 신호** | train 시차 상관으로 1~2시간 앞서는 관측소를 찾고, 그 관측소들의 시점 t `dwl_1h` 평균 | EDA 6.6절, 중간보고서 9장: HGB 기준 −2.4% | **가장 유망**. 관계는 학습 fold 사상으로만 추정 |
| 이웃 강우 추세 | 같은 격자·인접 격자의 `sfc_rain_1h` 증가 추세 | 미점검 | 잔차가 미래 6h 강우와 상관 0.18 → 비구름이 오는 방향을 잡는 신호가 될 수 있음 |

- 같은 격자 이웃끼리는 오차가 크게 겹친다(공통 몫 0.68~0.75, 대조 0.37~0.44). 그런데 이웃의 **현재** 상태는 그 오차를 거의 설명하지 못한다. 겹치는 오차는 대부분 t 이후에 올 비에서 생기는 것으로 보인다. 그래서 같은 시각 이웃보다 **시간적으로 앞서는** 관측소나 강우 신호가 더 유망하다.
- 규정: Q&A 10/6 (나)(다)에서 같은 사상 다른 관측소의 시점 t와 그 이전 값은 허용됐다. 관측소 간 관계는 train으로만 만든다.
- 조석 관측소에는 쓰지 않는다. 상류 변수가 조석에서는 오히려 나빠졌다(0.376 → 0.386).

## 3. 코드 뼈대

모두 `make_features` 뒤에 붙이는 함수다. 10/31에 train 전체로 세 함수를 실행했고, 누수 자가검증(사상 마지막 6시간 삭제 후 재계산)에서 세 피처 모두 값이 같았다. dev로 fit하는 부분은 fold마다 학습 사상으로 다시 fit하는 것이 원칙이다. 타깃을 쓰지 않는 입력 통계라면 `station_stats`처럼 dev 전체로 한 번 fit해도 된다(팀 기존 처리와 같음).

**격자 이웃 묶기** — 강우 시계열이 완전히 같은 관측소끼리. dev 기준 IMERG 301개, 지상강수 225개 격자가 나오고, 모든 공통 시각에서 값이 일치하는 것을 확인했다.

```python
def fit_cells(dev, col="imerg_rain_1h"):
    sig = dev.pivot_table(index="timestamp", columns="station_id", values=col).round(4)
    sig = sig.dropna(axis=0)          # 모든 관측소가 있는 시각만 (dev 301개)
    return sig.apply(lambda c: hash(tuple(c.values))).rename("cell")

def add_cell_neighbor(df, cells, src="dwl_1h", name="nb_dwl_1h"):
    df = df.merge(cells.reset_index(), on="station_id", how="left")
    g = df.groupby(["event_id", "timestamp", "cell"])[src]
    s, n = g.transform("sum"), g.transform("count")
    own, has = df[src].fillna(0), df[src].notna().astype(int)
    k = n - has
    df[name] = np.where(k > 0, (s - own) / k.clip(lower=1), np.nan)   # 자기 제외 평균
    return df.drop(columns="cell")
```

**수위 분위 q**

```python
def fit_wl_ref(dev):
    return {s: np.sort(v.values) for s, v in dev.groupby("station_id")["wl_t"]}

def add_wl_quantile(df, ref):
    q = np.full(len(df), np.nan)
    for s, idx in df.groupby("station_id").indices.items():
        if s in ref:
            q[idx] = np.searchsorted(ref[s], df["wl_t"].values[idx], side="right") / len(ref[s])
    df["wl_q"] = q
    df["wl_q_x_sfc6"] = q * np.log1p(df["sfc_rain_6h"])
    return df
```

**도착 예정 비** — 베이스라인 `lag()`로 `sfc_rain_1h_lag1~5`를 먼저 만든다(정확히 k시간 전 행만 사용).

```python
L = pd.read_csv(ROOT / "eda/cache/results/nb09_station_lag.csv").set_index("station_id")["L"]

def add_arriving_rain(df, L):
    l = df["station_id"].map(L).fillna(L.median()).round().astype(int).values
    out = np.zeros(len(df))
    for k in range(6):
        col = df["sfc_rain_1h"] if k == 0 else df[f"sfc_rain_1h_lag{k}"]
        use = (k >= l - 6) & (k <= l - 1)
        out += np.where(use, col.fillna(0).values, 0.0)
    df["arriving_rain"] = out
    return df
```

- L의 중앙값은 3시간이고 4.5 같은 반정수가 있어 반올림한다. L > 6인 관측소(8%)는 창이 lag 범위를 벗어나 0이 된다.
- L이 지상·IMERG 중 어느 원천 기준인지는 9권(`nb9_travel_time_and_rain_windows.ipynb`)에서 확인이 필요하다.
- 선행 관측소 코드는 5권(`nb5_station_network.ipynb`)과 중간보고서 9장에 있다. fold 안에서 다시 찾도록 바꿔야 한다.

## 4. 체크리스트

- [ ] 실험 공지(docs/04 4절) — 겹치지 않도록 누가 무엇을 할지 채팅에서 먼저 정하기
- [ ] 누수 자가검증 통과 (사상 마지막 6시간 삭제 후 재계산 → 값 동일)
- [ ] 이웃·선행 관계, q 기준 분포, L은 train(dev)으로만
- [ ] 기록: experiments.csv 한 줄 + 이유. 이유에는 **상승 시작 행 SSE**, **비조석 RMSE**, 상위 10사상 변화를 같이 적기
- [ ] 만들지 말 것: 조석 위상 변수, 사상 단계 격자, Tc_index, 레이더 p95 − 평균, 큰 유역 × 강한 비 단독 교차(이미 기각)

"""T1. 1장 개요(표 1-1) + 2장 품질 점검(표 2-1~2-4, 그림 2-1)."""
import numpy as np
import pandas as pd

from tb_common import (load_tb, station_table, style, figsize, save, save_table, save_json, plt,
                       load_json, C_NONTIDAL, C_ALL, WL, IMERG, SFC, RADAR, ENV, TARGET)

# ---------------- 표 1-1 변수 분류 ----------------
VARS = [
    ("row_id", "식별", "명목", "-", "-", "행 식별자"),
    ("timestamp", "식별", "구간(합성 시각)", "시각", "-", "사상 내 순서·간격만 유효"),
    ("station_id", "식별", "명목", "-", "-", "465곳, train/test 공유"),
    *[(c, "동적 입력", "구간", "m", "변함", "관측소별 기준면 상이 → 변화량만 의미") for c in WL],
    *[(c, "동적 입력", "비율", "mm", "변함", "과거 누적") for c in IMERG],
    *[(c, "동적 입력", "비율", "미확인(mm 추정)", "변함", "0.5 계단") for c in SFC],
    ("radar_reflectivity_mean", "동적 입력", "구간", "미기재", "변함", "로그형 척도"),
    ("radar_reflectivity_p95", "동적 입력", "구간", "미기재", "변함", "로그형 척도"),
    ("radar_reflectivity_ge20_fraction", "동적 입력", "비율", "0~1", "변함", "임계값 요약"),
    ("cat_area", "정적 입력", "비율", "원자료", "불변", "소권역 평균"),
    ("wsarea", "정적 입력", "비율", "원자료", "불변", "소권역 평균"),
    ("mean_elevation", "정적 입력", "구간", "m", "불변", ""),
    ("mean_slope", "정적 입력", "비율", "원자료", "불변", ""),
    ("urban_ratio", "정적 입력", "비율", "%", "불변", ""),
    ("forest_ratio", "정적 입력", "비율", "%", "불변", ""),
    ("stream_order", "정적 입력", "순서(평균 → 비정수)", "차수", "불변", ""),
    (TARGET, "타깃", "비율(차이값)", "m", "변함", "음수 가능, train만"),
]


def table_1_1():
    t = pd.DataFrame(VARS, columns=["변수", "역할", "측정 척도", "단위", "시간 변화", "비고"]).set_index("변수")
    save_table(t, "1-1", "variables")
    return t


# ---------------- 2.1 완전성: 사상 내 결측 vs 강우 ----------------
def completeness(tr):
    blk_len = tr.groupby("event_block")["timestamp"].agg(lambda s: s.nunique())
    g = tr.groupby(["event_block", "station_id"], observed=True).agg(
        n=("timestamp", "size"), rain=("imerg_rain_1h", "mean"))
    g["expected"] = g.index.get_level_values(0).map(blk_len)
    g["miss"] = 1 - g["n"] / g["expected"]
    g["decile"] = pd.qcut(g["rain"].rank(method="first"), 10, labels=range(1, 11))
    s = g.groupby("decile", observed=True)[["n", "expected"]].sum(); by = 1 - s["n"] / s["expected"]
    overall = 1 - g["n"].sum() / g["expected"].sum()

    style()
    fig, ax = plt.subplots(figsize=figsize(16, 6))
    ax.bar(by.index.astype(int), by.values * 100, color=C_NONTIDAL)
    ax.axhline(overall * 100, color=C_ALL, ls="--", lw=1)
    ax.text(10.4, overall * 100, f"전체 {overall*100:.1f}%", va="bottom", ha="right", color=C_ALL)
    ax.set_xlabel("관측소×사상의 평균 IMERG 1h 강우 10분위 (1 = 가장 적음)")
    ax.set_ylabel("사상 안 빠진 시각 비율 (%)")
    ax.set_xticks(range(1, 11))
    ax.set_title(f"그림 2-1  비가 많아도 빠진 시각이 늘지 않는다 (n = {len(g):,} 관측소×사상)")
    save(fig, "2-1", "missing_vs_rain")
    return {"overall_missing": overall, "by_rain_decile": by.round(4).to_dict()}


# ---------------- 2.3 유효성 규칙 ----------------
def validity(df):
    rules = {
        "강우 ≥ 0": (df[IMERG + SFC] < 0).any(axis=1),
        "IMERG 1h ≤ 6h ≤ 24h": (df.imerg_rain_1h > df.imerg_rain_6h + 1e-3) | (df.imerg_rain_6h > df.imerg_rain_24h + 1e-3),
        "지상 1h ≤ 6h ≤ 24h": (df.sfc_rain_1h > df.sfc_rain_6h + 1e-3) | (df.sfc_rain_6h > df.sfc_rain_24h + 1e-3),
        "레이더 비율 0~1": ~df.radar_reflectivity_ge20_fraction.between(0, 1),
        "레이더 평균 ≤ p95": df.radar_reflectivity_mean > df.radar_reflectivity_p95 + 1e-3,
        "도시 + 산림 ≤ 100%": (df.urban_ratio + df.forest_ratio) > 100 + 1e-3,
        "비율 변수 0~100%": ~(df.urban_ratio.between(0, 100) & df.forest_ratio.between(0, 100)),
        "면적 > 0": (df.cat_area <= 0) | (df.wsarea <= 0),
    }
    t = pd.DataFrame({"위반 행": {k: int(v.sum()) for k, v in rules.items()},
                      "위반 비율": {k: float(v.mean()) for k, v in rules.items()}})
    v = rules["레이더 평균 ≤ p95"]
    t["판정"] = "위반 없음"
    t.loc["레이더 평균 ≤ p95", "판정"] = (
        f"오류 아님 — 위반 행의 {(df.radar_reflectivity_p95[v] == 0).mean():.1%}가 p95=0·평균 소량"
        " (에코 화소 5% 미만인 약한 에코)")
    # 정보성 사실(위반이 아닌 특성)
    nz = df.sfc_rain_1h[df.sfc_rain_1h > 0]
    facts = {
        "wl_t 범위(m)": f"{df.wl_t.min():.2f} ~ {df.wl_t.max():.2f} (기준면 혼재)",
        "stream_order 비정수 관측소": int((df.groupby('station_id', observed=True).stream_order.first() % 1 != 0).sum()),
        "sfc_rain_1h>0 중 0.5 배수": float(np.isclose((nz * 2) % 1, 0, atol=1e-4).mean()),
        "sfc_rain_1h>0 중 0.1 배수": float(np.isclose((nz * 10) % 1, 0, atol=1e-3).mean()),
        "도시+산림 최대(%)": float((df.urban_ratio + df.forest_ratio).max()),
    }
    return t, facts


# ---------------- 2.4 일관성 (기존 s01 수치) ----------------
def consistency():
    s01 = load_json("s01_integrity.json")
    rows = []
    for part in ["train", "test"]:
        c = s01[part]["continuity"]
        for k in [1, 3, 6, 12, 24]:
            e = c[f"lag{k}h"]
            rows.append((part, f"wl_t_minus_{k}h = {k}시간 전 행 wl_t", e["row_exists"], e["match_rate_when_exists"]))
        for src in ["imerg", "sfc"]:
            r = s01[part]["rain_accum"][src]
            rows.append((part, f"{src} 6h = 1h 6개 합", r["sum1h_vs_6h"]["rows_checkable"], r["sum1h_vs_6h"]["match_rate"]))
            rows.append((part, f"{src} 24h = 1h 24개 합", r["sum1h_vs_24h"]["rows_checkable"], r["sum1h_vs_24h"]["match_rate"]))
    tr = s01["train"]["target_recompute"]
    rows.append(("train", "타깃 = 다음 6행 max − wl_t", tr["rows_with_6_future"], tr["match_rate_full"]))
    return pd.DataFrame(rows, columns=["데이터", "검산", "검산 가능 비율", "일치율"]).set_index(["데이터", "검산"])


# ---------------- 2.5 이상값 ----------------
LOGVARS = IMERG + SFC + ["cat_area", "wsarea"]


def outliers(tr, st):
    cols = ["dwl_1h", "dwl_3h", "dwl_24h"] + IMERG + SFC + RADAR[:2] + [TARGET]
    rows = {}
    for c in cols + ENV:
        x = (st[c] if c in ENV else tr[c]).astype(float)
        scale = "log1p" if c in LOGVARS else "원척도"
        xs = np.log1p(x) if scale == "log1p" else x
        q1, q3 = xs.quantile([.25, .75])
        iqr = q3 - q1
        tukey = ((xs < q1 - 1.5 * iqr) | (xs > q3 + 1.5 * iqr)).mean() if iqr > 0 else np.nan
        med = xs.median(); mad = (xs - med).abs().median()
        rz = (0.6745 * (xs - med).abs() / mad > 3.5).mean() if mad > 0 else np.nan
        rows[c] = {"단위": "관측소" if c in ENV else "행", "척도": scale, "n": len(x),
                   "Tukey 이상 비율": tukey, "MAD z>3.5 비율": rz,
                   "최소": x.min(), "p99": x.quantile(.99), "최대": x.max()}
    t = pd.DataFrame(rows).T
    t["판정"] = t.index.map(lambda c: VERDICT.get(c, VERDICT_DEFAULT))
    return t


VERDICT_DEFAULT = "실제 극값 — 기준을 넘는 값도 물리적으로 가능한 범위"
VERDICT = {
    "dwl_1h": "실제 극값 — 0에 몰린 분포라 IQR이 좁아 과다 탐지(기계적 기준의 한계)",
    "dwl_3h": "실제 극값 — 동일(0 집중으로 과다 탐지)",
    "dwl_24h": "실제 극값 — 동일, 조석 관측소 ±3m 포함",
    TARGET: "실제 극값 — 과제의 핵심 신호, 제거 금지",
    "imerg_rain_1h": "실제 극값 — 1h 100mm 초과 7행(국내 극한 호우 범위)",
    "sfc_rain_1h": "실제 극값 — 0.5 계단 분포라 MAD=0(계산 불가)",
    "cat_area": "실제 값 — 대유역, 로그 변환으로 대응",
    "wsarea": "실제 값 — 로그 척도에서는 이상 없음",
}


def cases(tr):
    """알려진 사례: 고정 관측소, 레이더 클러터, 분해능."""
    s = tr.groupby("station_id", observed=True)
    stuck = s.wl_t.std()
    stuck = stuck[stuck == 0].index.astype(str).tolist()
    dry = tr[(tr.imerg_rain_24h == 0) & (tr.sfc_rain_24h == 0)]
    clear = dry.groupby("station_id", observed=True).radar_reflectivity_mean.median()
    clutter = clear[clear > 4].sort_values(ascending=False)
    d = tr["dwl_1h"].abs()
    d = d[d > 0]
    return {
        "stuck_stations": stuck,
        "clear_day_radar_median_all": float(dry.radar_reflectivity_mean.median()),
        "clutter_stations_n": int(len(clutter)),
        "clutter_stations": {k: round(float(v), 2) for k, v in clutter.items()},
        "min_nonzero_abs_dwl_1h": float(d.min()),
        "target_zero_or_1cm_share": float((tr[TARGET].abs() <= 0.0101).mean()),
        "extreme_imerg_1h_gt_100": int((tr.imerg_rain_1h > 100).sum()),
        "extreme_sfc_1h_gt_80": int((tr.sfc_rain_1h > 80).sum()),
    }


# ---------------- 2.6 품질 → 전처리 결정 ----------------
def decisions(c):
    rows = [
        ("수위가 한 번도 변하지 않는 관측소 " + ", ".join(c["stuck_stations"]), "2.5",
         "test 예측 0 고정, 학습에서 제외"),
        ("wl_t는 관측소마다 기준면이 다른 구간 척도", "1.4, 2.3", "절대값 대신 차분(dwl_k)·관측소 내 상대값 사용"),
        ("유역면적 왜도 7~12", "2.5, 3.6", "log1p 변환(선형·신경망), 트리는 선택"),
        (f"맑은 날에도 레이더 반사도가 높은 관측소 {c['clutter_stations_n']}곳", "2.5", "관측소별 맑은 날 바탕값 차감 검토"),
        ("지상강수 0.5 계단, 약한 비 미감지", "2.3", "단위는 mm로 추정 표기, 약한 비는 IMERG로 보완"),
        ("수위 1cm 분해능, 타깃 절반 가까이가 0 또는 ±1cm", "2.5", "작은 값 정밀도보다 꼬리 예측에 집중"),
        ("결측·중복·정합성 문제 없음", "2.1~2.4", "별도 대치·삭제 없음. lag 파생변수 안심하고 사용"),
    ]
    return pd.DataFrame(rows, columns=["품질 발견", "근거 절", "전처리 결정"])


def main():
    t11 = table_1_1()
    cols = ["station_id", "timestamp", "event_block", "wl_t", "dwl_1h", "dwl_3h", "dwl_24h", TARGET] + IMERG + SFC + RADAR + ENV
    tr = load_tb("train_sorted", cols)
    te = load_tb("test_sorted", [c for c in cols if c not in (TARGET,)])
    st = station_table(tr)

    res = {"completeness": completeness(tr)}
    both = pd.concat([tr.drop(columns=TARGET), te], ignore_index=True)
    t21, facts = validity(both)
    save_table(t21, "2-1", "validity"); res["validity_facts"] = facts
    save_table(consistency(), "2-2", "consistency")
    t23 = outliers(tr, st)
    c = cases(tr); res["cases"] = c
    save_table(t23, "2-3", "outliers")
    save_table(decisions(c), "2-4", "decisions")
    res["n_rows"] = {"train": len(tr), "test": len(te)}
    save_json(res, "t01_quality.json")
    print(t21, "\n", facts, "\n", t23.round(4), "\n", {k: v for k, v in c.items() if k != "clutter_stations"})


if __name__ == "__main__":
    main()

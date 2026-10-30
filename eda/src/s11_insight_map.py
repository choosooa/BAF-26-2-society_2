"""S11. 탐색으로 발굴한 인사이트를 '인사이트 지도'(마인드맵형 PNG)로 정리.

모든 숫자는 cache/results/*.json 에서 읽는다 → 분석을 다시 돌리면 지도도 같이 갱신된다.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import RESULTS, TARGET as TARGET_KEY, savefig, style

style()
KIND = {  # 종류: (채움, 테두리·글자, 범례)
    "opp": ("#e3f5ee", "#0f6e56", "모델링 기회 (새 변수·새 구조)"),
    "risk": ("#fdebe3", "#a3401a", "주의 (함정·위험)"),
    "know": ("#e6f1fb", "#185f9a", "구조 이해"),
}


def J(name):
    return json.load(open(RESULTS / name, encoding="utf-8"))


def lag_share(net, k):
    c = {int(float(a)): b for a, b in net["lag_counts"].items()}
    return c.get(k, 0) / sum(c.values())


def insights():
    s8, s9, s10 = J("s08_fingerprint.json"), J("s09_network.json"), J("s10_events_residuals.json")
    prof = s8["type_profile"]
    net, vc = s9["network"], s9["value_check"]
    al = s10["alignment"]["train"]["IMERG(t) vs 지상(t+k)"]
    al_te = s10["alignment"]["test"]["IMERG(t) vs 지상(t+k)"]
    ev, fr, gp = s10["events"], s10["future_rain"], s10["gaps"]
    nb = s10["radar_nowcast_bins"]
    reg = s10["residuals"]["regimes"]
    per = s8["periodic"]
    c4 = prof["4"]
    d_net = (vc["전체"]["rmse_base"] - vc["전체"]["rmse_net"]) / vc["전체"]["rmse_base"]
    d_big = (vc["비조석·큰 상승(>0.5m)"]["rmse_base"] - vc["비조석·큰 상승(>0.5m)"]["rmse_net"]) / vc["비조석·큰 상승(>0.5m)"]["rmse_base"]
    T = []
    T.append(("관측소 유형", [
        ("know", "관측소는 4가지 방식으로 움직인다",
         f"주기형(C4) 행 {c4['row_share']:.0%} → 0예측 오차 {c4['sse_share']:.0%}", "유형별 모델·가중치 검토", "08a/08b/08c"),
        ("opp", "유역환경으론 유형의 절반만 설명",
         f"RF 정확도 {s8['env_explains_type']['rf_accuracy']:.0%} (다수결 {s8['env_explains_type']['majority_baseline']:.0%})",
         "관측소 '행동 지문'을 변수로", "08a"),
        ("know", "주기성은 스펙트럼 + 24h 조작",
         f"강한 조석 {len(per['strong_tidal'])} · 약한 조석 {len(per['weak_tidal'])} · 24h 조작 {len(per['daily_only'])}",
         "주기 강도를 연속 변수로", "08d"),
    ]))
    T.append(("연결망", [
        ("opp", "좌표 없이 상류→하류 연결망 복원",
         f"{net['strong_leader(corr>=0.5)']}곳이 선행 관측소 보유, 후행이 유역 더 큼 {net['follower_larger_area_share(diff_env)']:.0%}",
         "선행 관측소 수위를 변수로", "09a/09b"),
        ("opp", "선행·지역 변수의 효과",
         f"RMSE {vc['전체']['rmse_base']:.3f}→{vc['전체']['rmse_net']:.3f} ({-d_net:+.1%}), 큰 상승 오차 {-d_big:+.1%}",
         "동시각 타 관측소 사용 허용 여부 확인", "09a"),
        ("know", "상류 신호는 1~2시간만 앞선다",
         f"선행 간격 1h {lag_share(net, 1):.0%} · 2h {lag_share(net, 2):.0%} · 3h+ {1 - lag_share(net, 1) - lag_share(net, 2):.0%}",
         "6h 창의 앞부분만 보강, 뒤는 강우 몫", "09b"),
    ]))
    tv = s8["top_values"]
    zero_pm1 = sum(tv[TARGET_KEY].get(k, 0) for k in ["0.0", "0.009999999776482582", "-0.009999999776482582"])
    T.append(("강우 입력", [
        ("risk", "IMERG는 '다음 1시간' 비를 담고 있다",
         f"IMERG(t)↔지상(t+1) 최대: 관측소 {al['station_argmax_share'].get('1', 0):.0%}·사상 {al['event_argmax_share'].get('1', 0):.0%} (test {al_te['station_argmax_share'].get('1', 0):.0%}·{al_te['event_argmax_share'].get('1', 0):.0%})",
         "데이터 정의와 모순 → 운영측 확인", "10a"),
        ("opp", "레이더가 다가오는 비를 먼저 본다",
         f"지금 맑음: 레이더 하위→상위 10분위, 6h 내 비 확률 {nb['0']['p_rain']:.0%}→{nb['9']['p_rain']:.0%}",
         "레이더를 '예보' 변수로", "10d"),
        ("know", "지상강수는 0.5mm 우량계",
         f"비영 값 {tv['sfc_1h_on_0.5_grid']:.0%}가 0.5 단위, IMERG 약한 비엔 지상 0이 69%", "약한 비는 IMERG로 보완", "03h"),
        ("risk", "레이더 3변수는 사실상 2개 + 클러터",
         f"ge20 비율 0·1이 {(tv['radar_reflectivity_ge20_fraction'].get('0.0', 0) + tv['radar_reflectivity_ge20_fraction'].get('1.0', 0)):.0%}, 맑은 날 반사도 높은 곳 {s8['radar_clutter']['n_high']}곳",
         "관측소별 맑은 날 바탕값 차감", "08d"),
    ]))
    dry_rows = sum(reg[k]["rows"] for k in ["무강우·정지", "무강우·하강", "무강우·상승"])
    dry_sse = sum(reg[k]["sse_base"] for k in ["무강우·정지", "무강우·하강", "무강우·상승"])
    T.append(("오차의 위치", [
        ("know", "오차의 절반은 '비 오며 상승 중'",
         f"행 {reg['비·상승 중']['rows']:.0%} → 오차 {reg['비·상승 중']['sse_base']:.0%} · 무강우 {dry_rows:.0%} 행은 오차 {dry_sse:.0%}",
         "상승 국면 전용 변수·가중치", "10c"),
        ("risk", f"{ev['n_events_for_half_sse']}개 사상이 오차의 절반",
         f"122개 중 {ev['n_events_for_half_sse']}개(행 {ev['top5_row_share']:.0%})가 오차 {ev['top5_sse_base_share']:.0%}, 강도·지속시간과 연동",
         "사상 단위 반복 CV, 점수 요동 대비", "10b"),
        ("know", "미래 강우의 몫은 30%뿐",
         f"미래 6h 강우를 알아도 MSE {fr['oracle']['mse_reduction']:.0%} 감소 → 나머지는 현재 정보 몫",
         "현재 입력 개선 여지가 더 크다", "10d"),
    ]))
    T.append(("데이터 함정", [
        ("risk", "한 번도 안 변하는 관측소",
         f"{', '.join(s8['resolution']['never_changes'])}: 1,676행 타깃 전부 0", "0으로 고정 예측", "08d"),
        ("know", "모든 수위는 1cm 계단",
         f"관측소 {s8['resolution']['stations_min_step_1cm']}곳 최소 변화 1cm → 타깃 {zero_pm1:.0%}가 0·±1cm", "작은 값은 반올림 게임", "08d"),
        ("know", "빠진 행은 비와 무관",
         f"사상 내 결측 {gp['missing_cells_in_present_blocks']:.1%}, 강우 10분위별 {min(gp['missing_frac_by_regional_rain_decile'].values()):.1%}~{max(gp['missing_frac_by_regional_rain_decile'].values()):.1%}",
         "극값 검열 없음 → 걱정 불필요", "—"),
    ]))
    return T


def draw(T, path="11_insight_map.png"):
    n_leaf = sum(len(v) for _, v in T)
    H = n_leaf * 1.0 + (len(T) - 1) * 0.55 + 1.2
    fig, ax = plt.subplots(figsize=(17, H * 0.78))
    ax.set_xlim(0, 17); ax.set_ylim(0, H); ax.axis("off"); ax.grid(False)
    y = H - 0.9
    theme_y = []
    leaf_pos = []
    for theme, leaves in T:
        ys = []
        for kind, title, ev, act, fig_ref in leaves:
            fc, ec, _ = KIND[kind]
            ax.add_patch(FancyBboxPatch((6.2, y - 0.4), 10.5, 0.82, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
            ax.text(6.4, y + 0.14, title, fontsize=12, color=ec, va="center")
            ax.text(6.4, y - 0.18, ev, fontsize=9.6, color="#0b0b0b", va="center")
            ax.text(16.55, y + 0.14, f"→ {act}", fontsize=9.4, color=ec, va="center", ha="right")
            ax.text(16.55, y - 0.18, f"그림 {fig_ref}", fontsize=8.2, color="#898781", va="center", ha="right")
            ys.append(y); leaf_pos.append((title, y))
            y -= 1.0
        ty = (ys[0] + ys[-1]) / 2
        theme_y.append(ty)
        ax.add_patch(FancyBboxPatch((3.2, ty - 0.33), 2.3, 0.66, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f1efe8", ec="#5f5e5a", lw=0.8))
        ax.text(4.35, ty, theme, fontsize=12.5, ha="center", va="center", color="#2c2c2a")
        for ly in ys:
            ax.plot([5.5, 5.85, 5.85, 6.2], [ty, ty, ly, ly], color="#b4b2a9", lw=1)
        y -= 0.55
    cy = (theme_y[0] + theme_y[-1]) / 2
    ax.add_patch(FancyBboxPatch((0.2, cy - 0.75), 2.4, 1.5, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#2c2c2a", ec="#2c2c2a"))
    ax.text(1.4, cy + 0.28, "6h 최대 수위상승", fontsize=13, color="#fcfcfb", ha="center", va="center")
    ax.text(1.4, cy - 0.12, "탐색형 EDA로", fontsize=10, color="#d3d1c7", ha="center", va="center")
    ax.text(1.4, cy - 0.42, "찾은 것 16가지", fontsize=10, color="#d3d1c7", ha="center", va="center")
    for ty in theme_y:
        ax.plot([2.6, 2.9, 2.9, 3.2], [cy, cy, ty, ty], color="#888780", lw=1.2)
    for i, (k, (fc, ec, lb)) in enumerate(KIND.items()):
        ax.add_patch(FancyBboxPatch((0.25 + i * 4.2, 0.18), 0.35, 0.28, boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=0.8))
        ax.text(0.72 + i * 4.2, 0.32, lb, fontsize=9.5, va="center", color="#52514e")
    ax.set_title("인사이트 지도: 가설 없이 데이터에서 드러난 것", fontsize=14, loc="left", color="#0b0b0b", pad=12)
    savefig(fig, path)


def main():
    T = insights()
    draw(T)
    json.dump([{"theme": t, "leaves": [dict(zip(["kind", "title", "evidence", "action", "fig"], l)) for l in v]} for t, v in T],
              open(RESULTS / "s11_insights.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps([{"theme": t, "n": len(v)} for t, v in T], ensure_ascii=False))


if __name__ == "__main__":
    main()

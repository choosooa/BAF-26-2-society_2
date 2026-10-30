"""eda.ipynb 생성. 캐시가 있으면 결과만 보여주고, 없으면 해당 단계를 실행한다."""
import nbformat as nbf

from common import EDA

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# 하천수위 6시간 최대상승량 — 사전점검 + EDA 노트북

- 무거운 계산은 `eda/src/sNN_*.py`에 있고, 이 노트북은 **캐시가 있으면 결과만 불러와** 보여줍니다.
- 전부 다시 계산하려면 아래 `FORCE = True`로 바꾸고 실행하세요 (총 약 5분, 최대 메모리 ~0.9GB).
- 해설 요약은 `eda/archive/report.md` 참고.""")
code("""import sys, json
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import pandas as pd
from IPython.display import Image, display, Markdown
from common import RESULTS, FIG, CACHE, load_json

FORCE = False
def ensure(result_file, module):
    if FORCE or not (RESULTS / result_file).exists():
        __import__(module).main()
    return result_file
def fig(name, width=950):
    display(Image(filename=str(FIG / name), width=width))
pd.set_option('display.float_format', lambda v: f'{v:,.4f}')""")

md("## S0. 적재·프로파일\nCSV를 한 번만 읽어 float32 parquet로 저장 (train 231MB → 122MB).")
code("""if FORCE or not (CACHE / 'train.parquet').exists():
    __import__('s00_ingest').main()
p = load_json('s00_profile.json')
pd.DataFrame({k: {m: p[k][m] for m in ['shape','n_null','n_dup_rows','n_dup_station_time','n_station','n_timestamp','ts_min','ts_max']} for k in ['train','test']})""")
code("""print('타깃 부호 비율:', p['train']['target'])
pd.DataFrame(p['train']['describe']).T[['mean','std','min','50%','99%','max']]""")

md("## S1. 정합성 검산 + 호우사상 복원\n모든 후속 단계의 토대. lag 일치, 강우 누적 일치, 타깃 재계산을 확인하고 사상 블록을 복원합니다.")
code("""ensure('s01_integrity.json', 's01_integrity')
r = load_json('s01_integrity.json')
display(pd.DataFrame({n: {f'lag{k}h 일치율': r[n]['continuity'][f'lag{k}h']['match_rate_when_exists'] for k in [1,3,6,12,24]} for n in ['train','test']}))
display(pd.DataFrame({f'{s}_{w}': {n: r[n]['rain_accum'][s][f'sum1h_vs_{w}']['match_rate'] for n in ['train','test']} for s in ['imerg','sfc'] for w in ['6h','24h']}))
print('사상 블록:', {k: r['blocks'][k] for k in ['n_blocks_total','n_train','n_test']})
print('train 타깃 재계산:', r['train']['target_recompute'])
print('test 내부 미래 6행 존재 비율 (누수 위험, 모델 사용 금지):', r['test']['future_rows_in_test'])""")
code("fig('02d_event_blocks.png')")

md("## S2. 구조 도식\n시간창, 병합 구조, 관측소×사상 커버리지.")
code("""ensure('s02_structure.json', 's02_structure')
for f in ['02a_time_window.png', '02b_merge_structure.png', '02c_coverage_heatmap.png']:
    fig(f)
load_json('s02_structure.json')""")

md("## S3. 일반 EDA")
code("""ensure('s03_eda.json', 's03_eda')
e = load_json('s03_eda.json')
print(json.dumps(e['target'], ensure_ascii=False, indent=1))
fig('03a_target_dist.png')""")
code("fig('03b_input_dists.png'); fig('03c_env_station.png')")
code("fig('03d_target_vs_rain.png'); fig('03e_heteroscedasticity.png', 650)")
code("fig('03f_target_vs_wl.png'); fig('03g_spearman.png', 850)\npd.Series(e['spearman_with_target']).to_frame('Spearman(타깃)')")
code("fig('03h_rain_sources.png')\npd.DataFrame({h: e['rain_sources'][h] for h in ['1h','6h','24h']})")
code("fig('03i_env_moderation.png'); fig('03j_station_event_hetero.png'); fig('03k_target_by_event_pos.png', 700)\ne['variance_decomp']")

md("## S4. 수문 분석 — 조석 관측소 발견, 수문곡선, 지체시간")
code("""ensure('s04_hydro.json', 's04_hydro')
h = load_json('s04_hydro.json')
print({k: h['tidal'][k] for k in ['rule','n_tidal','tidal_sse_share(0pred)','tidal_row_share','top10_sse_that_are_tidal']})
display(pd.DataFrame(h['tidal']['tidal_stations']).T)
fig('04a_tidal_detection.png'); fig('04b_tidal_examples.png')""")
code("fig('04c_hydrographs.png'); fig('04d_lag_time.png')\nh['lag']")

md("## S5. 기준선 — 사상 단위 GroupKFold vs 무작위 KFold")
code("""ensure('s05_baseline.csv', 's05_baseline')
b = pd.read_csv(RESULTS / 's05_baseline.csv')
display(b.pivot(index='method', columns='split', values='rmse').sort_values('GroupKFold(사상)', ascending=False))
display(b[['split','method','rmse','fold_std','rmse_tidal','rmse_nontidal']])
load_json('s05_extra.json')""")

md("## S6. 적대적 검증 — train과 test가 구분되는가")
code("""ensure('s06_adversarial.json', 's06_adversarial')
a = load_json('s06_adversarial.json')
print('AUC:', round(a['auc_groupkfold'], 3))
fig('06a_adversarial.png')
pd.Series(a['ks_top']).to_frame('KS 통계량')""")

md("""# Part 2. 탐색형 인사이트 발굴 (S8~S11)
가설을 먼저 세우지 않고, 데이터가 스스로 드러내는 구조를 찾는다. 요약과 해설은 `archive/insights.md` 참고.""")

md("## S8. 관측소 행동 지문 → 비지도 유형 + 데이터 이상")
code("""ensure('s08_fingerprint.json', 's08_station_fingerprint')
s8 = load_json('s08_fingerprint.json')
display(pd.read_csv(RESULTS / 's08_type_profile.csv', index_col=0).T)
print('유역환경으로 유형 맞히기:', s8['env_explains_type'])
fig('08a_station_types.png'); fig('08b_type_profile.png'); fig('08c_type_signatures.png'); fig('08d_data_quirks.png')
print('주기성 분류:', {k: s8['periodic'][k] for k in ['strong_tidal', 'weak_tidal', 'daily_only']})""")

md("## S9. 좌표 없이 복원한 관측소 연결망")
code("""ensure('s09_network.json', 's09_network')
s9 = load_json('s09_network.json')
print(json.dumps(s9['network'], ensure_ascii=False, indent=1))
fig('09a_station_network.png'); fig('09b_lead_lag_example.png')
pd.DataFrame(s9['value_check']).T""")

md("## S10. 강우 시간 정렬 · 사상 해부 · 오차 지도 · 미래강우의 몫")
code("""ensure('s10_events_residuals.json', 's10_events_residuals')
s10 = load_json('s10_events_residuals.json')
fig('10a_rain_alignment.png')
print(json.dumps(s10['alignment']['train'], ensure_ascii=False, indent=1)[:2500])""")
code("""fig('10b_event_anatomy.png'); fig('10c_residual_map.png')
display(pd.DataFrame(s10['residuals']['regimes']).T)
{k: s10['events'][k] for k in ['n_events_for_half_sse', 'top5_sse_base_share', 'top5_row_share', 'test_events_above_train_p90_peak']}""")
code("""fig('10d_future_rain.png')
print(json.dumps(s10['future_rain'], ensure_ascii=False, indent=1)); s10['gaps']""")

md("## S11. 인사이트 지도")
code("""ensure('s11_insights.json', 's11_insight_map')
fig('11_insight_map.png', 1100)""")

md("## 부록. EDA 반영 인과 지도")
code("""if FORCE or not (FIG / '07_dag_updated.png').exists():
    __import__('s07_dag').main()
fig('07_dag_updated.png')""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, EDA / "archive" / "eda.ipynb")
print("wrote", EDA / "archive" / "eda.ipynb")

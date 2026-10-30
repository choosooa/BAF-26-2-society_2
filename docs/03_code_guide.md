# 03. 코드 가이드

두 노트북 모두 claude.ai에서 작성. 피처 실험 노트북은 실제 데이터로 실행 완료(결과는 docs/02 §5).
검증 환경: Python 3, pandas 3.0.2, scikit-learn 1.8.0, lightgbm 4.7.0. 실행 전 `%pip install lightgbm` 후 커널 재시작.

**데이터 경로**: 두 노트북 STEP 0은 실행 위치에서 `data/`를 찾고, 없으면 상위 폴더를 본다. 레포 루트·`notebooks/` 어디서 실행해도 됨.
```python
ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR.parent / "outputs"
```

## 1. `notebooks/00_baseline.ipynb` (공통 베이스라인, 재혁 관리)
원래 이름 `hou_baseline_v1.ipynb`. 현재 수아 로컬에는 없음.

| STEP | 내용 |
|---|---|
| 0 | 설정. `DATA_DIR`, `OUT_DIR`, `SEED=42`, `N_FOLDS=4`, 기대 행·사상 수 |
| 1 | 로드. `find_file()`·`read_table()`로 sample_submission·data_dictionary를 확장자(csv/xlsx)·인코딩(utf-8-sig/cp949) 상관없이 읽음. timestamp 파싱, 행 수 확인, test 관측소가 train에 있는지 확인 |
| 2 | `add_event_id(df, prefix, gap_h=1)`: 고유 timestamp를 정렬해 1시간 넘게 끊기는 지점에서 사상 분리. `TR_E000`, `TE_E000` 형식. 사상 수가 122/30인지 출력 |
| 3 | 전량 0 예측 RMSE (주최측 0.410 재현 확인), 사상별 분포 |
| 4 | `make_features(df, station_stats)` (아래 표). `station_stats`는 train의 관측소별 `wl_t` 중앙값·IQR |
| 4-1 | 누수 자가검증: 사상 마지막 6시간 행 삭제 후 피처 재계산 → 남은 행 값이 같은지 `np.isclose`로 비교. "미래 의존 피처: 없음 ✅"이어야 함 |
| 5 | LightGBM 사상 단위 GroupKFold 4-fold. `USE_STATION_CAT=True`면 station_id 범주형. early stopping 200, 최대 5000 라운드. `run_cv(features)` 반환: (score, oof, models) |
| 5-1 | 사상별 RMSE, 상위 5개 사상 오차제곱합 비중, gain 기준 피처 중요도 |
| 5-2 | (선택, `RUN_ABLATION=False`) 소스별 누적 ablation: 수위만 → +IMERG → +SFC → +레이더 → +유역환경 → +동시각 공간 |
| 6 | fold 모델 평균으로 test 예측 → sample_submission 순서로 병합 → 행 수·row_id·결측 assert → `outputs/submission_lgb_cv{점수}.csv` |

파라미터: `objective=regression, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42`

### `make_features` 피처 (원본 22개 입력 + station_id 외 추가분)
모든 이전 행 참조는 `(station_id, event_id)` 그룹 안에서 shift하고, **정확히 k시간 전 행일 때만** 값을 씀(`lag()` 내부 함수).

| 그룹 | 피처 |
|---|---|
| A. 수위 변화 | `dwl_{1,3,6,12,24}h`, `accel_wl`, `past_maxrise_6h`(직전 6h 최고 수위 − 6h 전 수위), `wl_below_past_max` |
| B. 강우 이력 | `imerg_rain_1h`·`sfc_rain_1h`·`radar_reflectivity_mean`·`radar_reflectivity_ge20_fraction`의 `_lag1~3`, `imerg_trend_1h`, `sfc_trend_1h`, `{imerg,sfc}_front_1_6`, `_front_6_24`, `_rain_7_24h`, `_event_cum`, `_event_max1h`, `imerg_minus_sfc_6h` |
| C. 사상 맥락 | `event_elapsed_h`(사상 시작 이후 경과시간), `wl_since_event_start` |
| D. 동시각 공간 | 같은 사상·같은 시각 전체 관측소 평균: `xs_mean_{imerg_rain_1h, imerg_rain_6h, sfc_rain_6h, dwl_1h, dwl_3h, dwl_6h}`, `xs_rising_frac`, `dwl_3h_vs_xs` |
| E. 관측소 정규화 | `st_wl_med`, `st_wl_iqr`, `wl_rel`, `dwl_{1,3,6}h_scaled` |

D 그룹은 Q&A 10/6 (나)로 규정상 허용. 절대 수위가 아닌 변화량·강우만 평균함. "실제 운영 시 모든 관측소 자료가 같은 시각에 들어온다"는 가정은 방법론 문서에 명시.

## 2. `notebooks/suah_feature_exp.ipynb` (수아 피처 실험)
원래 이름 `hou_feature_exp_v1.ipynb`. 베이스라인의 STEP 0–2, `make_features`, 누수 검증 방식을 그대로 가져와 실험만 추가.

| STEP | 내용 |
|---|---|
| 0 | 베이스라인 설정 + `RESULT_DIR = outputs/feature_exp_v1`, `USE_HOLDOUT=True`, `HOLDOUT_FRAC=0.15`, `MIN_GAIN=0.0005`, `MIN_FOLDS_IMPROVED=3` |
| 1–2 | 베이스라인과 동일 |
| 3 | 사상 단위 홀드아웃 분리(seed 42, 18개). `station_stats`도 dev 사상으로만 계산 |
| 4 | 베이스라인 피처 (`BASE_FEATURES`) |
| 5 | `add_exp_features(df)`: 실험 피처 생성. 접두사 `x1_`, `x2_`, `x3_`. 베이스라인과 이름 겹침 검사 |
| 5-1 | 누수 자가검증 (베이스라인 + 실험 피처 전체) |
| 6 | 고정 조건: `FOLDS`를 한 번 만들어 모든 실험이 공유. `run_cv(features, name)`이 OOF를 `oof_{name}.csv`로 저장 |
| 7 | 기준 → 1 → 2 → 3 실행 |
| 7-1 | 채택 기준 통과 묶음만 모아 "최종" 실행 (1개면 그 결과 재사용, 0개면 기준 유지) |
| 7-2 | 결과 표·사상별 오차 저장 |
| 7-3 | 우리 OOF 기준 상위 5·10개 사상 오차제곱합 비중, 상위 10개 사상의 실험별 RMSE, 나머지 사상 평균 변화 |
| 8 | 홀드아웃 확인 (`RUN_HOLDOUT=False`, 최종 확정 후 한 번만 True). dev 전체 학습, 반복 수는 CV best_iteration 평균 |
| 9 | `final_features.json` 저장 (채택 묶음, 피처 목록, 파라미터, 홀드아웃 사상) |

### 실험 피처
| 실험 | 피처 | 베이스라인과의 관계 |
|---|---|---|
| 1 수위 | `x1_step_t1~t5`(k시간 전의 1시간 변화량), `x1_dwl_{2,4,5}h`, `x1_max_step_6h`, `x1_rise_streak`(현재부터 연속 상승 시간) | 베이스라인은 2·4·5h 전 수위를 `past_maxrise_6h` 계산에만 사용 |
| 2 누적 강우 | `x2_{imerg,sfc}_rain_{2h,3h,12h,13_24h}`, `x2_radar_mean_3h` | 12h = 현재 6h 누적 + 6시간 전 행의 6h 누적 |
| 3 이동평균 | `x3_{imerg,sfc,radar,dwl1}_ewm{3,12}` (반감기 3h·12h) | 단순 이동평균은 실험 2 누적값을 시간으로 나눈 것과 같아 EWM으로 구성 |

### 산출물 (`outputs/feature_exp_v1/`)
| 파일 | 내용 | git |
|---|---|---|
| `experiment_summary.csv` | 실험별 피처 수, OOF RMSE, 기준 대비 변화(음수=개선), 개선 fold 수, fold RMSE, 실행 시간 | 올림 |
| `event_rmse.csv` | 실험 × 사상별 RMSE, SSE, 행 수 | 올림 |
| `final_features.json` | 한동에게 넘길 최종 피처·파라미터 | 올림 |
| `holdout_events.csv`, `holdout_check.csv` | 홀드아웃 사상 목록, 확인 결과 | 올림 |
| `oof_*.csv` | 실험별 OOF (row_id, oof), 파일당 약 22MB | 드라이브 (gitignore) |

홀드아웃이 빠져 있어 이 노트북의 CV 점수는 베이스라인 노트북(122개 사상 전체)과 직접 비교하지 않는다.

## 3. 미해결 수정 사항 (우선순위 순)

### ✅ fold를 공통 파일로 교체 (10/30 완료)
`suah_feature_exp.ipynb` STEP 3·6이 `outputs/common/event_folds.csv`를 읽음. 사상별 행 수 assert로 사상 ID 복원이 팀과 같은지 확인. 1차 결과와 분할이 같아 재실행 불필요.

### 🔴 `00_baseline.ipynb`는 아직 자체 GroupKFold (122개 전체)
재혁에게 공통 파일 사용 또는 "전체 참고용" 표시 요청.

### 🔴 실제 데이터 실행 확인 (베이스라인)
- STEP 2 사상 수가 train 122 / test 30인지. README상 "사상 간 시간 관계는 유지되지 않음"이라 사상 사이 간격이 우연히 1시간이면 두 사상이 합쳐질 수 있음. 숫자가 다르면 `gap_h` 조정 또는 `wl_t_minus_1h`와 이전 행 `wl_t`의 연속성으로 추가 검증
- 누수 자가검증 "없음 ✅" 확인

### 🟡 `imerg_minus_sfc_6h`
mm인 IMERG에서 단위 미확인인 sfc를 빼는 피처라 방법론 문서에서 의미를 설명하기 어려움. 선택지: 삭제(권장) 또는 비율로 대체. 바꾸면 베이스라인과 실험 노트북 모두 수정하고 한동에게 피처 목록 변경을 알릴 것.

### 🟡 기타
- 베이스라인의 `station_stats`는 train 전체로 계산(입력값 통계라 영향은 작음). 실험 노트북은 dev만 사용
- early stopping이 검증 fold 기준이라 OOF가 약간 낙관적. 최종 판단은 홀드아웃으로
- 감조 관측소 플래그 미구현 (팀 기존 목록 필요)

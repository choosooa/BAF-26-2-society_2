# eda/ 폴더 안내

KISTI 2026 — 호우 시 6시간 최대 하천수위 상승량 예측, 탐색적 데이터 분석(EDA)

## 먼저 볼 것
| 파일 | 내용 |
|---|---|
| `report_final.md` | **최종 EDA 보고서** (0장 원칙 + 8장 + 부록 A~F) |
| `EDA_중간보고서.md` | **중간보고서용** EDA 여덟 덩어리 + 향후 계획 (그림 `figures/midterm/`, 재현 `src/m01_midterm_figs.py`) |
| `research_findings.md` | 선행연구 가설 42개 판정 결과 |
| `AGENT_REFERENCE.md` | 확인된 사실 요약 (분석 시작 전 참고용) |

## 노트북
| 파일 | 내용 |
|---|---|
| `nb1` ~ `nb6` | 데이터 구조·품질 → 타깃·강우 → 수위·시간 → 관측소 유형·유역 → 연결망 → 검증·오차 |
| `nb5b_network_visualization` | 관측소 연결망 시각화 |
| `nb7` ~ `nb10` | 선행연구 가설 확인 (체제·수위-타깃·도달시간·조절변수) |
| `tidal_analysis` | 조석 관측소 별책 |
| `self.ipynb` | 개인 작업 노트북 |

## 기타 문서
- `textbook_plan.md`, `research_eda_plan.md` — 분석 계획서
- `event_length.md` — 이벤트 길이 분석
- `EDA_옵시디언.md` — 옵시디언 공유용 종합본

## 폴더
- `src/` — 계산 스크립트 (`s00_ingest.py`가 `data/` CSV → `cache/*.parquet` 생성)
- `cache/` — parquet 캐시, `results/` 수치 JSON
- `figures/` — 보고서 그림
- `archive/` — 이전 버전 보고서(`report.md`, `insights.md`, `report_textbook.md`)와 초기 노트북 `eda.ipynb`. `report_final.md`로 대체됨

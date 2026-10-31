# BAF-26-2-society_2 — 호우날두

2026 KISTI DATA·AI 분석 경진대회 「호우 시 유역환경을 반영한 6시간 최대 하천수위 상승량 예측」

## 폴더 구조
```
repo/
├─ README.md
├─ CLAUDE.md                  # 작업 규칙·누수 규정 요약
├─ requirements.txt
├─ docs/                      # 대회 규정, 팀 상태, 코드 가이드, 실험 규칙, 발표 자료
├─ data/                      # git 제외. 대회 원본을 각자 받아서 넣기
├─ eda/                       # EDA 스크립트·분석 자료
├─ notebooks/
│  ├─ 00_baseline.ipynb       # 공통 (재혁 관리)
│  ├─ eda_ver1.ipynb          # EDA
│  ├─ suah_feature_exp.ipynb  # 수아 실험
│  ├─ handong_tuning.ipynb    # 한동 실험
│  └─ jaehyuk_pipeline.ipynb  # 재혁: 공통 fold 생성·통합 파이프라인
└─ outputs/
   ├─ common/                 # event_folds.csv, holdout_events.json (재혁 관리, 수정 금지)
   └─ <이름>/                 # 개인 결과. experiments.csv 등 작은 파일만 커밋, OOF csv는 드라이브
```

## 실행 방법
1. `data/`에 `train.csv.gz`, `test.csv.gz`, `sample_submission.csv`, `data_dictionary.csv`를 넣는다.
2. 노트북은 **`notebooks/` 폴더 또는 레포 루트에서** 실행한다. 노트북이 현재 위치에서 `data/`를 찾고, 없으면 상위 폴더에서 찾는다.
3. 결과는 `outputs/` 아래에 저장된다.

## 작업 방식: 각자 RMSE를 낮추고, 결과와 이유를 공유
자세한 규칙은 [docs/04_experiment_rules.md](docs/04_experiment_rules.md).

1. **전원이 각자 모델링한다.** 피처·모델·튜닝·앙상블 모두 자유. 역할로 나누지 않는다.
2. **같은 잣대로 비교한다.** `outputs/common/event_folds.csv`의 공통 fold, dev 사상 104개, dev OOF RMSE. 공통 기준은 0.2182.
3. **낮아지면 채택, 아니면 기각.** 기준: OOF RMSE 0.0005 이상 감소 **그리고** 4개 fold 중 3개 이상 개선.
4. **채택·기각 모두 기록하고 이유를 설명한다.**
   - `outputs/<이름>/experiments.csv`에 한 줄 (가설, 변경점, 결과, 판정, 이유)
   - 팀 채팅에 요약 (기준 → 결과, 채택/기각, 이유). 하루 한 번 이상
   - 시작 전에 "지금 이걸 해봄" 한 줄 공지해서 중복을 막는다
5. **11/3에 합친다.** 각자 채택분(코드·파라미터·결과 포함)을 재혁에게 넘기고, 재혁이 하나씩 얹으며 재검증한다. **홀드아웃은 개인이 열지 않고**, 팀 최종 조합 1개만 재혁이 한 번 확인한다.

새 피처를 만들면 **누수 자가검증**(사상 마지막 6시간 삭제 후 피처 재계산 → 값 동일 여부)을 통과해야 한다.

## git 규칙
- 재혁이 공통 구조·공통 파일을 main에 올리고, 각자 최신 main을 받은 뒤 브랜치를 만든다.
  ```
  git switch main
  git pull upstream main        # 팀 레포(DGU-BAF) 기준
  git switch -c feature/<이름>-<작업>
  ```
- 각자 **자기 노트북·자기 `outputs/<이름>/`만** 수정하고 자기 브랜치에 push → PR → 재혁이 확인 후 main 머지. 공통 파일(`00_baseline`, `outputs/common/`, `requirements.txt`)은 재혁만 수정.
- **fold는 각자 만들지 않는다.** 공통 파일을 읽어서 쓴다.
- 커밋 전 노트북 출력 지우기(`nbstripout` 권장), `data/`·OOF csv는 올리지 않음.
- 11/3 통합 때 `.py` 분리는 필수가 아니다. 중복 핵심 함수만 필요한 만큼 분리하거나, 처음부터 끝까지 돌아가는 최종 노트북 하나를 확보한다.

## 일정
| 날짜 | 할 일 |
|---|---|
| 10/31 ~ 11/2 | 각자 모델링, 하루 한 번 공유 |
| 11/3 | 채택분 통합, 최종 조합 확정, 홀드아웃 1회 |
| 11/4 | 전체 재현 실행 |
| 11/5 | 방법론 문서, 제출 완료 |
| 11/6 | 여유일 |

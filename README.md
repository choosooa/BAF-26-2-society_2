# BAF-26-2-society_2 — 호우날두

2026 KISTI DATA·AI 분석 경진대회 「호우 시 유역환경을 반영한 6시간 최대 하천수위 상승량 예측」

## 폴더 구조
```
repo/
├─ README.md
├─ CLAUDE.md                  # 작업 규칙·누수 규정 요약
├─ docs/                      # 대회 규정, 팀 상태, 코드 가이드, 발표 자료
├─ data/                      # git 제외. 대회 원본을 각자 받아서 넣기
├─ notebooks/
│  ├─ 00_baseline.ipynb       # 공통 (재혁 관리)
│  ├─ eda_ver1.ipynb          # EDA
│  ├─ suah_feature_exp.ipynb  # 수아: 수위·강우 피처 실험
│  ├─ handong_tuning.ipynb    # 한동: 모델 비교·튜닝
│  └─ jaehyuk_pipeline.ipynb  # 재혁: 통합 파이프라인
└─ outputs/                   # 작은 결과(summary csv, json)만 커밋. OOF csv는 드라이브
```

## 실행 방법
1. `data/`에 `train.csv.gz`, `test.csv.gz`, `sample_submission.csv`, `data_dictionary.csv`를 넣는다.
2. 노트북은 **`notebooks/` 폴더 또는 레포 루트에서** 실행한다. 노트북이 현재 위치에서 `data/`를 찾고, 없으면 상위 폴더에서 찾는다.
3. 결과는 `outputs/` 아래에 저장된다.

## 협업 규칙
- 재혁이 폴더 구조·공통 베이스라인·`.gitignore`를 main에 먼저 올리고, 각자 최신 main을 받은 뒤 브랜치를 만든다.
  ```
  git switch main
  git pull upstream main        # 팀 레포(DGU-BAF) 기준. 포크를 쓰지 않으면 origin
  git switch -c feature/<이름>-<작업>
  ```
- 각자 자기 노트북만 수정하고 자기 브랜치에 push → PR → 재혁이 확인 후 main 머지. 공통 파일은 재혁만 수정.
- **fold는 각자 만들지 않는다.** 재혁이 확정한 `row_id`·`event_id`·fold 매핑과 홀드아웃 목록을 공유 파일로 받아 쓴다.
- 피처를 넘길 때는 ① 채택 피처 목록 ② 피처 생성 함수(코드) ③ 비교 결과·기준 파라미터를 같이 전달한다.
- 커밋 전 노트북 출력 지우기(`nbstripout` 권장), `data/`·OOF csv는 올리지 않음.
- 11/3 통합 때 `.py` 분리는 필수가 아님. 중복 핵심 함수만 필요한 만큼 분리하거나, 처음부터 끝까지 돌아가는 최종 노트북 하나를 확보한다.

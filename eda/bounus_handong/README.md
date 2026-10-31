# local_handong — 한동 로컬 전용

이 폴더는 `.git/info/exclude`에 등록되어 있어 git이 추적하지 않는다. 커밋·push 대상이 아니고, `git pull`이나 브랜치 전환을 해도 그대로 남는다.

- 공용 `.gitignore`는 건드리지 않았다(재혁 관리 파일이라 수정하면 커밋할 변경이 생긴다).
- 주의: 팀이 나중에 `local_handong/` 경로에 파일을 추적하기 시작하면, ignore된 파일이 pull 때 덮어써질 수 있다. 이 폴더 이름은 팀 구조에 없는 이름이라 그럴 일은 거의 없다.

## 구성

| 경로 | 내용 |
|---|---|
| `docs/` | 노트 4편 사본 + `images/` 그림 3장. **원본은 옵시디언 볼트 `수위예측/`**이다. 고칠 때는 볼트를 고친다 |
| `scripts/resid_eda.py` | 기준 모델 잔차 EDA: 상황·구간·사상별 오차, 보정, 사상×시각 공통 몫 → `cache/resid_eda.json`, `cache/resid.parquet` |
| `scripts/nb3.py` | 강우 격자 이웃 정의(전 관측소 공통 시각 해시)와 이웃 공통 오차·잔차 상관 → `cache/nb3.json`, `cache/cells_*.csv` |
| `scripts/nb2.py` | nb3의 이전 버전(결측 시각을 −1로 채운 해시). 반박 4번 비교용 |
| `scripts/figs.py` | 그림 3장 → `docs/images/` (인자로 다른 폴더 지정 가능) |
| `scripts/snips.py`, `test_snip.py` | 수아 참고자료의 코드 뼈대와 누수 자가검증 테스트 |
| `cache/` | 위 스크립트의 중간 산출물 |

## 실행

레포 루트에서 실행한다. `data/train.csv`와 `outputs/handong_tuning/oof_기준.csv`가 있어야 한다.

```bash
python3 local_handong/scripts/resid_eda.py
python3 local_handong/scripts/nb3.py
python3 local_handong/scripts/figs.py
python3 local_handong/scripts/test_snip.py
```

미래 강우 진단(전략용 EDA 4.5절)은 스크립트 없이 세션에서 한 번 계산했다. 다시 필요하면 `resid.parquet`에 같은 관측소 t+1~t+6 행의 `sfc_rain_1h` 합을 붙여 잔차와 비교하면 된다. 이 값은 진단 전용이며, 피처로 쓰면 누수다.

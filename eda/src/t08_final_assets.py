"""최종 EDA 보고서(report_final.md) 보조 자산.

1) 7~8권 노트북에 저장된 그림을 figures/final/ 로 꺼낸다.
2) 부록 C(구간 플래그)·D(단조 방향) 표를 결과 JSON에서 md로 만든다 → figures/final/*.md
수치를 손으로 옮기지 않기 위한 스크립트다. 실행: python src/t08_final_assets.py (eda/ 에서)
"""
import base64
import json
from pathlib import Path

EDA = Path(__file__).resolve().parents[1]
OUT = EDA / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
RES = EDA / "cache" / "results"

# (노트북, 셀 번호, 저장 이름) — 셀 번호는 이미지 출력이 있는 코드 셀
FIGS = [
    ("nb8_stage_and_target.ipynb", 8, "fig4-2b_falling_limb.png"),
    ("nb8_stage_and_target.ipynb", 12, "fig6-5b_event_stage.png"),
    ("nb7_regimes.ipynb", 9, "fig6-3b_dam_release.png"),
]


def extract_figures():
    for nb_name, cell_idx, out_name in FIGS:
        nb = json.loads((EDA / nb_name).read_text())
        cell = nb["cells"][cell_idx]
        pngs = [o["data"]["image/png"] for o in cell.get("outputs", [])
                if "image/png" in o.get("data", {})]
        assert pngs, f"{nb_name} 셀 {cell_idx}에 그림이 없다"
        (OUT / out_name).write_bytes(base64.b64decode(pngs[0]))
        print("그림:", out_name)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def regime_table():
    d = json.loads((RES / "nb07_summary.json").read_text())
    rows = []
    for r in d["regime_table"]:
        rows.append({
            "구간": r["regime"], "관측소": r["관측소"], "행": f'{r["행"]:,}',
            "행 비중": f'{r["행 비중"]:.1%}', "오차 비중": f'{r["오차 비중"]:.1%}',
            "RMSE(m)": f'{r["RMSE"]:.3f}', "편향(m)": f'{r["편향"]:+.4f}',
        })
    cols = ["구간", "관측소", "행", "행 비중", "오차 비중", "RMSE(m)", "편향(m)"]
    (OUT / "tabC_regimes.md").write_text(md_table(rows, cols))
    print("표: tabC_regimes.md")


def monotone_table():
    d = json.loads((RES / "nb10_summary.json").read_text())
    rows = []
    for r in d["b9"]:
        rows.append({
            "변수": r["변수"], "① 칸 순위상관": r["① 칸 순위상관"],
            "② 관측소 일치": f'{float(r["② 관측소 일치"]):.0%}',
            "③ 반대 부호 구간": ", ".join(eval(r["③ 반대 부호 구간"])) or "–",
            "④ 조건부 일치": f'{float(r["④ 조건부 일치"]):.0%}', "권고": r["권고"],
        })
    cols = ["변수", "① 칸 순위상관", "② 관측소 일치", "③ 반대 부호 구간", "④ 조건부 일치", "권고"]
    (OUT / "tabD_monotone.md").write_text(md_table(rows, cols))
    print("표: tabD_monotone.md")


if __name__ == "__main__":
    extract_figures()
    regime_table()
    monotone_table()

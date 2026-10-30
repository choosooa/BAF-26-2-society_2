"""별책: 관측소 연결망 시각화 (Network Visualization). 5권 계산을 다시 하고, 같은 연결망을 여러 모양으로 그린다."""
from nb_common import Book

b = Book()
b.md("""
# 🕸️ 별책 · Station Network Visualization — 관측소 연결망을 여러 모양으로 보기

> 시리즈: 5권 연결망의 **시각화 심화판**이에요. 계산 규칙(공통 신호 빼기 → 시차 상관 → 선행 관측소 판정)은 5권과 똑같고, 여기선 **그림만 여러 가지로** 그려요.

### 한 줄 요약
**같은 연결망도 어떻게 그리느냐에 따라 보이는 게 달라요. 유역면적을 세로축에 두면 "작은 유역 → 큰 유역" 흐름이 보이고, 사상을 반씩 나눠 겹쳐 그리면 튼튼한 뼈대(약 1/3)와 호우마다 바뀌는 살이 구분되고, 큰 호우 몇 개를 따로 그리면 선행 관계가 호우마다 얼마나 바뀌는지 눈으로 확인돼요.**

| 장 | 그림 | 무엇을 보나 |
|---|---|---|
| 1 | 5권 계산 다시 하기 | 선행 관측소 표 |
| 2 | 기본 힘-방향 그래프 | 전체 모양, 덩어리 |
| 3 | 유역면적 세로축 그래프 | 상류 → 하류 방향이 맞나 |
| 4 | 덩어리별 나무 그림 | 작은 덩어리 안의 층 구조 |
| 5 | 안정성 겹쳐 그리기 | 튼튼한 뼈대 vs 흔들리는 관계 |
| 6 | 허브와 영향 범위 | 누가 많은 관측소를 이끄나 |
| 7 | 상위 3개 선행 + 무리 나누기 | 한 명만 고르지 않으면 |
| 8 | 인접 행렬 히트맵 | 무리 안이 정말 빽빽한가 |
| 9 | 호(arc) 그림 | 방향을 한 줄로 |
| 10 | 모델 개선 위에 얹기 | 연결망 변수가 어디서 도움이 됐나 |
| 11 | 호우별 연결망 + 우연 기준선 | 호우마다 얼마나 바뀌나, 무엇이 반복되나 |
| 12 | 인터랙티브 그래프 | 마우스로 직접 둘러보기 |
| 13 | 믿을 수 있는 관계만 | 두 절반 모두에서 나온 선행 85쌍의 간격·방향·재등장 |
""")
b.setup(extra=r'''
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
FIG = EDA / "figures" / "network_viz"; FIG.mkdir(parents=True, exist_ok=True)
LAG_COL = {1: C_NT, 2: C_GREEN, 3: C_RAIN, 4: C_TIDAL}          # 선행 간격 색 (5h 이상은 보라)
lag_color = lambda L: LAG_COL.get(int(L), C_PURPLE)

def draw_edges(ax, G, pos, color=None, width=None, style="-", alpha=.8, rad=.08, z=2):
    """화살표 = 선행 → 후행. color/width는 (u, v, data) → 값 함수"""
    for u, v, d in G.edges(data=True):
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", mutation_scale=7, shrinkA=3, shrinkB=3,
                                     connectionstyle=f"arc3,rad={rad}", color=color(u, v, d) if color else C_ALL,
                                     lw=width(u, v, d) if width else .8, ls=style, alpha=alpha, zorder=z))

def packed_layout(G_, per_row=None, seed=3):
    """덩어리마다 따로 spring 배치 → 큰 덩어리부터 격자에 차곡차곡. 덩어리 크기 ∝ √노드 수"""
    cs = sorted(nx.weakly_connected_components(G_) if G_.is_directed() else nx.connected_components(G_), key=len, reverse=True)
    per_row = per_row or max(4, int(np.ceil(np.sqrt(len(cs)) * 1.3)))
    pos_, x0, y0, row_h, k_ = {}, 0., 0., 0., 0
    for c in cs:
        sub = G_.subgraph(c).to_undirected()
        p = nx.spring_layout(sub, seed=seed, k=1.2 / np.sqrt(len(c)), iterations=200) if len(c) > 2 else \
            {n_: np.array([i * .6, 0.]) for i, n_ in enumerate(c)}
        xy_ = np.array(list(p.values())); xy_ -= xy_.min(axis=0)
        span = max(np.ptp(xy_[:, 0]), np.ptp(xy_[:, 1]), 1e-9)
        size = .6 + .45 * np.sqrt(len(c)); xy_ = xy_ / span * size
        if k_ and (k_ % per_row == 0 or x0 > per_row * 1.6):
            x0, y0, row_h, k_ = 0., y0 - row_h - .6, 0., 0
        for n_, v in zip(p, xy_):
            pos_[n_] = (x0 + v[0], y0 - v[1])
        x0 += max(np.ptp(xy_[:, 0]), .3) + .6; row_h = max(row_h, np.ptp(xy_[:, 1])); k_ += 1
    return pos_

def lag_legend(ax, loc="lower left", title="화살표 색 = 선행 간격", **kw):
    h = [Line2D([], [], color=c, lw=2, label=f"{k}h 앞섬") for k, c in LAG_COL.items()] + [Line2D([], [], color=C_PURPLE, lw=2, label="5h 이상")]
    ax.legend(handles=h, loc=loc, fontsize=8, title=title, title_fontsize=8, **kw)
''')

# ── 1. 계산 ─────────────────────────────────────────────
b.md("""
## 1. 5권 계산 다시 하기

5권과 **같은 규칙**으로 선행 관측소를 찾아요. 짧게 복습하면:
1. 관측소마다 `dwl_1h`를 그 관측소의 전형적인 변동폭으로 나눠요(크기 맞추기).
2. 시각마다 전 관측소 중앙값을 빼요(전국이 같이 오르는 호우 신호 빼기).
3. 모든 쌍에 대해 시차 L = 0~8시간 상관을 재고, **같은 시각보다 0.05 이상 큰** 가장 센 선행을 고르고, 상관 0.5 이상이면 "뚜렷한 선행"이에요.
4. 조석·약한 조석·24h 주기·고정 관측소는 선행 후보와 후행 대상에서 빼요.

이번엔 그림에 쓰려고 **상위 3개 선행 후보**와 **사상 짝수/홀수 절반 결과**도 같이 저장해요.
""")
b.code(r'''
cols = ["station_id", "station_code", "hour", "event_block", TARGET, "dwl_1h", "wsarea", "stream_order", "mean_elevation", "urban_ratio"]
tr = add_flags(pd.read_parquet(CACHE / "train_sorted.parquet", columns=cols))
cats = sorted(tr.station_id.unique()); code_of = {s: i for i, s in enumerate(cats)}
env = tr.groupby("station_id")[["wsarea", "stream_order", "mean_elevation", "urban_ratio", "group"]].first()
print(f"train {len(tr):,}행 · 관측소 {len(cats)}곳 · 사상 {tr.event_block.nunique()}개")

hours = np.sort(tr.hour.unique()); hidx = pd.Series(np.arange(len(hours)), index=hours)
code = tr.station_id.map(code_of).to_numpy()
x = tr.dwl_1h.to_numpy(float)
g = pd.Series(x).groupby(code)
mad = g.apply(lambda v: np.median(np.abs(v - np.median(v)))) * 1.4826
scale = np.maximum(np.maximum(mad, .5 * g.std()), .005)
Zraw = np.full((len(hours), len(cats)), np.nan)
Zraw[hidx.reindex(tr.hour.to_numpy()).to_numpy(), code] = np.clip(x / scale.reindex(code).to_numpy(), -8, 8)
Z = Zraw - np.nanmedian(Zraw, axis=1, keepdims=True)

def masked_corr(A_, B_, min_n=150):
    MA, MB = ~np.isnan(A_), ~np.isnan(B_)
    A0, B0 = np.where(MA, A_, 0.), np.where(MB, B_, 0.)
    fa, fb = MA.astype(float), MB.astype(float)
    n_ = fa.T @ fb; sa, sb = A0.T @ fb, fa.T @ B0
    saa, sbb, sab = (A0 ** 2).T @ fb, fa.T @ (B0 ** 2), A0.T @ B0
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (sab - sa * sb / n_) / np.sqrt((saa - sa ** 2 / n_) * (sbb - sb ** 2 / n_))
    C[n_ < min_n] = np.nan
    return C

MAXLAG = 8
SPECIAL_CODES = np.array(sorted(code_of[s] for s in SPECIAL))

def lagged_corrs(Z_, row_ok=None, min_n=150):
    if row_ok is not None:
        Z_ = np.where(row_ok[:, None], Z_, np.nan)
    Cs = {}
    for L in range(0, MAXLAG + 1):
        i = np.arange(len(hours) - L)
        ok = hours[i + L] - hours[i] == L
        Cs[L] = masked_corr(Z_[i[ok]], Z_[i[ok] + L], min_n)
    return Cs

def find_leaders(Cs, exclude, k=1):
    """관측소(후행)마다 선행 후보 상위 k개. 반환: 긴 표 (follower, rank, leader, corr, lag)"""
    n_ = len(cats)
    stack = np.stack([Cs[L] for L in range(1, MAXLAG + 1)])
    stack = np.where(np.isnan(stack), -np.inf, stack)
    best, lag = stack.max(axis=0), stack.argmax(axis=0) + 1
    best[np.arange(n_), np.arange(n_)] = -np.inf; best[exclude, :] = -np.inf
    cand = np.where(best - np.nan_to_num(Cs[0], nan=0.) > .05, best, -np.inf)
    order = np.argsort(-cand, axis=0)[:k]                                  # (k, 후행)
    rows = []
    for r in range(k):
        for j in range(n_):
            i = order[r, j]; v = cand[i, j]
            if np.isfinite(v) and cats[j] not in SPECIAL:
                rows.append((cats[j], r + 1, cats[i], v, int(lag[i, j])))
    return pd.DataFrame(rows, columns=["follower", "rank", "leader", "corr", "lag"])

Cs = lagged_corrs(Z)
top3 = find_leaders(Cs, SPECIAL_CODES, k=3)
strong = top3[(top3["rank"] == 1) & (top3["corr"] >= .5)].set_index("follower")
considered = [s for s in cats if s not in SPECIAL]
print(f"대상 {len(considered)}곳 중 뚜렷한 선행: {len(strong)}곳 ({len(strong) / len(considered):.0%})")

blk_of_hour = tr.groupby("hour").event_block.first().reindex(hours).to_numpy()
half = {nm: find_leaders(lagged_corrs(Z, blk_of_hour % 2 == r), SPECIAL_CODES).set_index("follower")
        for nm, r in [("짝수", 0), ("홀수", 1)]}
strong.head()
''')

b.code(r'''
# 연결망 객체: 노드 = 관측소, 화살표 = 선행 → 후행
def build_graph(tab):
    G = nx.DiGraph()
    for f, r in tab.iterrows():
        G.add_edge(r.leader, f, corr=float(r["corr"]), lag=int(r.lag))
    for n_ in G.nodes:
        G.nodes[n_].update(ws=float(env.wsarea.get(n_, np.nan)), so=float(env.stream_order.get(n_, np.nan)),
                           elev=float(env.mean_elevation.get(n_, np.nan)), grp=env.group.get(n_, "?"))
    return G

G = build_graph(strong)
comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
comp_of = {n_: i for i, c in enumerate(comps) for n_ in c}
print(f"노드 {G.number_of_nodes()} · 화살표 {G.number_of_edges()} · 덩어리 {len(comps)}개")
print("큰 덩어리 크기 상위 10:", [len(c) for c in comps[:10]])
print(f"크기 2짜리(한 쌍) 덩어리: {sum(len(c) == 2 for c in comps)}개")
''')

# ── 2. 기본 ─────────────────────────────────────────────
b.md("""
## 2. 기본 힘-방향 그래프 — 전체 모양 보기

**힘-방향(spring) 배치**는 이어진 관측소끼리 끌어당기고, 안 이어진 관측소끼리는 밀어내서 자리를 잡아요. 좌표가 아니라 **"연결"만으로 만든 지도**예요.
그냥 전체를 한 번에 배치하면 작은 덩어리들이 바깥 고리로 밀려나고 큰 덩어리는 가운데서 뭉개져요. 그래서 **덩어리마다 따로 배치한 뒤 큰 것부터 격자에 채워** 넣었어요(왼쪽 위 = 가장 큰 덩어리).
- 노드 크기 = 상류면적(log), 노드 색 = 하천차수, 화살표 색 = 선행 간격, 화살표 굵기 = 상관 크기
- 5권 그림은 상위 12개 덩어리만 그렸는데, 여기선 **덩어리 전부**를 그려요.
""")
b.code(r'''
pos = packed_layout(G)
fig, ax = plt.subplots(figsize=(14, 10))
draw_edges(ax, G, pos, color=lambda u, v, d: lag_color(d["lag"]), width=lambda u, v, d: .5 + 3 * (d["corr"] - .5))
nodes = list(G.nodes); xy = np.array([pos[n_] for n_ in nodes])
so = np.array([G.nodes[n_]["so"] for n_ in nodes])
sc = ax.scatter(xy[:, 0], xy[:, 1], s=10 + 22 * np.log10(np.array([G.nodes[n_]["ws"] for n_ in nodes]) + 1),
                c=so, cmap="viridis", vmin=1, vmax=4, edgecolor="white", lw=.6, zorder=3)
plt.colorbar(sc, ax=ax, shrink=.5, label="하천차수 (클수록 큰 강)")
lag_legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax.set_axis_off(); ax.grid(False)
ax.set_title(f"전체 선행 연결망 · 덩어리 {len(comps)}개 (노드 크기 = 상류면적)"); fig.tight_layout()
fig.savefig(FIG / "01_spring_all.png", dpi=150, bbox_inches="tight"); plt.show()
''')
b.md("""
- 거대한 하나의 망이 아니라 **작은 덩어리 여러 개**로 쪼개져 있어요. 한 덩어리 ≈ 한 하천 구간이나 한 권역이라고 볼 수 있어요.
- 화살표는 대부분 **파랑(1시간)**이에요. 바로 위 관측소가 잡힌다는 5권 결론과 같아요.
- 큰 덩어리일수록 가지가 여러 갈래로 뻗은 **나뭇가지 모양**이에요. 가지 끝으로 갈수록 노드가 커지고 밝아지는(큰 유역·큰 차수) 곳이 많아요.
""")

# ── 3. 유역면적 세로축 ─────────────────────────────────
b.md("""
## 3. 유역면적 세로축 그래프 — 상류 → 하류가 보이나

힘-방향 배치는 세로·가로에 뜻이 없어요. 이번엔 **세로축 = 상류면적(log)**으로 고정하고, 가로축만 덩어리별로 나눠요.
- 진짜 상류 → 하류라면 화살표는 **위로(작은 유역 → 큰 유역)** 가야 해요.
- 🔵 파랑 = 위로 가는 화살표(예상대로), 🟠 주황 = 아래로 가는 화살표(예상과 반대), 회색 = 유역면적이 같은 쌍(같은 소권역)
""")
b.code(r'''
big = [c for c in comps if len(c) >= 3]
fig, ax = plt.subplots(figsize=(16, 7))
pos_v, x0, ticks = {}, 0., []
for c in big:
    sub = G.subgraph(c)
    px = nx.spring_layout(sub.to_undirected(), seed=1, k=.6)
    xs = np.array([px[n_][0] for n_ in c]); w = max(1.5, .35 * len(c))
    xs = (xs - xs.min()) / (np.ptp(xs) + 1e-9) * w
    for n_, xv in zip(c, xs):
        pos_v[n_] = (x0 + xv, np.log10(G.nodes[n_]["ws"]))
    ticks.append(x0 + w / 2); x0 += w + 1.2
H = G.subgraph(pos_v)
def dir_color(u, v, d):
    a, b_ = H.nodes[u]["ws"], H.nodes[v]["ws"]
    return C_ALL if np.isclose(a, b_) else (C_NT if b_ > a else C_TIDAL)
draw_edges(ax, H, pos_v, color=dir_color, width=lambda u, v, d: 1.1, rad=.15)
xy = np.array([pos_v[n_] for n_ in H.nodes])
ax.scatter(xy[:, 0], xy[:, 1], s=16, color="#5f5e5a", zorder=3)
cnt = pd.Series([dir_color(u, v, d) for u, v, d in H.edges(data=True)]).value_counts()
up, down = cnt.get(C_NT, 0), cnt.get(C_TIDAL, 0)
ax.set_xticks(ticks); ax.set_xticklabels([f"#{i + 1}\n({len(c)})" for i, c in enumerate(big)], fontsize=7)
ax.set_xlabel("덩어리 번호 (관측소 수)"); ax.set_ylabel("상류면적 log10(km²)")
ax.legend(handles=[Line2D([], [], color=C_NT, lw=2, label=f"작은 유역 → 큰 유역 ({up})"),
                   Line2D([], [], color=C_TIDAL, lw=2, label=f"큰 유역 → 작은 유역 ({down})"),
                   Line2D([], [], color=C_ALL, lw=2, label=f"같은 유역 ({cnt.get(C_ALL, 0)})")], loc="upper right", fontsize=8)
ax.set_title(f"세로축 = 유역 크기: 위로 가는 화살표 {up / (up + down):.0%} (무작위라면 50%)"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(FIG / "02_wsarea_vertical.png", dpi=150, bbox_inches="tight"); plt.show()
dlog = pd.DataFrame([{"방향": "위로" if G.nodes[v]["ws"] > G.nodes[u]["ws"] else "아래로", "lag": d["lag"], "corr": d["corr"],
                      "|Δlog10 면적|": abs(np.log10(G.nodes[v]["ws"]) - np.log10(G.nodes[u]["ws"]))}
                     for u, v, d in G.edges(data=True) if not np.isclose(G.nodes[u]["ws"], G.nodes[v]["ws"])])
dlog.groupby("방향").agg(화살표=("lag", "size"), 면적차_중앙값=("|Δlog10 면적|", "median"), 상관_중앙값=("corr", "median"),
                        h1_비율=("lag", lambda v: (v == 1).mean())).round(2)
''')
b.md("""
- 위로 가는 화살표가 주황보다 확실히 많아요. 5권 "뒤따르는 쪽이 더 큰 유역 약 2/3"를 **그림으로** 본 거예요.
- 주황(반대 방향) 화살표도 적지 않고, 표를 보면 면적 차이가 꼭 작은 것도 아니에요. 이런 쌍은 물이 흐르는 관계라기보다 **비가 먼저 닿은 순서**(비구름 이동 방향)일 가능성이 커요.
- 같은 덩어리 안에서도 파랑과 주황이 섞여 있어요. 덩어리 = 한 줄기 강이 아니라, "강 흐름 + 비 이동"이 섞인 묶음이에요.
""")

# ── 4. 덩어리별 나무 ───────────────────────────────────
b.md("""
## 4. 덩어리별 나무 그림 — 층 구조 보기

큰 덩어리 12개를 따로 떼어, **"선행이 없는 뿌리"에서 몇 단계 아래인가**로 층을 나눠 그려요(왼쪽 = 가장 먼저 움직이는 쪽). 층 번호가 곧 "몇 번 거쳐 전달되나"예요.
""")
b.code(r'''
def depth_layers(sub):
    """뿌리(들어오는 화살표 없음)에서의 최단 거리. 순환만 있어 뿌리가 없으면 출력이 가장 많은 노드를 뿌리로."""
    roots = [n_ for n_, d in sub.in_degree() if d == 0] or [max(sub.out_degree(), key=lambda t: t[1])[0]]
    dist = {}
    for r in roots:
        for n_, dd in nx.single_source_shortest_path_length(sub, r).items():
            dist[n_] = min(dist.get(n_, 99), dd)
    return {n_: dist.get(n_, 0) for n_ in sub.nodes}

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
depths = []
for ax, (i, c) in zip(axes.flat, enumerate(comps[:12])):
    sub = G.subgraph(c).copy()
    lay = depth_layers(sub); depths += list(lay.values())
    nx.set_node_attributes(sub, lay, "layer")
    p = nx.multipartite_layout(sub, subset_key="layer")
    draw_edges(ax, sub, p, color=lambda u, v, d: lag_color(d["lag"]), width=lambda u, v, d: .6 + 2.5 * (d["corr"] - .5), rad=.1)
    xy = np.array([p[n_] for n_ in sub.nodes])
    ax.scatter(xy[:, 0], xy[:, 1], s=8 + 20 * np.log10(np.array([sub.nodes[n_]["ws"] for n_ in sub.nodes]) + 1),
               c=[sub.nodes[n_]["so"] for n_ in sub.nodes], cmap="viridis", vmin=1, vmax=4, zorder=3, edgecolor="white", lw=.5)
    if len(c) <= 14:
        for n_ in sub.nodes:
            ax.annotate(n_[2:], p[n_], fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="#5f5e5a")
    ax.set_title(f"덩어리 #{i + 1} · {len(c)}곳 · 최대 {max(lay.values())}단계", fontsize=9); ax.set_axis_off()
lag_legend(axes.flat[0], loc="upper left")
fig.suptitle("큰 덩어리 12개: 왼쪽 = 뿌리(먼저 움직임) → 오른쪽 = 뒤따름 · 노드 색 = 하천차수", y=1.0)
fig.tight_layout(); fig.savefig(FIG / "03_component_trees.png", dpi=150, bbox_inches="tight"); plt.show()
print("뿌리에서의 단계 분포 (상위 12 덩어리):", pd.Series(depths).value_counts().sort_index().to_dict())
''')
b.md("""
- 관측소마다 선행을 **하나만** 고르니 모양은 나무(가지치기)예요. 작은 덩어리는 1~3단계, 큰 덩어리는 **5~8단계의 긴 사슬**도 있어요(예: #3, #6은 한 줄로 쭉 이어짐).
- 한 관측소가 여러 관측소를 이끄는 **부채꼴**(예: #4, #7, #10)도 흔해요.
- 긴 사슬의 끝까지 "1h + 1h + 2h …"로 더하면 앞날이 6시간을 넘는 경우도 있어요. **여러 단계 위 관측소**를 쓰면 6시간 예측창의 뒷부분도 보강할 수 있다는 아이디어가 나와요(검증 필요).
- 부채꼴의 가운데(뿌리)는 "그 권역에서 비에 **가장 먼저** 반응하는 관측소"로 볼 수 있어요. 꼭 최상류가 아니어도 돼요.
""")

# ── 5. 안정성 ──────────────────────────────────────────
b.md("""
## 5. 안정성 겹쳐 그리기 — 튼튼한 뼈대 vs 흔들리는 관계

5권에서 **사상을 짝수/홀수로 반씩 나눠** 따로 선행을 찾으면, 같은 선행이 나오는 곳이 약 1/3이었어요. 이걸 그래프로 겹쳐 그려요.
- 🔵 굵은 실선 = **두 절반 모두** 같은 선행(튼튼한 뼈대)
- 🟢 가는 선 = 짝수 절반에서만 · 🟠 가는 선 = 홀수 절반에서만
- 자리는 세 종류 화살표를 **모두 합친** 그래프로 다시 배치했어요(2장보다 선이 많아 덩어리가 더 커요).
""")
b.code(r'''
e_even = {(r.leader, f) for f, r in half["짝수"][half["짝수"]["corr"] >= .4].iterrows()}
e_odd = {(r.leader, f) for f, r in half["홀수"][half["홀수"]["corr"] >= .4].iterrows()}
e_both = e_even & e_odd
S = nx.DiGraph(); S.add_nodes_from(G.nodes)
for kind, es in [("even", e_even - e_both), ("odd", e_odd - e_both), ("both", e_both)]:
    for u, v in es:
        S.add_edge(u, v, kind=kind)
S.remove_nodes_from([n_ for n_ in list(S.nodes) if S.degree(n_) == 0])
pos_s = packed_layout(S)
fig, ax = plt.subplots(figsize=(14, 10))
sty = {"even": (C_GREEN, .6, .35), "odd": (C_TIDAL, .6, .35), "both": (C_NT, 2.2, .95)}
for kind in ["even", "odd", "both"]:
    sub = nx.DiGraph([(u, v, d) for u, v, d in S.edges(data=True) if d["kind"] == kind])
    c_, w_, a_ = sty[kind]
    draw_edges(ax, sub, pos_s, color=lambda *_: c_, width=lambda *_: w_, alpha=a_, z=2 + (kind == "both"))
xy = np.array([pos_s[n_] for n_ in S.nodes])
ax.scatter(xy[:, 0], xy[:, 1], s=12, color="#5f5e5a", zorder=4)
ax.legend(handles=[Line2D([], [], color=C_NT, lw=2.5, label=f"두 절반 모두 ({len(e_both)})"),
                   Line2D([], [], color=C_GREEN, lw=1, label=f"짝수 사상에서만 ({len(e_even - e_both)})"),
                   Line2D([], [], color=C_TIDAL, lw=1, label=f"홀수 사상에서만 ({len(e_odd - e_both)})")], loc="lower left", fontsize=9)
ax.set_axis_off(); ax.grid(False); ax.set_title("사상을 반씩 나눠 찾은 선행 관계 겹쳐 보기 (선행 상관 ≥ 0.4)")
fig.tight_layout(); fig.savefig(FIG / "04_stability_overlay.png", dpi=150, bbox_inches="tight"); plt.show()

stab = pd.DataFrame([{"종류": k, "화살표 수": len(es),
                      "1h 비율": np.mean([G.edges[e]["lag"] == 1 for e in es if G.has_edge(*e)]) if any(G.has_edge(*e) for e in es) else np.nan,
                      "후행이 더 큰 유역": np.mean([env.wsarea[v] > env.wsarea[u] for u, v in es if not np.isclose(env.wsarea[v], env.wsarea[u])])}
                     for k, es in [("두 절반 모두", e_both), ("한쪽에서만", (e_even ^ e_odd))]])
stab.round(2)
''')
b.md("""
- 두 절반을 합치면 선이 많아져 큰 덩어리 하나로 뭉쳐요. 그 안에서 굵은 파랑(뼈대)은 **일부 짧은 연결**뿐이고, 나머지는 초록·주황 한쪽에만 나타나요.
- 한쪽에서만 나온 선은 "강으로 이어졌다"기보다 "**그 절반의 호우들에서 비가 먼저 닿았다**"는 뜻일 가능성이 커요.
- 뼈대 쪽이 "후행이 더 큰 유역" 비율도 높아요(표). → 모델에 넣을 땐 **뼈대 화살표만** 쓰는 게 안전해요.
""")

# ── 6. 허브 ────────────────────────────────────────────
b.md("""
## 6. 허브와 영향 범위 — 누가 많은 관측소를 이끄나

- **직접 이끄는 수(출력 차수)**: 이 관측소를 선행으로 고른 관측소 수
- **영향 범위(후손 수)**: 화살표를 따라가면 닿는 관측소 수(간접 포함)
- **PageRank(뒤집은 그래프)**: 화살표를 거꾸로 돌려서 계산한 중요도. "많은 관측소, 그리고 중요한 관측소가 기대는 선행"일수록 커요.
""")
b.code(r'''
hub = pd.DataFrame({"직접 이끄는 수": dict(G.out_degree()),
                    "영향 범위": {n_: len(nx.descendants(G, n_)) for n_ in G.nodes},
                    "PageRank": nx.pagerank(G.reverse(), weight="corr")})
hub = hub.join(env[["wsarea", "stream_order", "group"]]).sort_values("PageRank", ascending=False)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
top = hub.head(15)
ax[0].barh(top.index[::-1], top["PageRank"][::-1], color=C_NT); ax[0].set_title("PageRank 상위 15 (선행으로서 중요도)")
ax[1].scatter(hub["직접 이끄는 수"] + np.random.default_rng(0).uniform(-.15, .15, len(hub)), hub["영향 범위"], s=18, color=C_NT, alpha=.6)
for n_, r in hub.head(6).iterrows():
    ax[1].annotate(n_, (r["직접 이끄는 수"], r["영향 범위"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("직접 이끄는 수"); ax[1].set_ylabel("영향 범위 (간접 포함)"); ax[1].set_title("직접 vs 간접")
leaders_ = hub[hub["직접 이끄는 수"] > 0]; followers_only = hub[hub["직접 이끄는 수"] == 0]
bins = np.linspace(np.log10(env.wsarea.min()), np.log10(env.wsarea.max()), 30)
ax[2].hist(np.log10(followers_only.wsarea), bins=bins, color=C_ALL, alpha=.6, label=f"따라가기만 ({len(followers_only)})")
ax[2].hist(np.log10(leaders_.wsarea), bins=bins, color=C_NT, alpha=.6, label=f"이끄는 쪽 ({len(leaders_)})")
ax[2].set_xlabel("상류면적 log10"); ax[2].set_ylabel("관측소 수"); ax[2].legend(fontsize=8); ax[2].set_title("이끄는 관측소는 유역이 작은가")
fig.tight_layout(); fig.savefig(FIG / "05_hubs.png", dpi=150, bbox_inches="tight"); plt.show()
print(f"상류면적 중앙값: 이끄는 쪽 {leaders_.wsarea.median():.0f} km² · 따라가기만 {followers_only.wsarea.median():.0f} km²")
hub.head(10).round(3)
''')
b.md("""
- 소수의 **허브**가 여러 관측소를 이끌어요. 허브가 결측이거나 이상하면 여러 관측소의 연결망 변수가 한꺼번에 흔들리니, **허브 품질 점검**이 중요해요.
- 이끄는 쪽의 유역이 따라가기만 하는 쪽보다 작은 경향이 있어요(상류일수록 먼저 반응).
""")

# ── 7. 상위 3 + 무리 ───────────────────────────────────
b.md("""
## 7. 상위 3개 선행 + 무리 나누기 — 한 명만 고르지 않으면

5권 결론: "한 관측소만 고르면 흔들리니 **주변 여러 관측소의 요약**을 쓰자." 그래서 관측소마다 **선행 후보 상위 3개**(상관 ≥ 0.4)를 모두 이어요.
그리고 **무리 나누기(Louvain)**로 서로 촘촘히 이어진 관측소끼리 묶어요. 한 무리 ≈ "같이 움직이는 이웃"이에요.
""")
b.code(r'''
k3 = top3[top3["corr"] >= .4]
G3 = nx.Graph()
for _, r in k3.iterrows():
    if G3.has_edge(r.leader, r.follower):
        G3.edges[r.leader, r.follower]["weight"] = max(G3.edges[r.leader, r.follower]["weight"], r["corr"])
    else:
        G3.add_edge(r.leader, r.follower, weight=float(r["corr"]))
comm = nx.community.louvain_communities(G3, weight="weight", seed=0, resolution=1.0)
comm = sorted(comm, key=len, reverse=True)
comm_of = {n_: i for i, c in enumerate(comm) for n_ in c}
Q = nx.community.modularity(G3, comm, weight="weight")
print(f"노드 {G3.number_of_nodes()} · 선 {G3.number_of_edges()} · 무리 {len(comm)}개 · 모듈성 Q = {Q:.2f} (0.3 넘으면 무리 구조가 뚜렷)")

pal = plt.cm.tab20(np.linspace(0, 1, 20))
col_of = lambda n_: pal[comm_of[n_] % 20] if len(comm[comm_of[n_]]) >= 4 else (.75, .75, .72, 1)
pos3 = nx.spring_layout(G3, seed=7, k=.18, iterations=300, weight="weight")
fig, ax = plt.subplots(figsize=(14, 10))
for u, v, d in G3.edges(data=True):
    same = comm_of[u] == comm_of[v]
    ax.plot(*zip(pos3[u], pos3[v]), color=col_of(u) if same else "#d3d1c7", lw=.4 + 2 * (d["weight"] - .4), alpha=.7 if same else .4, zorder=1)
xy = np.array([pos3[n_] for n_ in G3.nodes])
ax.scatter(xy[:, 0], xy[:, 1], s=10 + 18 * np.log10(env.wsarea.reindex(list(G3.nodes)).to_numpy() + 1),
           color=[col_of(n_) for n_ in G3.nodes], edgecolor="white", lw=.5, zorder=3)
for i, c in enumerate(comm[:12]):
    cx = np.mean([pos3[n_] for n_ in c], axis=0)
    ax.annotate(f"무리 {i + 1}", cx, fontsize=9, weight="bold", ha="center", color="#2c2c2a",
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.7), zorder=5)
ax.set_axis_off(); ax.grid(False)
ax.set_title(f"상위 3개 선행(상관 ≥ 0.4)으로 이은 이웃망 · Louvain 무리 {len(comm)}개 (4곳 미만 무리는 회색)")
fig.tight_layout(); fig.savefig(FIG / "06_top3_communities.png", dpi=150, bbox_inches="tight"); plt.show()

cprof = pd.DataFrame([{"무리": f"무리 {i + 1}", "관측소 수": len(c),
                       "상류면적 중앙값": env.wsarea.reindex(list(c)).median(),
                       "하천차수 평균": env.stream_order.reindex(list(c)).mean(),
                       "고도 평균": env.mean_elevation.reindex(list(c)).mean(),
                       "도시 비율 평균": env.urban_ratio.reindex(list(c)).mean()} for i, c in enumerate(comm[:12])])
cprof.round(2)
''')
b.md("""
- 한 명만 고를 때(2장)보다 선이 많아져 **덩어리들이 서로 이어지고**, 그 안에서 무리가 또렷이 갈려요.
- 무리마다 고도·도시 비율 같은 유역 성격이 달라요(표). 무리는 **"같은 권역·같은 지형"** 을 데이터로 다시 찾은 셈이에요.
- 모델링 아이디어: **무리 번호**를 범주 변수로, 또는 **같은 무리의 같은 시각 평균 상승**을 연결망 변수로 쓸 수 있어요(규정 확인 필요, 5권 6장).
""")

# ── 8. 히트맵 ──────────────────────────────────────────
b.md("""
## 8. 인접 행렬 히트맵 — 무리 안이 정말 빽빽한가

그래프는 선이 겹치면 읽기 어려워요. 같은 정보를 **표(행렬)** 로 그리면 겹침이 없어요.
- 행 = 선행 후보, 열 = 후행, 칸 색 = 시차 1~8시간 중 가장 큰 상관
- 관측소를 **무리 순서**로 정렬하면, 무리 안쪽이 대각선을 따라 **진한 블록**으로 보여야 해요.
""")
b.code(r'''
best = np.nanmax(np.stack([Cs[L] for L in range(1, MAXLAG + 1)]), axis=0)
order = [n_ for c in comm for n_ in sorted(c, key=lambda s: env.wsarea[s])]
rest = [s for s in considered if s not in comm_of]
idx = [code_of[s] for s in order + rest]
M = best[np.ix_(idx, idx)].copy(); np.fill_diagonal(M, np.nan)
fig, ax = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.3, 1]})
im = ax[0].imshow(M, cmap="magma_r", vmin=0, vmax=.8, interpolation="nearest")
b_edges = np.cumsum([len(c) for c in comm])
for e in b_edges[:-1]:
    ax[0].axhline(e - .5, color=C_NT, lw=.3); ax[0].axvline(e - .5, color=C_NT, lw=.3)
ax[0].set_xlabel("후행 (무리 순서)"); ax[0].set_ylabel("선행 후보 (무리 순서)"); ax[0].grid(False)
ax[0].set_title("시차 상관 행렬 (무리 순서로 정렬 · 파랑 선 = 무리 경계)"); plt.colorbar(im, ax=ax[0], shrink=.7, label="최대 시차 상관")
# 큰 무리 하나 확대
c0 = sorted(comm[0], key=lambda s: env.wsarea[s]); i0 = [code_of[s] for s in c0]
M0 = best[np.ix_(i0, i0)].copy(); np.fill_diagonal(M0, np.nan)
ax[1].imshow(M0, cmap="magma_r", vmin=0, vmax=.8)
ax[1].set_xticks(range(len(c0))); ax[1].set_xticklabels([s[2:] for s in c0], rotation=90, fontsize=6)
ax[1].set_yticks(range(len(c0))); ax[1].set_yticklabels([s[2:] for s in c0], fontsize=6); ax[1].grid(False)
ax[1].set_title(f"무리 1 확대 ({len(c0)}곳, 유역 작은 → 큰 순)")
fig.tight_layout(); fig.savefig(FIG / "07_adjacency_heatmap.png", dpi=150, bbox_inches="tight"); plt.show()
inside = np.nanmean([best[code_of[a], code_of[b_]] for c in comm if len(c) >= 4 for a in c for b_ in c if a != b_])
outside = np.nanmean(best[np.ix_([code_of[s] for s in considered], [code_of[s] for s in considered])])
print(f"평균 시차 상관: 같은 무리 안 {inside:.2f} · 전체 쌍 {outside:.2f}")
iu0, il0 = np.triu_indices(len(c0), 1), np.tril_indices(len(c0), -1)
print(f"무리 1 확대: 위쪽 삼각형(작은→큰) 평균 {np.nanmean(M0[iu0]):.3f} · 아래쪽(큰→작은) 평균 {np.nanmean(M0[il0]):.3f}")
''')
b.md("""
- 대각선을 따라 진한 블록이 보여요. 무리 나누기가 **억지로 만든 게 아니라** 실제로 촘촘한 묶음이라는 뜻이에요.
- 무리 경계 밖에도 진한 블록이 있어요(예: 무리 1 ↔ 무리 3). 무리끼리도 느슨하게 이어져 있다는 뜻이에요 — 7장 그림에서 무리들이 서로 붙어 있는 것과 같아요.
- 오른쪽 확대: 행 = 선행 후보, 열 = 후행이고 유역 작은 → 큰 순이에요. 대각선 **위쪽 삼각형**(작은 유역이 큰 유역을 이끔)이 아래쪽보다 진한지 아래 숫자로 확인해요.
""")

# ── 9. Arc ─────────────────────────────────────────────
b.md("""
## 9. 호(arc) 그림 — 방향을 한 줄로

관측소를 **상류면적 순서로 한 줄**에 세우고, 선행 → 후행을 호로 그려요.
- **위쪽 호** = 작은 유역 → 큰 유역(예상 방향) · **아래쪽 호** = 반대 방향
- 호가 **짧으면** 비슷한 크기끼리, **길면** 크기 차이가 큰 쌍이에요.
""")
b.code(r'''
line = sorted(G.nodes, key=lambda s: env.wsarea[s]); xpos = {s: i for i, s in enumerate(line)}
fig, ax = plt.subplots(figsize=(16, 6))
t = np.linspace(0, np.pi, 40)
n_up = n_dn = 0
for u, v, d in G.edges(data=True):
    a, b_ = xpos[u], xpos[v]; up = env.wsarea[v] > env.wsarea[u]
    n_up += up; n_dn += (not up) and not np.isclose(env.wsarea[v], env.wsarea[u])
    r = abs(b_ - a) / 2; cx = (a + b_) / 2
    ax.plot(cx + r * np.cos(t) * np.sign(b_ - a) * -1, (1 if up else -1) * r * np.sin(t),
            color=lag_color(d["lag"]), lw=.4 + 1.5 * (d["corr"] - .5), alpha=.45)
ax.scatter(range(len(line)), np.zeros(len(line)), s=6, color="#2c2c2a", zorder=3)
q = np.linspace(0, len(line) - 1, 7).astype(int)
ax.set_xticks(q); ax.set_xticklabels([f"{env.wsarea[line[i]]:.0f}" for i in q]); ax.set_xlabel("관측소 (상류면적 km² 순서)")
ax.set_yticks([]); ax.axhline(0, color=C_ALL, lw=.5); ax.grid(False)
ax.text(.01, .95, f"▲ 작은 → 큰 유역 {n_up}개", transform=ax.transAxes, color=C_NT, fontsize=10)
ax.text(.01, .04, f"▼ 큰 → 작은 유역 {n_dn}개", transform=ax.transAxes, color=C_TIDAL, fontsize=10)
lag_legend(ax, loc="upper right"); ax.set_title("호 그림: 위 = 예상 방향, 아래 = 반대 방향 (색 = 선행 간격)")
fig.tight_layout(); fig.savefig(FIG / "08_arc_diagram.png", dpi=150, bbox_inches="tight"); plt.show()
span = pd.DataFrame([{"방향": "위(작은→큰)" if env.wsarea[v] > env.wsarea[u] else "아래(큰→작은)", "순위 거리": abs(xpos[v] - xpos[u])}
                     for u, v in G.edges if not np.isclose(env.wsarea[v], env.wsarea[u])])
span.groupby("방향")["순위 거리"].describe()[["count", "25%", "50%", "75%"]].round(0)
''')
b.md("""
- 위쪽 호가 아래쪽보다 **약 2배** 많아요. 3장과 같은 결론을 한 줄 그림으로 본 거예요.
- 호 길이(순위 거리)는 표로 비교해요. 위아래 길이가 비슷하다면, 반대 방향 관계도 "비슷한 크기끼리"만은 아니라는 뜻이에요 — 비 이동 방향 탓일 가능성이 커요.
- 긴 호도 대부분 파랑(1h)·초록(2h)이에요. 유역 크기 차이가 커도 선행 간격은 짧게 잡혀요.
""")

# ── 10. 모델 개선 ──────────────────────────────────────
b.md("""
## 10. 모델 개선 위에 얹기 — 연결망 변수가 어디서 도움이 됐나

5권 5장의 교차검증 예측(`cache/oof_s09.parquet`)으로 **관측소별 RMSE 비율**(연결망 모델 ÷ 기준 모델)을 계산해 노드 색으로 칠해요.
🔵 파랑 = 좋아짐(비율 < 1), 🔴 빨강 = 나빠짐. 자리는 2장과 같아요.
""")
b.code(r'''
oof = pd.read_parquet(CACHE / "oof_s09.parquet")
m = tr[["station_id", "station_code", "hour", TARGET]].merge(oof, on=["station_code", "hour"])
m["eb"], m["en"] = (m[TARGET] - m.oof_base) ** 2, (m[TARGET] - m.oof_net) ** 2
ratio = m.groupby("station_id").apply(lambda q: np.sqrt(q.en.mean() / q.eb.mean()))
fig, ax = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.6, 1]})
draw_edges(ax[0], G, pos, color=lambda *_: "#c9c7bf", width=lambda *_: .6)
nodes = list(G.nodes); xy = np.array([pos[n_] for n_ in nodes])
norm = mcolors.TwoSlopeNorm(vmin=.9, vcenter=1, vmax=1.1)
sc = ax[0].scatter(xy[:, 0], xy[:, 1], s=26, c=ratio.reindex(nodes).clip(.9, 1.1), cmap="RdBu_r", norm=norm, edgecolor="#5f5e5a", lw=.3, zorder=3)
plt.colorbar(sc, ax=ax[0], shrink=.6, label="RMSE 비율 (연결망 ÷ 기준)"); ax[0].set_axis_off(); ax[0].grid(False)
ax[0].set_title("연결망 위의 관측소별 개선 정도")
role = pd.Series("연결망 밖", index=considered)
role[[s for s in considered if s in G and G.out_degree(s) > 0 and G.in_degree(s) == 0]] = "뿌리 (이끌기만)"
role[[s for s in considered if s in G and G.in_degree(s) > 0 and G.out_degree(s) == 0]] = "끝 (따라가기만)"
role[[s for s in considered if s in G and G.in_degree(s) > 0 and G.out_degree(s) > 0]] = "중간"
rr = pd.DataFrame({"role": role, "ratio": ratio.reindex(role.index)}).dropna()
ordr = ["연결망 밖", "뿌리 (이끌기만)", "중간", "끝 (따라가기만)"]
ax[1].boxplot([rr[rr.role == r].ratio.clip(.8, 1.2) for r in ordr], labels=[f"{r}\n({(rr.role == r).sum()})" for r in ordr], showfliers=False)
for i, r in enumerate(ordr):
    v = rr[rr.role == r].ratio.clip(.8, 1.2)
    ax[1].scatter(np.full(len(v), i + 1) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=6, alpha=.4, color=C_NT)
ax[1].axhline(1, color="k", lw=.6); ax[1].set_ylabel("RMSE 비율"); ax[1].set_title("연결망 속 역할별 개선")
fig.tight_layout(); fig.savefig(FIG / "09_model_gain_overlay.png", dpi=150, bbox_inches="tight"); plt.show()
rr.groupby("role").ratio.agg(["count", "median", lambda v: (v < 1).mean()]).rename(columns={"<lambda_0>": "개선 비율"}).reindex(ordr).round(3)
''')
b.md("""
- **뒤따르는 쪽(끝·중간)** 이 가장 많이 좋아져요. 선행 관측소의 신호를 **받는** 입장이니 당연해요.
- **뿌리**는 자기보다 먼저 움직이는 관측소가 없어서 이득이 작아요. 뿌리에는 연결망 변수보다 **강우 정보**가 더 중요해요.
- **연결망 밖**(뚜렷한 선행이 없는 곳)은 개선 비율이 1/3 정도로, 오히려 조금 나빠지는 곳이 많아요. 쓸모없는 변수가 잡음만 더한 셈이에요 → 연결망 변수는 **선행이 있는 관측소에만** 켜는 게 좋아요.
""")

# ── 11. 호우별 ─────────────────────────────────────────
b.md("""
## 11. 호우별 연결망 — 호우마다 얼마나 바뀌나

5권 4-2의 "선행이 호우마다 바뀐다"를 직접 봐요. **가장 긴 호우 4개**에서 각각 따로 선행을 찾아 2장과 같은 자리에 그려요.
- 🔵 파랑 화살표 = 전체 연결망(2장)에도 있는 관계 · 🟠 주황 = 이 호우에서만 나타난 관계

⚠️ **함정**: 호우 하나는 40~150시간뿐이라, 관측소 440곳 × 시차 8개 중 **가장 큰 상관**을 고르면 우연만으로도 0.5를 쉽게 넘어요.
그래서 **우연 기준선**을 같이 봐요: 그 호우의 시각 순서를 무작위로 섞어(시차 관계를 깨뜨려) 똑같이 선행을 찾으면 몇 개가 나오나?
진짜 관계라면 "원래 개수"가 "섞은 개수"보다 확실히 많아야 해요. 기준을 상관 **0.6**으로 올리고, 최소 시각 수는 30으로 낮췄어요.
""")
b.code(r'''
EV_TH, EV_MIN = .6, 30
def event_leaders(ev, shuffle=False, seed=0):
    Z_ = Z
    if shuffle:                                   # 이 호우 안에서 시각 순서만 섞기 (시차 구조 파괴, 관측소끼리 같은 시각 관계는 유지)
        rows = np.where(blk_of_hour == ev)[0]; Z_ = Z.copy()
        Z_[rows] = Z[np.random.default_rng(seed).permutation(rows)]
    lt = find_leaders(lagged_corrs(Z_, blk_of_hour == ev, min_n=EV_MIN), SPECIAL_CODES).set_index("follower")
    return lt[lt["corr"] >= EV_TH]

hrs_of = pd.Series(blk_of_hour).value_counts()
big_ev = hrs_of.index[:4]
allE = set(G.edges)
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
summary = []
for ax, ev in zip(axes.flat, big_ev):
    Ge = build_graph(event_leaders(ev))
    n_null = np.mean([len(event_leaders(ev, True, sd)) for sd in range(3)])
    shared = [e for e in Ge.edges if e in allE]
    xy_all = np.array([pos[n_] for n_ in G.nodes])
    ax.scatter(xy_all[:, 0], xy_all[:, 1], s=5, color="#d3d1c7", zorder=1)
    Ge_in = Ge.subgraph([n_ for n_ in Ge.nodes if n_ in pos])
    draw_edges(ax, Ge_in, pos, color=lambda u, v, d: C_NT if (u, v) in allE else C_TIDAL,
               width=lambda u, v, d: 1.6 if (u, v) in allE else .6, alpha=.8)
    ax.set_title(f"사상 #{ev} · {hrs_of[ev]}시간 · 선행 {Ge.number_of_edges()}개 (우연 기준 {n_null:.0f}개) · 전체 망과 같음 {len(shared)}개", fontsize=10)
    ax.set_axis_off(); ax.grid(False)
    summary.append({"사상": ev, "시간": hrs_of[ev], "선행 관계": Ge.number_of_edges(), "우연 기준(섞은 평균)": round(n_null),
                    "전체 망과 같음": len(shared), "같은 비율": round(len(shared) / max(Ge.number_of_edges(), 1), 2)})
fig.legend(handles=[Line2D([], [], color=C_NT, lw=2, label="전체 연결망에도 있음"), Line2D([], [], color=C_TIDAL, lw=1, label="이 호우에서만 (화면 밖 노드와의 관계는 생략)")],
           loc="lower center", ncol=2, fontsize=10)
fig.suptitle(f"가장 긴 호우 4개에서 따로 찾은 선행 연결망 (상관 ≥ {EV_TH}, 자리는 전체 연결망 배치)", y=.99)
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(FIG / "10_per_event.png", dpi=150, bbox_inches="tight"); plt.show()
pd.DataFrame(summary)
''')
b.code(r'''
# 전체 연결망의 화살표 하나하나가 긴 호우 12개 중 몇 개에서 다시 나타나나
ev12 = hrs_of.index[:12]
ev_edges = {ev: {(r.leader, f) for f, r in event_leaders(ev).iterrows()} for ev in ev12}
rep_cnt = pd.Series({e: sum(e in ev_edges[ev] for ev in ev12) for e in G.edges})
null_rate = np.mean([len(ev_edges[ev]) for ev in ev12]) / (len(considered) * (len(cats) - len(SPECIAL) - 1))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
vc = rep_cnt.value_counts().sort_index()
ax[0].bar(vc.index, vc.values, color=C_NT)
ax[0].set_xlabel(f"긴 호우 {len(ev12)}개 중 다시 나타난 횟수"); ax[0].set_ylabel("전체 망 화살표 수")
ax[0].set_title(f"전체 망 화살표의 재등장 (아무 쌍이나 고르면 호우당 약 {null_rate:.1%})")
# 재등장 횟수를 굵기로
draw_edges(ax[1], G, pos, color=lambda u, v, d: plt.cm.Blues(.3 + .7 * min(rep_cnt[(u, v)], 6) / 6),
           width=lambda u, v, d: .4 + .6 * rep_cnt[(u, v)])
xy_all = np.array([pos[n_] for n_ in G.nodes]); ax[1].scatter(xy_all[:, 0], xy_all[:, 1], s=5, color="#5f5e5a", zorder=3)
ax[1].set_axis_off(); ax[1].grid(False); ax[1].set_title("굵고 진할수록 여러 호우에서 반복된 관계")
fig.tight_layout(); fig.savefig(FIG / "11_edge_recurrence.png", dpi=150, bbox_inches="tight"); plt.show()
print(f"전체 망 화살표 {len(rep_cnt)}개: 한 번도 안 나타남 {(rep_cnt == 0).mean():.0%} · 3개 이상 호우에서 반복 {(rep_cnt >= 3).mean():.0%}")
print(f"뼈대(5장 두 절반 모두)인 화살표의 재등장 중앙값 {rep_cnt[[e for e in rep_cnt.index if e in e_both]].median():.0f} · 나머지 {rep_cnt[[e for e in rep_cnt.index if e not in e_both]].median():.0f}")
''')
b.md("""
- **가장 긴 호우(#125, 155시간)**: 선행 288개 vs 우연 기준 약 1개 → 거의 전부 진짜 신호예요. 그중 전체 망과 같은 관계가 약 1/3이고, 나머지(주황)는 **이 호우에서만** 나타난 관계예요. 호우마다 선행이 바뀐다는 5권 결론을 그림으로 확인한 거예요.
- **짧은 호우(50~70시간)**: 선행 관계 수가 우연 기준(150~280개)과 크게 다르지 않아요. 그림의 주황 화살표 대부분은 **우연**이라는 뜻이에요. 짧은 호우 하나로 연결망을 새로 찾는 건 믿기 어려워요.
- 그래도 짧은 호우에서도 파랑(전체 망과 같음)은 30~47개 나와요. 아무 쌍이나 고르면 맞힐 확률이 아주 낮아서, 이것들은 의미가 있어요.
- 아래 그림: 전체 망 화살표 중 약 1/4은 긴 호우 12개 어디에서도 다시 안 나오고, 약 30%는 **3개 이상 호우에서 반복**돼요. 5장 **뼈대** 화살표는 재등장 중앙값이 3번으로, 나머지(1번)보다 확실히 자주 반복돼요.
- → 모델링 시사점: 고정된 연결망 하나를 믿기보다, **반복되는 뼈대만** 쓰거나 **주변 여러 관측소 요약**(7장 무리 평균 등)을 쓰는 게 새 호우(test)에 강할 거예요.
""")

# ── 12. 인터랙티브 ─────────────────────────────────────
b.md("""
## 12. 인터랙티브 그래프 — 마우스로 둘러보기

plotly로 2장 그래프를 다시 그려요. 노드에 마우스를 올리면 관측소 번호·상류면적·하천차수·무리·RMSE 비율·선행 관측소가 보여요.
같은 그림을 `figures/network_viz/interactive_network.html`로도 저장해서 브라우저에서 따로 열 수 있어요.
""")
b.code(r'''
import plotly.graph_objects as go
ex, ey = [], []
for u, v in G.edges:
    ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
# 화살촉 대신 후행 쪽 1/4 지점에 작은 점
hx = [pos[u][0] * .25 + pos[v][0] * .75 for u, v in G.edges]; hy = [pos[u][1] * .25 + pos[v][1] * .75 for u, v in G.edges]
nodes = list(G.nodes)
lead_txt = {f: f"{r.leader} ({int(r.lag)}h, r={r['corr']:.2f})" for f, r in strong.iterrows()}
txt = [f"<b>{n_}</b><br>상류면적 {env.wsarea[n_]:.0f} km²<br>하천차수 {env.stream_order[n_]:.1f}<br>무리 {comm_of.get(n_, -1) + 1}"
       f"<br>RMSE 비율 {ratio.get(n_, np.nan):.3f}<br>선행: {lead_txt.get(n_, '없음')}<br>직접 이끄는 수 {G.out_degree(n_)}" for n_ in nodes]
fig = go.Figure([
    go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#c9c7bf", width=1), hoverinfo="skip", name="선행 → 후행"),
    go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=4, color="#9a9890"), hoverinfo="skip", showlegend=False),
    go.Scatter(x=[pos[n_][0] for n_ in nodes], y=[pos[n_][1] for n_ in nodes], mode="markers", text=txt, hoverinfo="text",
               marker=dict(size=[5 + 4 * np.log10(env.wsarea[n_] + 1) for n_ in nodes],
                           color=[env.stream_order[n_] for n_ in nodes], colorscale="Viridis", cmin=1, cmax=4,
                           colorbar=dict(title="하천차수"), line=dict(color="white", width=.5)), name="관측소")])
fig.update_layout(title="관측소 선행 연결망 (마우스를 올려 보세요 · 회색 점 = 화살표 끝 쪽)", width=1000, height=750,
                  xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white", showlegend=False)
fig.write_html(FIG / "interactive_network.html", include_plotlyjs="cdn")
fig.show()
''')

# ── 13. 믿을 수 있는 관계만 ─────────────────────────────
b.md("""
## 13. 믿을 수 있는 관계만 — 공유용 요약

지금까지 그림은 전체에서 찾은 선행 242개를 모두 그렸어요. 그런데 5장에서 봤듯이 사상을 반으로 나누면 그중 약 2/3는 다른 관측소로 바뀌어요.
그래서 **두 절반 모두에서 같은 선행이 나온 관계(5장 뼈대)**만 남겨 다시 봐요. 이 관계가 정말 믿을 만한지는 방향(작은 유역 → 큰 유역)과 다른 호우에서의 재등장(11장)으로 확인해요.
""")
b.code(r'''
# 믿을 수 있는 관계 = 사상을 짝·홀로 나눠 두 절반 모두에서 같은 선행이 나온 관계(5장 뼈대)
K = nx.DiGraph()
for u, v in e_both:
    K.add_edge(u, v, lag=int(half["짝수"].loc[v, "lag"]), corr=float((half["짝수"].loc[v, "corr"] + half["홀수"].loc[v, "corr"]) / 2))
dir_of = lambda u, v: "같은 소권역" if np.isclose(env.wsarea[u], env.wsarea[v]) else ("작은 → 큰 유역" if env.wsarea[v] > env.wsarea[u] else "큰 → 작은 유역")
DIR_COL = {"작은 → 큰 유역": C_NT, "큰 → 작은 유역": C_TIDAL, "같은 소권역": C_ALL}
dcnt = pd.Series([dir_of(u, v) for u, v in K.edges]).value_counts()
up, down = dcnt.get("작은 → 큰 유역", 0), dcnt.get("큰 → 작은 유역", 0)
rec = lambda e: sum(e in ev_edges[ev] for ev in ev12)          # 긴 호우 12개 중 다시 나온 횟수 (11장)
dropped = set(G.edges) - e_both                                  # 전체에서 찾았지만 두 절반 검사를 통과 못 한 후보

# 그림 1: 선행 간격 + 예시 한 쌍
fig, ax = plt.subplots(1, 2, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 2.4]})
lagc = pd.Series([d["lag"] for *_, d in K.edges(data=True)]).value_counts(normalize=True).reindex(range(1, 6), fill_value=0) * 100
ax[0].bar(lagc.index, lagc.values, color=[lag_color(L) for L in lagc.index])
ax[0].set(xlabel="선행 간격 (h)", ylabel="관계 비율 (%)", title=f"믿을 수 있는 선행 {len(K.edges)}쌍: 대부분 1시간 앞섬")
ex_l, ex_f = "ST0323", "ST0156"; L = K.edges[ex_l, ex_f]["lag"]
ev = hrs_of.index[0]; q = tr[tr.event_block == ev].pivot_table(index="hour", columns="station_id", values="dwl_1h")
t = q.index - q.index.min()
ax[1].plot(t + L, q[ex_l], color=C_TIDAL, lw=1.4, label=f"{ex_l} (선행, {L}h 늦춰 그림)")
ax[1].plot(t, q[ex_f], color=C_NT, lw=1.4, label=f"{ex_f} (후행)")
ax[1].set(xlabel=f"사상 #{ev} 경과 시간 (h)", ylabel="dwl_1h (m)", title=f"예: 선행 관측소를 {L}시간 늦추면 겹친다 (상관 {K.edges[ex_l, ex_f]['corr']:.2f})")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "12_reliable_lead_lag.png", dpi=150, bbox_inches="tight"); plt.show()

# 그림 2: 방향(상류 → 하류) + 다른 호우에서의 재등장
fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
for k, c in DIR_COL.items():
    es = [(u, v) for u, v in K.edges if dir_of(u, v) == k]
    ax[0].scatter([np.log10(env.wsarea[u]) for u, v in es], [np.log10(env.wsarea[v]) for u, v in es], s=30, color=c, alpha=.85, label=f"{k} ({len(es)})")
lim = [1.1, 4.1]; ax[0].plot(lim, lim, ls=":", color="gray")
ax[0].set(xlim=lim, ylim=lim, xlabel="선행 관측소 상류면적 log10(km²)", ylabel="후행 관측소 상류면적 log10(km²)",
          title=f"대각선 위 = 뒤따르는 쪽이 더 큰 유역 ({up / (up + down):.0%}, 무작위라면 50%)")
ax[0].legend(fontsize=9, loc="lower right")
vals = [np.median([rec(e) for e in e_both]), np.median([rec(e) for e in dropped])]
ax[1].bar(["믿을 수 있는 관계", "탈락한 후보"], vals, color=[C_NT, C_ALL])
for i, v in enumerate(vals):
    ax[1].text(i, v, f"{v:.0f}번", ha="center", va="bottom")
ax[1].set(ylabel=f"긴 호우 {len(ev12)}개 중 다시 나온 횟수 (중앙값)", title="다른 호우에서도 다시 나오나")
fig.tight_layout(); fig.savefig(FIG / "13_reliable_direction.png", dpi=150, bbox_inches="tight"); plt.show()

comps_k = sorted(nx.weakly_connected_components(K), key=len, reverse=True)
print(f"믿을 수 있는 관계 {len(K.edges)}쌍 = 후행 관측소 {len({v for u, v in K.edges})}곳 (일반 {len(considered)}곳 중, 전체 후보 {len(strong)}곳)")
print(f"1h 앞섬 {np.mean([d['lag'] == 1 for *_, d in K.edges(data=True)]):.0%} · 2h {np.mean([d['lag'] == 2 for *_, d in K.edges(data=True)]):.0%} · 두 절반 간격 일치 {np.mean([half['짝수'].loc[v, 'lag'] == half['홀수'].loc[v, 'lag'] for u, v in e_both]):.0%}")
print(f"방향: 작은→큰 {up} · 큰→작은 {down} · 같은 소권역 {dcnt.get('같은 소권역', 0)} → 작은→큰 비율 {up / (up + down):.0%}")
print(f"재등장 중앙값: 믿을 수 있는 관계 {vals[0]:.0f} · 탈락한 후보 {vals[1]:.0f}")
print(f"덩어리 {len(comps_k)}개 · 가장 큰 덩어리 {len(comps_k[0])}곳")
''')
b.md("""
아래는 같은 85쌍을 **연결망 한 장**으로 그린 거예요. 동그라미 하나가 서로 이어진 관측소 묶음이고, 묶음 안에서는 작은 유역이 위, 큰 유역이 아래에 오도록 돌렸어요. 그래서 **아래로 향하는 파란 화살표**가 예상 방향(상류 → 하류)이에요.
""")
b.code(r'''
# 믿을 수 있는 85쌍을 한 장으로: 묶음(연결 성분)마다 동그라미 하나, 원 채우기로 배치
from matplotlib.patches import Circle
rep = {e: rec(e) for e in K.edges}
la = {n: np.log10(env.wsarea[n]) for n in K.nodes}

def comp_layout(c):
    """묶음 하나를 배치하고, 작은 유역 → 큰 유역 흐름이 아래를 향하도록 돌린다"""
    sub = K.subgraph(c)
    if len(c) == 2:
        p = {n: np.array([i * 1.35, 0.]) for i, n in enumerate(c)}
    else:
        p = {n: np.array(v) for n, v in nx.kamada_kawai_layout(sub.to_undirected()).items()}
        L = np.mean([np.linalg.norm(p[u] - p[v]) for u, v in sub.edges])
        p = {n: v / L * 1.35 for n, v in p.items()}
    d = sum(((p[v] - p[u]) * np.sign(la[v] - la[u]) for u, v in sub.edges), np.zeros(2))
    if np.linalg.norm(d) > 1e-9:
        a = np.arctan2(-1, 0) - np.arctan2(d[1], d[0])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        p = {n: R @ v for n, v in p.items()}
    ctr = (np.max(list(p.values()), axis=0) + np.min(list(p.values()), axis=0)) / 2
    return {n: v - ctr for n, v in p.items()}

comps_k = sorted(nx.weakly_connected_components(K), key=len, reverse=True)
lays = [comp_layout(c) for c in comps_k]
rads = [max(np.linalg.norm(v) for v in l.values()) + .5 for l in lays]
placed = []                                              # 원 채우기: 가능한 자리 중 가운데에 가장 가까운 곳
GAP, ASP = .12, 1.45
for r in rads:
    if not placed:
        placed.append((np.zeros(2), r)); continue
    best, bd = None, np.inf
    for c0, r0 in placed:
        for th in np.linspace(0, 2 * np.pi, 96, endpoint=False):
            c = c0 + (r0 + r + GAP) * np.array([np.cos(th), np.sin(th)])
            dd = np.hypot(c[0] / ASP, c[1])
            if dd < bd and all(np.linalg.norm(c - c2) >= r + r2 + GAP - 1e-9 for c2, r2 in placed):
                best, bd = c, dd
    placed.append((best, r))
pos = {n: v + c for l, (c, _) in zip(lays, placed) for n, v in l.items()}

fig, ax = plt.subplots(figsize=(14, 9.6))
for c, r in placed:
    ax.add_patch(Circle(c, r, facecolor="#f1efe8", edgecolor="none", zorder=0))
for u, v in sorted(K.edges, key=lambda e: rep[e]):
    ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", mutation_scale=10, shrinkA=6, shrinkB=7,
                 connectionstyle="arc3,rad=.08", color=DIR_COL[dir_of(u, v)], lw=1.0 + .35 * rep[(u, v)],
                 alpha=.95, zorder=2, capstyle="round", joinstyle="round"))
ns = list(K.nodes); xy = np.array([pos[n] for n in ns])
sc = ax.scatter(xy[:, 0], xy[:, 1], s=[22 + 40 * (la[n] - 1.2) for n in ns], c=[la[n] for n in ns], cmap="YlGnBu",
                vmin=1.0, vmax=4.0, edgecolor="#2c2c2a", lw=.7, zorder=3)
cb = fig.colorbar(sc, ax=ax, shrink=.4, pad=.0, aspect=22)
cb.set_ticks([1.5, 2, 3, 4]); cb.set_ticklabels(["30", "100", "1,000", "10,000"]); cb.set_label("상류면적 (km²)")
cb.outline.set_visible(False)
leg = [Line2D([], [], color=c, lw=2.6, label=f"{k} ({dcnt.get(k, 0)}쌍)") for k, c in DIR_COL.items()]
leg += [Line2D([], [], color="#5f5e5a", lw=w, label=lab) for w, lab in [(1.0, "굵기: 긴 호우 12개 중 재등장 0번"), (2.75, "… 5번")]]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9, title="화살표 = 선행 → 후행", title_fontsize=9)
ax.set_aspect("equal"); ax.set_axis_off(); ax.grid(False); ax.autoscale_view()
ax.set_title(f"믿을 수 있는 선행 관계 {len(K.edges)}쌍", fontsize=15, pad=26)
ax.text(.5, 1.01, "동그라미 하나 = 함께 움직이는 관측소 묶음 · 묶음 안에서 위 → 아래 = 작은 유역 → 큰 유역", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=10, color="#5f5e5a")
ex_l, ex_f = "ST0323", "ST0156"
for n_ in (ex_l, ex_f):
    ax.annotate(n_, pos[n_], xytext=(pos[n_][0] + .5, pos[n_][1]), fontsize=8, color="#2c2c2a", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#8a8880", lw=.6))
fig.tight_layout(); fig.savefig(FIG / "14_reliable_network.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"예시 {ex_l} → {ex_f}: {dir_of(ex_l, ex_f)} · 상류면적 {env.wsarea[ex_l]:,.0f} → {env.wsarea[ex_f]:,.0f} km² · 긴 호우 {len(ev12)}개 중 {rep[(ex_l, ex_f)]}개에서 재등장")
''')
b.md("""
- 믿을 수 있는 관계는 **85쌍**이에요. 일반 관측소 442곳 중 85곳이 이런 선행을 가지고, 전체 후보 242곳의 약 1/3이에요.
- 간격은 1시간이 73%, 2시간이 19%예요. 두 절반에서 간격까지 같은 경우가 82%예요.
- 유역 크기가 다른 56쌍 중 **73%**가 작은 유역 → 큰 유역이에요(무작위라면 50%, 전체 후보는 68%). 나머지 29쌍은 같은 소권역 안의 이웃이에요.
- 긴 호우 12개 중 **중앙값 3개**에서 다시 나타나요. 탈락한 후보는 1개예요.
- 덩어리는 58개로 흩어져 있고 가장 큰 것도 6곳이에요. 믿을 수 있는 건 **가까운 관측소끼리의 짧은 연결**이에요.
- 예시 쌍 ST0323 → ST0156(상류면적 163 → 1,114 km², 1시간 앞섬)은 긴 호우 12개 중 10개에서 다시 나타나요.
""")

b.md("""
## 정리

| 그림 | 보이는 것 | 모델링 시사점 |
|---|---|---|
| 기본 힘-방향 (2장) | 거대한 망 하나가 아니라 작은 덩어리 여러 개, 화살표 대부분 1h | 연결망 변수는 짧은 앞날(1~2h)만 보강 |
| 유역면적 세로축 (3장) · 호 그림 (9장) | 작은 유역 → 큰 유역 방향이 우세, 반대 방향은 비슷한 크기끼리 | 방향 정보는 믿을 만함 |
| 덩어리 나무 (4장) | 부채꼴과 5~8단계 긴 사슬이 섞임 | 여러 단계 위 관측소로 6h 뒷부분 보강 아이디어 |
| 안정성 겹쳐 그리기 (5장) | 튼튼한 뼈대 ≈ 덩어리 안쪽 짧은 연결 | **뼈대 화살표만** 쓰기 |
| 허브 (6장) | 소수 관측소가 여러 관측소를 이끔 | 허브 결측·이상치 점검 |
| 상위 3 + 무리 (7·8장) | 무리가 뚜렷(모듈성 Q), 무리마다 유역 성격이 다름 | 무리 번호, 무리 평균 상승을 변수 후보로 |
| 모델 개선 얹기 (10장) | 뒤따르는 관측소가 가장 많이 좋아지고, 연결망 밖은 오히려 나빠짐 | 선행 있는 관측소에만 연결망 변수 켜기 |
| 호우별 (11장) | 호우마다 선행 관계가 크게 바뀌지만, 반복되는 뼈대가 있음 | 반복되는 뼈대만 쓰기, 고정 연결망 과신 금지 |
| 믿을 수 있는 관계만 (13장) | 85쌍, 1h 앞섬 73%, 작은 → 큰 유역 73%, 다른 호우에서 중앙값 3번 재등장 | 공유·모델링에는 이 85쌍만 쓰기 |

⚠️ 5권과 같은 주의: 연결망 **찾기는 train에서만**, 모델에 넣을 땐 fold마다 다시 찾기, test에서 다른 관측소의 **같은 시각(또는 과거)** 값만 쓰기, 그리고 **대회 규정 확인 후** 사용.
""")

b.write("nb5b_network_visualization.ipynb")

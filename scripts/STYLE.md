# Thesis Figure Style System

How every figure and table in the results chapter is produced. The look the user
signed off on (Figures 4.4–4.11: cross-play heatmaps, persona bars, Self-Refine
forest + stacked bars) comes entirely from this pipeline — follow it and new
figures drop in without restyling.

**Single rule:** all fonts, colours, CI conventions, the ties-excluded win-rate
definition, the heatmap renderer, and the save/table/macro helpers live in
[`style.py`](style.py). Import them; never hardcode a colour or re-define a CI in
a figure script.

---

## 1. The pipeline

```
scripts/thesis_figures/
  _bootstrap.py   # paths + makes explorer/analysis loaders importable; import FIRST
  style.py        # THE design system: apply_thesis_style(), colours, CIs, heatmap, save_fig, Numbers
  fig_benchmark.py    fig_personas.py    fig_self_refine.py    fig_team.py
  tab_cost_benefit.py
  make_all.py     # runs every script, prints a manifest
```

- Run everything: `python scripts/thesis_figures/make_all.py`. Each `fig_*.py`
  is also runnable standalone.
- Outputs go **inside the thesis tree** (from `_bootstrap.py`):
  - figures → `context/MSc_Thesis/figures/results/<slug>.pdf` (vector PDF)
  - tables → `context/MSc_Thesis/tables/results/<slug>.tex`
  - numbers → `context/MSc_Thesis/tables/results/numbers_<section>.tex`
- `_bootstrap` must be imported before `style`/loaders — it injects the repo root
  and `explorer/` onto `sys.path` so the `explorer/analysis` loaders are reused
  (no data logic is duplicated in these scripts) and silences the Streamlit-cache
  warnings those loaders emit off-runtime.

Every script's `__main__` calls `style.apply_thesis_style()` **once** before
plotting, then builds figures, then `nums.write()`.

## 2. The theme — `apply_thesis_style()`

```python
"font.family": "serif",          # matches the feupteses body text
"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
"xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
"axes.spines.top": False, "axes.spines.right": False,
"figure.dpi": 150, "savefig.format": "pdf", "savefig.bbox": "tight",
"figure.constrained_layout.use": True,   # never call tight_layout()
"axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "axes.axisbelow": True,
```

Consequences for figure code: **serif, 8–9 pt** (figures are placed at true size,
so text must read at `\linewidth`); top/right spines already off; a faint grid is
already on and behind the data; layout is constrained — do **not** call
`tight_layout`/`subplots_adjust`.

### Sizing
- `style.FULL_WIDTH = 6.0` in (≈ text block), `style.HALF_WIDTH = 3.0`.
- A 1×3 panel row (the dominant layout — one panel per game or per tier) is
  `figsize=(style.FULL_WIDTH, 2.0–2.3)`. A small single panel ≈ `(FULL_WIDTH*0.75, 1.7)`.

## 3. Colours (Okabe-Ito, colour-blind safe) — use the dicts, never literals

```python
FAMILY_COLORS  = {gemma: blue #0173b2, qwen: green #029e73, ministral/mistral: orange #de8f05}
PERSONA_COLORS = {default: blue, desperate: orange, cunning: green}
STRATEGY_COLORS= {Default: gray, Self-Refine: blue, Team (homo): orange, Team (hetero): green}
COND_COLORS    = {DD: gray, RR: green, DR: sky #56b4e9, RD: blue}
```

- Family identity is **always** Gemma=blue / Qwen=green / Mistral=orange, in the
  order `FAMILIES = ["gemma", "ministral", "qwen"]` when laying out grouped bars.
- Reference lines (parity / neutral): grey dashed, `color="0.4"` or `"0.5"`,
  `ls="--"`, `lw≈0.8`.
- Pooled / "All" marker: solid **black diamond** (`fmt="D"`, `color="black"`).
- Heatmap magnitude: `Blues` (see §5).

## 4. Labels & helpers (canonical — do not re-spell)

```python
SIZE_ORDER  = ["very_small", "small", "medium"]
SIZE_LABEL  = {very_small: "4–9B", small: "12–14B", medium: "24–27B"}   # en-dash, param-count bins
GAME_ORDER  = ["Trading", "Ultimatum", "BuySell"]
FAMILY_LABEL= {gemma: "Gemma", qwen: "Qwen", ministral/mistral: "Mistral"}
family_of(model_name) -> "gemma"|"qwen"|"ministral"
```

The published tier axis is always the **B-bins** (`4–9B` …), never very/small/medium.

## 5. Statistics — one definition, reused everywhere

- `wilson_ci(k, n)` → `(lo, hi)` for any **proportion** (win rate, completion rate).
- `bootstrap_ci(values, n_boot=10000, seed=0)` → `(lo, hi)` for any **mean**
  (payoffs). Deterministic via fixed seed, so re-runs are byte-stable.
- `win_rate(wins, losses)` → `(rate, k, n)`, **ties excluded**: `wins/(wins+losses)`.
  This is the Chapter-3 definition; never count ties/no-deals in the denominator.
- `errbars_from_ci(centers, cis)` → the 2×N yerr/xerr array, already
  `np.clip(..., 0, None)` so an exact 0/1 rate (whose Wilson bound floats a hair
  past the centre) cannot crash matplotlib.
- **No hypothesis tests / significance stars.** Evidence is phrased "CI clear of
  parity", not p-values.

## 6. Figure vocabulary (the recipes actually in use)

**Grouped bars with CI** — completion by tier (`fig_completion_by_tier`):
1×3 panels (one per game), three family bars per tier, `width=0.26`, centred with
`x + (fi-1)*width`, `yerr=errbars_from_ci(...)`, `error_kw=dict(lw=0.7, capsize=1.5)`,
`color=FAMILY_COLORS[fam]`. Shared legend via
`fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)`.

**Point-and-line slope** — retry recovery (`fig_retry_recovery`):
two x-positions `[0,1]` = (no retries, retry-3), one line per tier coloured by a
grey→blue ramp `["#b8b8b8", "#5d9bc4", "#1f4e79"]`, `marker="o"`, CI caps.

**Annotated heatmap panels** — cross-play (`fig_crossplay`): use
`style.heatmap(ax, mat, row_labels, col_labels, fmt, vmin, vmax)`. It is the
matplotlib stand-in for `sns.heatmap`: `Blues`, NaN cells masked to `#f0f0f0`
(the no-self-play diagonal / unplayed pairs), annotation text white when
`val > lo + 0.55*(hi-lo)` else black, spines off, no grid, ticks length 0.
**Compute `vmin/vmax` once across all panels** so tiers share a scale, then add a
single `fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)` with an 8 pt label.
Rows = Player 1, cols = Player 2, x-ticks rotated 30°, y-ticklabels only on panel 0.

**Forest / dumbbell** — Self-Refine head-to-head (`fig_sr_head_to_head`):
1×3 (per game), rows = `["All"] + tiers` with `ypos` reversed, horizontal
`ax.errorbar(rates, ypos, xerr=...)`, `ax.axvline(0.5, ls="--", lw=0.8, color="0.4")`
for "no effect", tier rows as blue circles (`fmt="o"`), the pooled **All** row as
a black diamond (`fmt="D", color="black"`), `xlim(0,1)`.

**Horizontal stacked share bars** — offer dynamics (`fig_sr_offer_dynamics`):
`ax.barh(games, shares, left=left, color=...)` accumulating `left`, label a
segment with `f"{s:.0%}"` only when `s > 0.07`, `ax.invert_yaxis()`,
`xlim(0,1)`, `fig.legend(loc="outside upper center", ncols=4, frameon=False)`.

## 7. Tables & the macro system (`Numbers`)

- `style.write_table(latex, slug)` → `<slug>.tex`. Tables are booktabs
  (`\toprule/\midrule/\bottomrule`), `\small`, `\multirow` for the grouped first
  column. CI cells use `style.fmt_ci(value, ci, fmt)` → `"$0.68~[0.61, 0.74]$"`
  (math mode so minus signs typeset); NaN → `"--"`.
- **Headline numbers are macros, never typed into prose.** Each script owns a
  `nums = style.Numbers("<section>")`; call `nums.add("AlphaOnlyName", value)`
  while computing figures, then `nums.write()` → `numbers_<section>.tex` of
  `\newcommand`s. `chapter4.tex` cites `\ResMediumTradingCompRetry` etc., so
  re-running the pipeline can never desync the text from the figures. Macro names
  must be alphabetic only (tier slugs become `VerySmall/Small/Medium`).

## 8. Adding a figure — checklist

- [ ] `import _bootstrap as bs` first, then `import style` and the relevant
      `analysis.*` loader (reuse loaders; don't re-read logs).
- [ ] Pull colours from the `*_COLORS` dicts; labels from `SIZE_LABEL`/`GAME_ORDER`/
      `FAMILY_LABEL`; never literal hex or "Very Small".
- [ ] Proportions → `wilson_ci`; means → `bootstrap_ci`; win rate → `win_rate`
      (ties excluded); error bars → `errbars_from_ci`.
- [ ] Width is `FULL_WIDTH`/`HALF_WIDTH`; rely on constrained_layout (no
      `tight_layout`).
- [ ] Title is declarative and names the seat/role when the game is asymmetric
      ("Player 1", "buyer payoff", "Refiner win rate", "responder (P2)").
- [ ] Grey dashed reference line where a baseline (0.5 parity / neutral) exists.
- [ ] Legend `frameon=False`, off the data (shared → `loc="outside upper center"`).
- [ ] `style.save_fig(fig, "fig_<slug>")`; register headline values via `nums.add`;
      add the script to `make_all.py` if new.

---
Rationale/history of these conventions: `~/.claude/.../memory/feedback_plot_style.md`.
The exploration notebooks in `_notebooks/oss/` are scratch; this script pipeline is
the source of truth for anything that appears in the thesis.

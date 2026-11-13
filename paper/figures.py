"""Figures for the write-up (paper/draft/) and the README.

Reads paper/data/ (paper/collect.py), paper/figures/data/ (paper/figure_data.py, run on the server) and
train/analysis/probes/summary.md, and writes paper/figures/<name>.pdf and .png.

    python paper/figures.py

Needs matplotlib, numpy and pandas. The server's venv has no matplotlib, so these were drawn on another machine
(matplotlib 3.10, pandas 2.2); every number comes from the tables above.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
FDATA = os.path.join(HERE, "figures", "data")
OUT = os.path.join(HERE, "figures")

# The paper is set in Latin Modern at 11pt, with 10pt captions. The figures use Computer Modern, which Latin
# Modern was digitised from and which matplotlib ships (cmr10, and cmb10 for bold), at 9pt for labels and 8pt for
# ticks and values. Each figure is drawn at the width it is printed at, so LaTeX includes it unscaled: TEXT is
# the text width in inches, and the narrower figures use the fractions of it that main.tex gives them.
TEXT = 6.5
BOLD = "cmb10"
# cmr10 keeps TeX's old OT1 layout: '<', '>', '|', '{', '}' and '_' show other glyphs, and it has no minus sign,
# en dash, '±' or '≥'. Symbols go through mathtext ($\pm$, $<$, $\geq$), and so do tick labels, for the minus;
# the en dash is typed as '{', which is its slot in OT1.
ENDASH = "{"

# Colours. Each model keeps one colour in every figure. The hues are Okabe and Ito's colour-blind-safe set,
# checked on white; sky blue (pilot-20), orange (T) and reddish purple (G-E2) are below 3:1 contrast with white,
# so every mark in those colours is labelled. Figures about something other than a model use neutral greys.
INK, INK2, MUTED, GRID = "#000000", "#4d4d4d", "#8c8c8c", "#dcdcdc"
COL = {"O": "#999999", "E": "#0072b2", "G-O": "#d55e00", "pilot-20": "#56b4e9", "T": "#e69f00",
       "G-E2": "#cc79a7", "R-O": "#009e73", "paper-unedited": "#c8c8c8", "paper-edit": "#595959"}
SUITES = ["code", "lcb", "mmlupro", "ifeval", "math500"]
SUITE = {"code": "MBPP+ and HumanEval+", "lcb": "LiveCodeBench", "mmlupro": "MMLU-Pro", "ifeval": "IFEval",
         "math500": "MATH-500"}
STEPS = [0, 25, 50, 75, 100, 125, 150]

plt.rcParams.update({
    "font.family": "cmr10", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.titlepad": 4, "axes.labelpad": 3, "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.6, "legend.borderaxespad": 0.2,
    "figure.constrained_layout.use": True, "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.h_pad": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        # no creation date in the PDFs, so redrawing an unchanged figure leaves its file unchanged
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), facecolor="white", metadata=meta)
    plt.close(fig)
    print("wrote", name)


def shapes():
    s = pd.read_csv(os.path.join(FDATA, "shapes.csv"))
    return {(r["model"], r["suite"]): r for r in s.to_dict("records")}


def fig_headline():
    """Scores and loop cut-offs on the five held-out suites: the original model, the best edit, GRPO."""
    s = shapes()
    models = ["O", "E", "G-O"]
    names = ["Original model (O)", "Best weight edit (E)", "GRPO from the original model (G-O)"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.55), sharey=True)
    h = 0.26
    y = np.arange(len(SUITES))[::-1].astype(float)
    for ax, col, label in zip(axes, ["pass", "cut"], ["Score (%)", "Answers cut at the 32,768-token limit (%)"]):
        for i, m in enumerate(models):
            ax.barh(y + (1 - i) * h, [s[(m, su)][col] for su in SUITES], height=h, color=COL[m],
                    edgecolor="white", linewidth=0.4, label=names[i], zorder=2)
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.xaxis.grid(True)
        ax.set_xlabel(label)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [SUITE[x] for x in SUITES])
    axes[0].set_ylim(-0.55, len(SUITES) - 0.45)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save(fig, "headline")


def fig_overview():
    """Every evaluated model: the mean over the five suites of its paired difference with the original."""
    ov = pd.read_csv(os.path.join(FDATA, "overview.csv")).set_index("eval")
    rows = [
        ("Editing sessions", None, None, None),
        ("pilot-10 (10 passes, from O)", "pilot-10-pp15", "E", None),
        ("pilot-12 (20, from pilot-10)", "pilot-12-pp15", "E", None),
        ("pilot-13 (20, from pilot-10)", "pilot-13-pp15", "E", None),
        ("combo-12-13 (the two summed)", "combo-12-13-pp15", "E", None),
        ("pilot-14 (20, from the combo) = E", "pilot-14-pp15", "E", "bold"),
        ("pilot-15 (20, from the combo)", "pilot-15-pp15", "E", None),
        ("pilot-16 (20, from E)", "pilot-16-pp15", "E", None),
        ("pilot-17 (20, from E)", "pilot-17-pp15", "E", None),
        ("pilot-18 (12, from E)", "pilot-18-pp15", "E", None),
        ("pilot-19 (24, from E)", "pilot-19-pp15", "E", None),
        ("Training on the same problems", None, None, None),
        ("RFT from O, two rounds (R-O)", "R-O-pp15", "R-O", None),
        ("GRPO from O, step 100 (G-O)", "G-O-pp15", "G-O", "bold"),
        ("GRPO from E, step 50 (G-E2)", "G-E2-pp15", "G-E2", None),
        ("Editing and training together", None, None, None),
        ("G-O plus E's weight change (T)", "T-pp15", "T", None),
        ("pilot-20: an edit made on G-O", "pilot-20-pp15", "pilot-20", None),
    ]
    fig, ax = plt.subplots(figsize=(TEXT, 3.55))
    y = 0
    ticks, labels, weights, heads = [], [], [], []
    for label, name, color, weight in rows:
        if name is None:
            y -= 0.35 if y else 0
            # placed after the layout is done (below), so kept out of it
            heads.append(ax.text(0, y, label, fontfamily=BOLD, ha="left", va="center", in_layout=False,
                                 transform=ax.get_yaxis_transform()))
            y -= 1
            continue
        m, se = ov.loc[name, "diff"], ov.loc[name, "se"]
        ax.errorbar(m, y, xerr=2 * se, fmt="o", color=COL[color], ecolor=COL[color], elinewidth=1.1,
                    markersize=4, capsize=0, zorder=3)
        ax.text(1.01, y, f"{m:+.1f}", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2,
                va="center", ha="left")
        ticks.append(y)
        labels.append(label)
        weights.append(weight)
        y -= 1
    ax.set_yticks(ticks, labels)
    for t, w in zip(ax.get_yticklabels(), weights):
        if w:
            t.set_fontfamily(BOLD)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(-1.5, 34)  # a margin left of zero, where the longest group heading runs past the row labels
    ax.set_xticks(range(0, 31, 5))
    ax.xaxis.grid(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 34)
    ax.set_xlabel(r"Mean gain over the original model across the five suites (points; bars are $\pm$2 SE)")
    # the group headings start where the longest row label starts, which is known once the layout is done
    fig.canvas.draw()
    left = min(t.get_window_extent().x0 for t in ax.get_yticklabels())
    for t in heads:
        t.set_x(ax.transAxes.inverted().transform((left, 0))[0])
    save(fig, "overview")


def fig_training():
    """GRPO checkpoints from O and from E, checked with the evaluation's sampling settings."""
    ck = pd.read_csv(os.path.join(DATA, "checkpoints.csv")).set_index("check")
    runs = [("G-O", "start-O", "GRPO from O (G-O)", 100, "from O"),
            ("G-E2", "start-E", "GRPO from E (G-E2)", 50, "from E")]
    panels = [("val2_pass", "Validation pass rate,\nnew tests (%)", (0, 25)),
              ("panel_mean", "Panel mean (%)", (20, 62)),
              ("val_cut", "Validation answers cut\nat 32k tokens (%)", (0, 80))]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.45))
    for ax, (col, title, lim) in zip(axes, panels):
        for arm, start, _, pick, short in runs:
            ys = [ck.loc[start, col]] + [ck.loc[f"{arm}/step_{s}", col] for s in STEPS[1:]]
            ax.plot(STEPS, ys, color=COL[arm], lw=1.3, marker="o", ms=3, zorder=3)
            ax.plot([pick], [ys[STEPS.index(pick)]], "o", ms=7.5, mfc="none", mec=COL[arm], mew=1.0, zorder=4)
            if col == "val_cut":
                ax.text(160, ys[-1], short, color=INK2, fontsize=8, va="center", clip_on=False)
        ax.set_xticks(STEPS[::2])
        ax.set_xlim(-6, 156)
        ax.set_ylim(*lim)
        ax.yaxis.grid(True)
        ax.set_title(title)
        ax.set_xlabel("GRPO step")
    handles = [Line2D([], [], color=COL[a], lw=1.3, marker="o", ms=3, label=lab) for a, _, lab, _, _ in runs]
    handles.append(Line2D([], [], ls="none", marker="o", ms=7.5, mfc="none", mec=INK2, mew=1.0,
                          label="checkpoint the rule picked"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    save(fig, "training")


def fig_hacking():
    """The first GRPO run from E, rewarded on APPS problems tested only on their statement's own examples."""
    ck = pd.read_csv(os.path.join(DATA, "checkpoints.csv")).set_index("check")
    keys = ["start-E"] + [f"G-E/step_{s}" for s in STEPS[1:]]
    old = [ck.loc[k, "val_pass"] for k in keys]
    new = [ck.loc[k, "val2_pass"] for k in keys]
    fig, ax = plt.subplots(figsize=(0.55 * TEXT, 2.2))
    ax.plot(STEPS, old, color=MUTED, lw=1.3, marker="o", ms=3)
    ax.plot(STEPS, new, color=INK, lw=1.3, marker="o", ms=3)
    # labels above the right-hand end of each line, where both are flat
    ax.text(150, max(old[3:]) + 4, "tests the reward used\n(the statement's examples)", color=INK2, fontsize=8,
            ha="right", va="bottom")
    ax.text(150, max(new[3:]) + 3, "new tests it never saw", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(STEPS[::2])
    ax.set_xlim(-6, 156)
    ax.set_ylim(0, 88)
    ax.set_yticks(range(0, 81, 20))
    ax.yaxis.grid(True)
    ax.set_xlabel("GRPO step (step 0 is E)")
    ax.set_ylabel("Validation pass rate (%)")
    save(fig, "reward_hacking")


def fig_combine():
    """Paired differences for the questions about combining editing and training."""
    p = pd.read_csv(os.path.join(FDATA, "pairs.csv"))
    comps = [("G-O", "E", "GRPO from O\nminus the best edit"),
             ("G-E2", "G-O", "GRPO from E\nminus GRPO from O"),
             ("T", "G-O", "G-O plus E's change\nminus G-O"),
             ("pilot-20", "G-O", "pilot-20 (edit of G-O)\nminus G-O")]
    order = SUITES + ["mean5"]
    names = [SUITE[s] for s in SUITES] + ["Mean of the five"]
    ys = [-i - (0.3 if s == "mean5" else 0) for i, s in enumerate(order)]
    fig, axes = plt.subplots(1, 4, figsize=(TEXT, 2.5), sharey=True)
    for ax, (a, b, title) in zip(axes, comps):
        for yy, suite in zip(ys, order):
            r = p[(p.a == a) & (p.b == b) & (p.suite == suite) & (p.subset == "all")].iloc[0]
            bold = suite == "mean5"
            ax.errorbar(r["diff"], yy, xerr=2 * r["se"], fmt="o", color=INK, ecolor=INK if bold else INK2,
                        elinewidth=1.3 if bold else 0.9, ms=4.5 if bold else 3.2, capsize=0, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        ax.set_xlim(-11, 17)
        ax.set_xticks([-10, 0, 10])
        ax.xaxis.grid(True)
        ax.set_title(title)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys, names)
    axes[0].get_yticklabels()[-1].set_fontfamily(BOLD)
    fig.supxlabel(r"Paired difference in points (dots), $\pm$2 SE (bars)", fontsize=9)
    save(fig, "combining")


def fig_endings():
    """How answers end on the two suites where looping matters most: cut at the limit, finished after closing
    the thinking more than once, or finished cleanly. Shares of all answers."""
    s = shapes()
    models = ["O", "E", "R-O", "G-O", "G-E2", "T", "pilot-20"]
    parts = [("Cut at the limit", "#4d4d4d", "white"),
             ("Finished, thinking closed 2+ times", "#a6a6a6", INK),
             ("Finished, thinking closed once", "#e3e3e3", INK)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.65), sharey=True)
    fig.get_layout_engine().set(wspace=0.05)  # keeps the left panel's 100 clear of the right panel's 0
    y = np.arange(len(models))[::-1]
    for ax, suite in zip(axes, ["lcb", "math500"]):
        for yi, m in zip(y, models):
            r = s[(m, suite)]
            reopen = 100 * r["reclose_answers"] / r["answers"]
            left = 0.0
            for (name, color, tc), v in zip(parts, [r["cut"], reopen, 100 - r["cut"] - reopen]):
                ax.barh(yi, v, left=left, height=0.66, color=color, edgecolor="white", linewidth=0.6, zorder=2,
                        label=name if yi == y[0] else None)
                if v >= 9:
                    ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8, color=tc)
                left += v
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 25))
        ax.set_title(SUITE[suite])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Share of answers (%)")
    axes[0].set_yticks(y, models)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save(fig, "answer_endings")


def fig_paper():
    """Against the original paper's own published answers, on the three suites both evaluated."""
    p = pd.read_csv(os.path.join(FDATA, "paper_compare.csv"))
    sc = p[p.kind == "score"].set_index(["a", "suite"])["value"]
    suites = ["mmlupro", "ifeval", "math500"]
    series = [("paper-unedited", "Paper: original model"), ("paper-edit", "Paper: its 50-pass math edit"),
              ("E", "This work: best edit (E)"), ("G-O", "This work: GRPO from O (G-O)")]
    fig, ax = plt.subplots(figsize=(0.72 * TEXT, 2.3))
    n, width = len(series), 0.82
    w = width / n
    x = np.arange(len(suites))
    for i, (k, name) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, [sc[(k, su)] for su in suites], width=w, color=COL[k], edgecolor="white",
               linewidth=0.5, label=name, zorder=2)
    ax.set_xticks(x, [SUITE[su] for su in suites])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Score (%)")
    # two columns: the paper's bars in the first, mine in the second
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "paper_comparison")


def probe_table():
    """The layer-23 switches' 'on' shares from train/analysis/probes/summary.md: {unit: {model: percent}}."""
    text = open(os.path.join(ROOT, "train", "analysis", "probes", "summary.md"), encoding="utf-8").read()
    block = text.split("The layer-23 switches")[1].split("\n\n")[1]
    lines = [ln for ln in block.splitlines() if ln.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    out = {}
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        out[cells[0]] = {h: float(c.split("%")[0]) for h, c in zip(head[2:], cells[2:])}
    return out


def fig_probes():
    """Do E's switches still fire where the agent built them to, as GRPO trains from E?"""
    t = probe_table()
    units = [("L23.2664", "cycle stop", INK, 1.5), ("L23.3109", "brake", MUTED, 1.1),
             ("L23.pair", "box pair", "#b3b3b3", 1.1)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.35), sharey=True)
    for ax, arm, title in zip(axes, ["G-E", "G-E2"], ["First GRPO run from E (G-E)", "GRPO from E (G-E2)"]):
        for unit, name, color, lw in units:
            ys = [t[unit]["E"]] + [t[unit][f"{arm}/step_{s}"] for s in STEPS[1:]]
            ax.plot(STEPS, ys, color=color, lw=lw, marker="o", ms=3, zorder=3)
            if arm == "G-E2":
                ax.text(158, ys[-1], name, color=INK2, fontsize=8, va="center", clip_on=False)
        ax.set_xticks(STEPS[::2])
        ax.set_xlim(-6, 152)
        ax.set_ylim(-4, 90)
        ax.yaxis.grid(True)
        ax.set_title(title)
        ax.set_xlabel("GRPO step (step 0 is E)")
    axes[0].set_ylabel("Labelled positions\nwhere it fires (%)")
    save(fig, "edit_survival")


def fig_dose():
    """pilot-20's teacher-forced check of how hard to push toward closing the thinking. The bins are the ones
    the agent reported in runs/pilot-20/final_answer.md."""
    bins = [r"$<0.1$", f"0.3{ENDASH}0.6", f"0.6{ENDASH}0.9", f"0.9{ENDASH}1.2", r"$\geq 1.2$"]
    p = [0.0007, 0.02, 0.11, 0.24, 0.64]
    fig, ax = plt.subplots(figsize=(0.5 * TEXT, 2.3))
    # dots rather than bars: on a log axis a bar's length depends on where the axis happens to start
    ax.plot(range(len(bins)), p, color=COL["pilot-20"], lw=1.1, marker="o", ms=5.5, mec="white", mew=0.8,
            zorder=3)
    for i, v in enumerate(p):
        # above each dot, except the first, where the line to the second dot would run through it
        ax.text(i, v / 1.6 if i == 0 else v * 1.6, f"{v:g}", ha="center", va="top" if i == 0 else "baseline",
                fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(1.5e-4, 2)
    ax.set_xlim(-0.6, len(bins) - 0.4)  # room for the first value's label clear of the y tick labels
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(len(bins)), bins)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True)
    ax.set_xlabel("Push along the model's own exit direction")
    ax.set_ylabel("Probability of closing\nthe thinking")
    save(fig, "exit_dose")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_headline, fig_overview, fig_training, fig_hacking, fig_combine, fig_endings, fig_paper,
              fig_probes, fig_dose):
        f()

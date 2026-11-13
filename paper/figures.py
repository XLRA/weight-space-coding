"""Figures for the write-up (paper/draft/) and the README.

Reads paper/data/ (paper/collect.py), paper/figures/data/ (paper/figure_data.py, run on the server) and
train/analysis/probes/summary.md, and writes paper/figures/<name>.pdf and .png.

    python paper/figures.py

Needs matplotlib, numpy and pandas. The server's venv has no matplotlib, so these were drawn on another machine
(matplotlib 3.10, pandas 2.2); every number comes from the tables above.
"""
import math
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

# Colours. Each model keeps one colour in every figure; the categorical hues are a fixed order checked for
# colour-vision deficiencies on white. Aqua (pilot-20) and yellow (T) are below 3:1 contrast on white, so every
# mark in those colours is labelled. Figures about something other than a model use neutral greys.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"O": "#a3a19a", "E": "#2a78d6", "G-O": "#eb6834", "pilot-20": "#1baf7a", "T": "#eda100",
       "G-E2": "#6250d6", "R-O": "#008300", "paper-unedited": "#c3c2b7", "paper-edit": "#52514e"}
SUITES = ["code", "lcb", "mmlupro", "ifeval", "math500"]
SUITE = {"code": "MBPP+ and HumanEval+", "lcb": "LiveCodeBench", "mmlupro": "MMLU-Pro", "ifeval": "IFEval",
         "math500": "MATH-500"}
STEPS = [0, 25, 50, 75, 100, 125, 150]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 6, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.0, "legend.columnspacing": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "xtick.major.size": 2.5, "ytick.major.size": 0,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), facecolor="white")
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
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True, gridspec_kw={"wspace": 0.1})
    h = 0.26
    y = np.arange(len(SUITES))[::-1].astype(float)
    for ax, col, title in zip(axes, ["pass", "cut"], ["Score (%)", "Answers cut at the 32,768-token limit (%)"]):
        for i, m in enumerate(models):
            ax.barh(y + (1 - i) * h, [s[(m, su)][col] for su in SUITES], height=h, color=COL[m],
                    edgecolor="white", linewidth=0.8, label=names[i], zorder=2)
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.xaxis.grid(True)
        ax.set_title(title)
    axes[0].set_yticks(y, [SUITE[x] for x in SUITES])
    axes[0].set_ylim(-0.55, len(SUITES) - 0.45)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.11), ncol=3, borderaxespad=0)
    save(fig, "headline")


def five_suite_mean(ev, name):
    rows = ev[(ev["eval"] == name) & (ev["subset"] == "all") & (ev["suite"].isin(SUITES))]
    assert len(rows) == 5, name
    return rows["diff"].mean(), math.sqrt((rows["se"] ** 2).sum()) / 5


def fig_overview():
    """Every evaluated model: the mean over the five suites of its paired difference with the original."""
    ev = pd.read_csv(os.path.join(DATA, "evals.csv"))
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
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    y = 0
    ticks, labels, weights = [], [], []
    for label, name, color, weight in rows:
        if name is None:
            y -= 0.35 if y else 0
            ax.text(-0.5, y, label, fontsize=8, color=INK, ha="left", va="center", fontweight="bold",
                    transform=ax.get_yaxis_transform())
            y -= 1
            continue
        m, se = five_suite_mean(ev, name)
        ax.errorbar(m, y, xerr=2 * se, fmt="o", color=COL[color], ecolor=COL[color], elinewidth=1.2,
                    markersize=4.5, capsize=0, zorder=3)
        ax.text(1.005, y, f"{m:+.1f}", transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK2,
                va="center", ha="left")
        ticks.append(y)
        labels.append(label)
        weights.append(weight)
        y -= 1
    ax.set_yticks(ticks, labels)
    for t, w in zip(ax.get_yticklabels(), weights):
        if w:
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(0, 34)
    ax.set_xticks(range(0, 31, 5))
    ax.xaxis.grid(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean gain over the original model across the five suites (points; bars are ±2 SE)")
    save(fig, "overview")


def fig_training():
    """GRPO checkpoints from O and from E, checked with the evaluation's sampling settings."""
    ck = pd.read_csv(os.path.join(DATA, "checkpoints.csv")).set_index("check")
    runs = [("G-O", "start-O", "GRPO from the original model (G-O)", 100, "from O"),
            ("G-E2", "start-E", "GRPO from the edit E (G-E2)", 50, "from E")]
    panels = [("val2_pass", "Validation pass rate,\nnew tests (%)", (0, 25)),
              ("panel_mean", "Panel mean (%)", (20, 62)),
              ("val_cut", "Validation answers cut\nat 32k tokens (%)", (0, 80))]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2), gridspec_kw={"wspace": 0.34})
    for ax, (col, title, lim) in zip(axes, panels):
        for arm, start, _, pick, short in runs:
            ys = [ck.loc[start, col]] + [ck.loc[f"{arm}/step_{s}", col] for s in STEPS[1:]]
            ax.plot(STEPS, ys, color=COL[arm], lw=1.6, marker="o", ms=3.2, zorder=3)
            ax.plot([pick], [ys[STEPS.index(pick)]], "o", ms=8, mfc="none", mec=COL[arm], mew=1.2, zorder=4)
            if col == "val_cut":
                ax.text(160, ys[-1], short, color=INK2, fontsize=7.5, va="center", clip_on=False)
        ax.set_xticks(STEPS[::2])
        ax.set_xlim(-6, 156)
        ax.set_ylim(*lim)
        ax.yaxis.grid(True)
        ax.set_title(title)
        ax.set_xlabel("GRPO step")
    handles = [Line2D([], [], color=COL[a], lw=1.6, marker="o", ms=3.2, label=lab) for a, _, lab, _, _ in runs]
    handles.append(Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=INK2, mew=1.2,
                          label="checkpoint the rule picked"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.11, 1.0), ncol=3, borderaxespad=0)
    save(fig, "training")


def fig_hacking():
    """The first GRPO run from E, rewarded on APPS problems tested only on their statement's own examples."""
    ck = pd.read_csv(os.path.join(DATA, "checkpoints.csv")).set_index("check")
    keys = ["start-E"] + [f"G-E/step_{s}" for s in STEPS[1:]]
    old = [ck.loc[k, "val_pass"] for k in keys]
    new = [ck.loc[k, "val2_pass"] for k in keys]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(STEPS, old, color=MUTED, lw=1.6, marker="o", ms=3.2)
    ax.plot(STEPS, new, color=INK, lw=1.6, marker="o", ms=3.2)
    ax.text(154, old[-1], "tests the reward used\n(the statement's examples)", color=INK2, fontsize=7,
            va="center")
    ax.text(154, new[-1] + 1, "new tests it never saw", color=INK2, fontsize=7, va="center")
    ax.set_xticks(STEPS[::2])
    ax.set_xlim(-6, 150)
    ax.set_ylim(0, 75)
    ax.yaxis.grid(True)
    ax.set_xlabel("GRPO step (step 0 is E)")
    ax.set_title("Validation pass rate (%)")
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
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.35), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (a, b, title) in zip(axes, comps):
        for yy, suite in zip(ys, order):
            r = p[(p.a == a) & (p.b == b) & (p.suite == suite) & (p.subset == "all")].iloc[0]
            bold = suite == "mean5"
            ax.errorbar(r["diff"], yy, xerr=2 * r["se"], fmt="o", color=INK, ecolor=INK if bold else INK2,
                        elinewidth=1.4 if bold else 1, ms=4.5 if bold else 3.5, capsize=0, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_xlim(-11, 17)
        ax.set_xticks([-10, 0, 10])
        ax.xaxis.grid(True)
        ax.set_title(title, fontsize=7.5)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(ys, names)
    last = axes[0].get_yticklabels()[-1]
    last.set_fontweight("bold")
    last.set_color(INK)
    fig.text(0.5, -0.04, "Paired difference in points (dots), ±2 SE (bars)", ha="center", fontsize=8, color=INK2)
    save(fig, "combining")


def fig_endings():
    """How answers end on the two suites where looping matters most: cut at the limit, finished after closing
    the thinking more than once, or finished cleanly. Shares of all answers."""
    s = shapes()
    models = ["O", "E", "R-O", "G-O", "G-E2", "T", "pilot-20"]
    parts = [("Cut at the token limit", INK2, "white"),
             ("Finished, thinking closed 2+ times", "#a9a79f", INK),
             ("Finished, thinking closed once", "#e6e4dd", INK)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(models))[::-1]
    for ax, suite in zip(axes, ["lcb", "math500"]):
        for yi, m in zip(y, models):
            r = s[(m, suite)]
            reopen = 100 * r["reclose_answers"] / r["answers"]
            left = 0.0
            for (name, color, tc), v in zip(parts, [r["cut"], reopen, 100 - r["cut"] - reopen]):
                ax.barh(yi, v, left=left, height=0.64, color=color, edgecolor="white", linewidth=1.0, zorder=2,
                        label=name if yi == y[0] else None)
                if v >= 9:
                    ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7, color=tc)
                left += v
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 25))
        ax.set_title(SUITE[suite])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Share of answers (%)")
    axes[0].set_yticks(y, models)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, borderaxespad=0)
    save(fig, "answer_endings")


def fig_paper():
    """Against the original paper's own published answers, on the three suites both evaluated."""
    p = pd.read_csv(os.path.join(FDATA, "paper_compare.csv"))
    sc = p[p.kind == "score"].set_index(["a", "suite"])["value"]
    suites = ["mmlupro", "ifeval", "math500"]
    series = [("paper-unedited", "Paper: original model"), ("paper-edit", "Paper: its 50-pass math edit"),
              ("E", "Ours: best edit (E)"), ("G-O", "Ours: GRPO from O (G-O)")]
    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    n, width = len(series), 0.82
    w = width / n
    x = np.arange(len(suites))
    for i, (k, name) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, [sc[(k, su)] for su in suites], width=w, color=COL[k], edgecolor="white",
               linewidth=0.9, label=name, zorder=2)
    ax.set_xticks(x, [SUITE[su] for su in suites])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True)
    ax.set_ylim(0, 80)
    ax.set_title("Score (%)")
    ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
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
    units = [("L23.2664", "cycle stop", INK, 1.8), ("L23.3109", "brake", MUTED, 1.2),
             ("L23.pair", "box pair", "#b9b7af", 1.2)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.25), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, arm, title in zip(axes, ["G-E", "G-E2"], ["First GRPO run from E (G-E)", "GRPO from E (G-E2)"]):
        for unit, name, color, lw in units:
            ys = [t[unit]["E"]] + [t[unit][f"{arm}/step_{s}"] for s in STEPS[1:]]
            ax.plot(STEPS, ys, color=color, lw=lw, marker="o", ms=3, zorder=3)
            if arm == "G-E2":
                ax.text(156, ys[-1], name, color=INK2, fontsize=7.5, va="center", clip_on=False)
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
    bins = ["<0.1", "0.3–0.6", "0.6–0.9", "0.9–1.2", "≥1.2"]
    p = [0.0007, 0.02, 0.11, 0.24, 0.64]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    # dots rather than bars: on a log axis a bar's length depends on where the axis happens to start
    ax.plot(range(len(bins)), p, color=COL["pilot-20"], lw=1.2, marker="o", ms=6, mec="white", mew=1, zorder=3)
    for i, v in enumerate(p):
        ax.text(i, v * 1.6, f"{v:g}", ha="center", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(len(bins)), bins)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True)
    ax.set_xlabel("Push along the model's own exit direction")
    ax.set_title("P(</think>) at newlines inside loops")
    save(fig, "exit_dose")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_headline, fig_overview, fig_training, fig_hacking, fig_combine, fig_endings, fig_paper,
              fig_probes, fig_dose):
        f()

"""
Violin plots of normalized final objective values, one panel per dimension
(or any other grouping), one violin per method (solver).

Give the script the CSV files to use, or folders containing them:

    python plot_violin.py path/to/logs

File names must follow  <method>_<problem>_<panel>.csv
To use another naming scheme, edit parse_name().

Each CSV has one row per evaluation, with a seed column ("seed" or "trial")
and an objective column (default "y"; change it with --value-col).
Rows are assumed to be in evaluation order.

For every seed, the script keeps the best (lowest) value found, min-max
normalizes it per (panel, problem) over all methods and seeds
(0 = best, 1 = worst), and plots the distribution for each method.

The width of a violin shows how the seeds' normalized
values are distributed. The thick horizontal line inside each violin is the
median over seed and the thin lines at the ends mark the minimum and maximum.

------------------------------------------------------------------------
Example
------------------------------------------------------------------------

Folder:

    logs/
    ├── solverA_sphere_10.csv
    ├── solverA_sphere_50.csv
    ├── solverA_rosenbrock_10.csv
    ├── solverA_rosenbrock_50.csv
    ├── solverB_sphere_10.csv
    ├── ...
    └── solverC_rosenbrock_50.csv

Content of logs/solverA_sphere_10.csv:

    seed,y
    0,41.20
    0,38.75
    0,22.10
    ...
    1,52.10
    1,47.30
    ...

Command:

    python plot_violin.py logs --out violin.pdf

Result: violin.pdf with two panels (10d and 50d) and three violins in each
(solverA, solverB, solverC), plus a table of mean ± standard error of the
normalized values printed in the terminal.

Options:

    --value-col best_so_far   use another objective column
    --n-init 20 --budget 100  only count the first 20 + 100 evaluations
    --out figures/violin.png  choose the output file and format
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 16,
})


def natural_key(text):
    """Sort strings with numbers in numeric order."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def find_csv_files(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files += glob.glob(os.path.join(path, "*.csv"))
        else:
            files.append(path)
    return sorted(files)


def parse_name(path):
    """Read (method, problem, panel) from <method>_<problem>_<panel>.csv.

    Edit this function to match another naming scheme.
    """
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    if len(parts) < 3:
        raise ValueError("name must look like <method>_<problem>_<panel>.csv")
    return parts[0], "_".join(parts[1:-1]), parts[-1]


def load_results(files, value_col, n_init, budget):
    rows = []
    for path in files:
        try:
            method, problem, panel = parse_name(path)
            df = pd.read_csv(path)
            # Name of the seed column in your logs
            seed_col = "seed" if "seed" in df.columns else "trial"
            for seed, run in df.groupby(seed_col):
                values = pd.to_numeric(run[value_col], errors="coerce").dropna()
                if budget is not None:
                    values = values.iloc[: n_init + budget]
                if len(values):
                    rows.append({
                        "method": method,
                        "problem": problem,
                        "panel": panel,
                        "seed": seed,
                        "final_value": values.min(),
                    })
        except Exception as e:
            print(f"[skip] {path}: {e}")
    return pd.DataFrame(rows)


def normalize(df):
    def scale(v):
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else v * 0.0

    df["normalized"] = df.groupby(["panel", "problem"])["final_value"].transform(scale)
    return df


def plot(df, out):
    methods = sorted(df["method"].unique(), key=natural_key)
    panels = sorted(df["panel"].unique(), key=natural_key)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_of = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.5 * len(panels) + 1, 5.5), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        sub = df[df["panel"] == panel]
        names = [m for m in methods if (sub["method"] == m).any()]
        data = [sub.loc[sub["method"] == m, "normalized"] for m in names]

        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for body, name in zip(parts["bodies"], names):
            body.set_facecolor(color_of[name])
            body.set_edgecolor("black")
            body.set_alpha(0.45)
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(3.0)

        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        # Show "20" as "20d" when the panel is a dimension
        title = f"{panel}d" if panel.isdigit() else panel
        ax.set_title(title, fontweight="bold")

    axes[0].set_ylabel(r"Normalized $f_{\mathrm{final}}$")
    fig.tight_layout()

    # Explain the lines drawn inside each violin
    legend_lines = [
        Line2D([], [], color="black", linewidth=3.0, label="Median"),
        Line2D([], [], color="black", linewidth=1.0, label="Min / max"),
    ]
    fig.legend(handles=legend_lines, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=16)

    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="CSV files or folders of CSV files.")
    parser.add_argument("--value-col", default="y", help="Objective column (default: y).")
    parser.add_argument("--n-init", type=int, default=0, help="Initial design size.")
    parser.add_argument("--budget", type=int, default=None,
                        help="Evaluations counted after the initial design (default: all).")
    parser.add_argument("--out", default="violin.pdf", help="Output figure.")
    args = parser.parse_args()

    files = find_csv_files(args.paths)
    df = load_results(files, args.value_col, args.n_init, args.budget)
    if df.empty:
        raise SystemExit("No results loaded. Check the paths and --value-col.")

    df = normalize(df)

    stats = df.groupby(["panel", "method"])["normalized"].agg(["mean", "sem"])
    table = stats.round(3).unstack("method")
    table = table.reindex(sorted(table.index, key=natural_key))
    print(table.to_string())

    plot(df, args.out)


if __name__ == "__main__":
    main()

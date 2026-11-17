"""Aggregate T16 results: summary statistics + comparison plots.

Reads results/results.csv (produced by run_experiments.py), writes
results/summary.csv, results/analysis.md and plots/*.png.

Statistics are grouped per (dataset, solver, loss, reg, init, impl) across
seeds. Implementation gaps are reported BOTH per-seed (paired by case, even
though identical seeds do not imply identical initialization) and
best-of-seeds, which is the fairer comparison under non-unique local optima.

Usage:  python analyze_results.py [--results DIR]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def load_rows(results_dir: Path):
    with (results_dir / "results.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def group_key(r):
    return (r["dataset"], r["solver"], r["loss"], r["reg_name"], r["init"], r["impl"])


def stats(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return dict(n=0, mean="", std="", min="", median="", max="")
    return dict(
        n=len(vals),
        mean=statistics.mean(vals),
        std=statistics.stdev(vals) if len(vals) > 1 else 0.0,
        min=min(vals),
        median=statistics.median(vals),
        max=max(vals),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args()
    results_dir = Path(args.results)
    plots_dir = HERE / "plots"
    plots_dir.mkdir(exist_ok=True)
    rows = load_rows(results_dir)
    ok = [r for r in rows if r["status"] == "ok" and r["objective_status"] == "ok"]

    groups = defaultdict(list)
    for r in ok:
        groups[group_key(r)].append(r)

    summary_fields = [
        "dataset", "solver", "loss", "reg_name", "init", "impl", "n_runs",
        "objective_mean", "objective_std", "objective_min", "objective_median",
        "data_loss_mean", "recon_mean", "penalty_mean",
        "n_iter_mean", "fit_seconds_mean",
    ]
    summary_rows = []
    for key in sorted(groups):
        g = groups[key]
        obj = stats([fnum(r["full_objective"]) for r in g])
        dl = stats([fnum(r["data_loss"]) for r in g])
        rec = stats([fnum(r["reconstruction_error"]) for r in g])
        pen = stats([fnum(r["regularization_penalty"]) for r in g])
        nit = stats([fnum(r["n_iter"]) for r in g])
        tim = stats([fnum(r["fit_seconds"]) for r in g])
        summary_rows.append(dict(zip(summary_fields, [
            *key, obj["n"], obj["mean"], obj["std"], obj["min"], obj["median"],
            dl["mean"], rec["mean"], pen["mean"], nit["mean"], tim["mean"],
        ])))
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # Paired and best-of-seeds gaps (nmflib relative to sklearn).
    by_case = defaultdict(dict)
    for r in ok:
        case = (r["dataset"], r["solver"], r["loss"], r["reg_name"], r["init"], r["seed"])
        by_case[case][r["impl"]] = fnum(r["full_objective"])
    paired_gaps = defaultdict(list)
    for case, impls in by_case.items():
        if "nmflib" in impls and "sklearn" in impls:
            jn, js = impls["nmflib"], impls["sklearn"]
            if js and js > 0:
                paired_gaps[case[:5]].append((jn - js) / js)
    best = defaultdict(lambda: defaultdict(list))
    for r in ok:
        best[(r["dataset"], r["solver"], r["loss"], r["reg_name"], r["init"])][r["impl"]].append(
            fnum(r["full_objective"]))
    best_gap_rows = []
    for key in sorted(best):
        impls = best[key]
        if "nmflib" in impls and "sklearn" in impls:
            bn, bs = min(impls["nmflib"]), min(impls["sklearn"])
            best_gap_rows.append((key, bn, bs, (bn - bs) / bs if bs > 0 else float("nan")))

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in ok})
    combos = [("MU", "frobenius"), ("CD", "frobenius"), ("MU", "kl")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, ds in zip(axes.ravel(), datasets):
        data, labels = [], []
        for solver, loss in combos:
            for impl in ("nmflib", "sklearn"):
                vals = [fnum(r["full_objective"]) for r in ok
                        if r["dataset"] == ds and r["solver"] == solver
                        and r["loss"] == loss and r["reg_name"] == "none"
                        and r["init"] == "Random" and r["impl"] == impl]
                data.append(vals)
                labels.append(f"{solver}-{loss[:4]}\n{impl}")
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_title(f"{ds}: full objective, no reg, Random, 5 seeds")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("T16: independent full objective (lower is better)")
    fig.tight_layout()
    fig.savefig(plots_dir / "objective_by_combo.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, ds in zip(axes.ravel(), datasets):
        data, labels = [], []
        for solver, loss in combos:
            for impl in ("nmflib", "sklearn"):
                vals = [fnum(r["reconstruction_error"]) for r in ok
                        if r["dataset"] == ds and r["solver"] == solver
                        and r["loss"] == loss and r["reg_name"] == "none"
                        and r["init"] == "Random" and r["impl"] == impl]
                data.append(vals)
                labels.append(f"{solver}-{loss[:4]}\n{impl}")
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(f"{ds}: ||V-WH||_F, no reg, Random")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "reconstruction_error.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    seeds = sorted({r["seed"] for r in ok})
    data, labels = [], []
    for seed in seeds:
        for impl in ("nmflib", "sklearn"):
            vals = [fnum(r["full_objective"]) for r in ok
                    if r["dataset"] == "A2" and r["solver"] == "MU"
                    and r["loss"] == "frobenius" and r["reg_name"] == "none"
                    and r["init"] == "Random" and r["impl"] == impl
                    and r["seed"] == seed]
            data.append(vals)
            labels.append(f"{seed}\n{impl[:3]}")
    ax.boxplot(data, tick_labels=labels)
    ax.set_title("A2 MU-Frobenius no-reg: per-seed objective distribution")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "seed_distribution.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    reg_names = ["none", "l1", "l2", "l1l2"]
    for ax, metric, title in [
        (axes[0], "full_objective", "full objective J"),
        (axes[1], "regularization_penalty", "penalty Omega"),
        (axes[2], "reconstruction_error", "||V-WH||_F"),
    ]:
        for impl, marker in (("nmflib", "o"), ("sklearn", "s")):
            means = []
            for reg in reg_names:
                vals = [fnum(r[metric]) for r in ok
                        if r["dataset"] == "A2" and r["solver"] == "MU"
                        and r["loss"] == "frobenius" and r["init"] == "Random"
                        and r["impl"] == impl and r["reg_name"] == reg]
                means.append(statistics.mean(vals) if vals else float("nan"))
            ax.plot(reg_names, means, marker=marker, label=impl)
        ax.set_title(f"A2 MU-Frobenius: {title}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "regularization_effect.png", dpi=130)
    plt.close(fig)

    # ---- analysis.md ----
    lines = ["# T16 analysis (auto-generated by analyze_results.py)", ""]
    lines.append(f"rows total: {len(rows)}; evaluable ok rows: {len(ok)}")
    errs = [r for r in rows if r["status"] == "error"]
    klinf = [r for r in rows if r["objective_status"].startswith("kl_infinite")]
    lines.append(f"error runs: {len(errs)}; kl_infinite objectives: {len(klinf)}")
    for r in errs:
        lines.append(f"- ERROR {r['impl']} {r['run_id']}: {r['error']}")
    for r in klinf:
        lines.append(f"- KL-INFINITE {r['impl']} {r['run_id']}: {r['objective_status']}")
    lines += ["", "## Best-of-seeds objective gap (nmflib vs sklearn)", "",
              "| dataset | solver | loss | reg | init | J_nmf_best | J_sk_best | rel gap |",
              "|---|---|---|---|---|---:|---:|---:|"]
    for key, bn, bs, gap in best_gap_rows:
        lines.append(f"| {' | '.join(key)} | {bn:.6g} | {bs:.6g} | {gap:+.3%} |")
    lines += ["", "## Paired per-seed relative gap stats (J_nmf - J_sk)/J_sk", "",
              "| dataset | solver | loss | reg | init | n | mean | median | min | max |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for key in sorted(paired_gaps):
        g = paired_gaps[key]
        lines.append(
            f"| {' | '.join(key)} | {len(g)} | {statistics.mean(g):+.3%} | "
            f"{statistics.median(g):+.3%} | {min(g):+.3%} | {max(g):+.3%} |")
    (results_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote summary.csv ({len(summary_rows)} groups), analysis.md, "
          f"and 4 plots to {plots_dir}")


if __name__ == "__main__":
    main()
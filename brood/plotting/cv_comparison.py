#!/usr/bin/env python3
"""Cross-validation figures for average precision.

Reads per-sample metrics from ``data/runs/<dataset>/*/metrics/cv.csv`` and
renders three figures:

1. ``<dataset>_cv_ap.svg`` - every model's average precision, one faint dot
   per CV sample, coloured by model. This is the screening
   leaderboard. Each model's horizontal bar is its 95% simultaneous
   confidence interval; the best model's interval is bounded by two dotted
   vertical lines, so any model whose interval clears those lines is
   significantly worse than the best.
2. ``<dataset>_cv_representation.svg`` - average precision per molecular
   representation, for the later representation stage. Representations are
   nominal so nothing is connected across them; only the two learners at the
   same representation are joined, because that is a paired comparison.
3. ``<dataset>_cv_representation_ap.svg`` - figure 1's leaderboard applied to
   the representation runs.

Screening passes ``--skip-representation`` and gets figure 1 alone; the
representation stage passes ``--skip-leaderboard`` and gets figures 2 and 3
alone. Each stage must name only the figure it owns, otherwise it overwrites
the other stage's figure with the wrong model set.

Statistics follow Ash et al., *J. Chem. Inf. Model.* 2025, 65, 9398-9411
("Practically Significant Method Comparison Protocols"): a repeated-measures
ANOVA whose Error term is the sample-by-model interaction, followed by the
Tukey HSD post hoc (``rm_tukey_hsd``). Each of the 5x5 CV samples is a
subject, every model is measured on every sample. When the parametric
assumptions are violated (non-normal residuals or non-spherical contrasts)
the comparison falls back to Friedman + Conover-Holm; ``--test`` forces
either path.

``--output-dir`` is the root figure directory: the dataset and ``cv`` are
appended automatically, so pass ``figures``, not ``figures/<dataset>/cv``.

Usage
-----
    brood-plot-cv --dataset stokes
    brood-plot-cv --dataset stokes --output-dir figures
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.lines import Line2D
from scipy import stats
from scipy.linalg import helmert
from statsmodels.stats.libqsturng import psturng, qsturng

from brood.core.logging import setup_logging
from brood.plotting.style import (
    LEARNER_COLORS,
    LEARNER_LABELS,
    decorate_axis,
    save_fig,
    set_axes,
    use_style,
)
from brood.representations import REPRESENTATION_REGISTRY

logger = logging.getLogger(__name__)

# Must precede every ``plt.subplots()`` call: seaborn's set_style only reaches
# axes created after it runs.
use_style()

#: Plot order for the representation axis (fixed, so runs stay comparable).
REPRESENTATION_ORDER = tuple(REPRESENTATION_REGISTRY)

#: Human labels for the representation axis.
REPRESENTATION_LABELS = {
    name: rep.label or name for name, rep in REPRESENTATION_REGISTRY.items()
}

#: Models that have one run per representation and so get a line in fig. 2.
#: The first is the minuend of the figure's ``Δ`` label.
_FACTORIAL_LEARNERS = ("tabpfn", "lgbm")

#: Runs without a representation variant, drawn as reference lines in fig. 2.
_REFERENCE_LEARNERS = ("chemprop", "knn")

#: Significance level for the pairwise comparisons.
_ALPHA = 0.05

#: Greenhouse-Geisser epsilon below which sphericity is treated as violated.
_SPHERICITY_MIN = 0.75

#: Shapiro p below which residual normality is treated as violated.
_NORMALITY_MIN = 0.05


@dataclass
class Comparison:
    """Within-subjects comparison of models across CV samples.

    ``p_values`` is the symmetric matrix of adjusted pairwise p-values, so it
    already accounts for every pairwise comparison in ``table``.
    ``ci_half`` is the per-model simultaneous 95% interval half-width; it is
    only defined for the parametric (Tukey) path.
    """

    means: pd.Series
    p_values: pd.DataFrame
    omnibus_p: float
    test: str
    alpha: float = _ALPHA
    ci_half: float | None = None
    reason: str = ""

    def significant(self, a: str, b: str) -> bool:
        """Whether two models differ at the chosen level."""
        return bool(self.p_values.loc[a, b] < self.alpha)

    def best(self) -> str:
        """Name of the highest-mean model."""
        return str(self.means.idxmax())


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--data-dir", default="data", help="Data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the SVGs (default: figures/<dataset>/cv)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only include these model run names in the comparison.",
    )
    parser.add_argument(
        "--test",
        choices=("auto", "parametric", "nonparametric"),
        default="auto",
        help=(
            "Pairwise test: 'parametric' is repeated-measures Tukey HSD, "
            "'nonparametric' is Friedman + Conover-Holm, 'auto' picks the "
            "nonparametric fallback when the Tukey assumptions fail."
        ),
    )
    parser.add_argument(
        "--skip-representation",
        action="store_true",
        help="Skip figures 2 and 3 (the representation-stage figures)",
    )
    parser.add_argument(
        "--skip-leaderboard",
        action="store_true",
        help="Skip figure 1 (the screening leaderboard)",
    )
    return parser.parse_args()


def _load_cv_metrics(
    data_dir: Path, dataset: str, models: list[str] | None = None
) -> pd.DataFrame:
    """Load per-sample CV metrics from each run's ``metrics/cv.csv``.

    These are the values :func:`brood.cv.loop.run_cv_model` already derived
    from the raw fold predictions, so the figures read one source instead of
    recomputing AP/BEDROC/Brier a second time. Repeated CV writes one row per
    (repeat, fold); those are combined into a unique ``sample`` key so the
    paired comparison treats all 5x5 samples as the blocking factor.
    """
    files = sorted((data_dir / "runs" / dataset).glob("*/metrics/cv.csv"))
    if models is not None:
        selected = set(models)
        files = [p for p in files if p.parent.parent.name in selected]
    if not files:
        raise FileNotFoundError(
            f"No CV metrics CSVs found under {data_dir / 'runs' / dataset}"
        )
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if "repeat" in df.columns:
        df["sample"] = df["repeat"].astype(str) + "_" + df["fold"].astype(str)
    else:
        df["sample"] = df["fold"].astype(str)
    return df


def _split_model(name: str) -> tuple[str, str]:
    """'tabpfn_morgan512' -> ('tabpfn', 'morgan512').

    Runs with no representation variant ('chemprop', 'knn') map to
    their own model type with an empty representation, so they keep their
    model colour instead of falling back to the unknown-run grey.
    """
    learner, _, rep = name.partition("_")
    if learner in LEARNER_COLORS:
        return learner, rep
    return "other", name


def _fold_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sample x model matrix of one metric (incomplete samples dropped)."""
    table = (
        df.dropna(subset=[metric])
        .pivot(index="sample", columns="model", values=metric)
        .dropna()
    )
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Need at least 2 samples and 2 models")
    return table


def _warn_on_incomplete_folds(df: pd.DataFrame, table: pd.DataFrame) -> None:
    """Name any run whose CV samples are incomplete.

    A run with fewer samples than the rest forces ``dropna`` to discard those
    samples for *every* model, silently shrinking the comparison.
    """
    coverage = df.groupby("model")["sample"].nunique()
    expected = int(coverage.max())
    short = {model: int(n) for model, n in coverage.items() if n < expected}
    dropped = sorted(set(df["sample"]) - set(table.index))
    if short or dropped:
        detail = ", ".join(
            f"{m}={n}/{expected}" for m, n in sorted(short.items())
        )
        logger.warning(
            "Paired comparison uses only {}/{} samples across {} models; "
            "runs with incomplete samples: {}; samples dropped from every "
            "model: {}. Re-run them before reading the figure.".format(
                table.shape[0],
                expected,
                table.shape[1],
                detail or "none",
                dropped or "none",
            )
        )


def rm_tukey_hsd(table: pd.DataFrame, alpha: float = _ALPHA) -> Comparison:
    """Repeated-measures ANOVA with the Tukey HSD post hoc.

    Each CV sample is a subject and every model is measured on every sample,
    so the Error term is the sample-by-model interaction. That Error mean
    square and its degrees of freedom feed the studentized-range distribution
    exactly as in Ash et al.'s ``rm_tukey_hsd``; the per-model ``ci_half`` is
    the simultaneous 95% interval used by the leaderboard.
    """
    n_samples, n_models = table.shape
    values = table.to_numpy()
    means = table.mean(axis=0)
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    grand_mean = float(values.mean())
    residual = values - row_means[:, None] - col_means[None, :] + grand_mean
    error_df = (n_samples - 1) * (n_models - 1)
    error_mse = float((residual**2).sum() / error_df)

    columns = list(table.columns)
    p_values = pd.DataFrame(
        np.ones((n_models, n_models)), index=columns, columns=columns
    )
    if error_mse:
        se = np.sqrt(2.0 * error_mse / n_samples)
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                diff = float(means[left] - means[right])
                q_stat = abs(diff) / se
                p_val = float(
                    np.atleast_1d(
                        psturng(q_stat * np.sqrt(2.0), n_models, error_df)
                    )[0]
                )
                p_val = min(max(p_val, 0.0), 1.0)
                p_values.loc[left, right] = p_val
                p_values.loc[right, left] = p_val

    model_ss = n_samples * ((col_means - grand_mean) ** 2).sum()
    model_mse = model_ss / (n_models - 1)
    f_stat = model_mse / error_mse if error_mse else np.inf
    omnibus_p = float(stats.f.sf(f_stat, n_models - 1, error_df))

    critical = float(
        np.atleast_1d(qsturng(1.0 - alpha, n_models, error_df))[0]
    )
    ci_half = critical / np.sqrt(2.0) * np.sqrt(error_mse / n_samples)
    return Comparison(
        means=means,
        p_values=p_values,
        omnibus_p=omnibus_p,
        test="parametric",
        alpha=alpha,
        ci_half=float(ci_half),
    )


def friedman_conover(table: pd.DataFrame, alpha: float = _ALPHA) -> Comparison:
    """Nonparametric fallback: Friedman omnibus + Conover pairwise (Holm).

    Used when the Tukey assumptions fail. With two models the Friedman
    omnibus is undefined and the paired Wilcoxon signed-rank test takes its
    place; the post hoc Conover test still works at two groups.
    """
    columns = list(table.columns)
    if len(columns) < 3:
        omnibus_p = float(
            stats.wilcoxon(table[columns[0]], table[columns[1]]).pvalue
        )
    else:
        omnibus_p = float(
            stats.friedmanchisquare(
                *[table[c].to_numpy() for c in columns]
            ).pvalue
        )
    with warnings.catch_warnings():
        # A model with zero between-sample variance makes Conover's test
        # divide by zero; the resulting p-value is still well defined.
        warnings.simplefilter("ignore", RuntimeWarning)
        p_values = sp.posthoc_conover_friedman(table, p_adjust="holm")
    p_values = p_values.reindex(index=columns, columns=columns).fillna(1.0)
    return Comparison(
        means=table.mean(axis=0),
        p_values=p_values,
        omnibus_p=omnibus_p,
        test="nonparametric",
        alpha=alpha,
    )


def _assumptions(table: pd.DataFrame) -> dict[str, float]:
    """Shapiro normality of the residuals and Greenhouse-Geisser epsilon.

    Normality is tested on the mean-centred residuals (the ANOVA error); the
    epsilon measures how far the contrasts are from spherical (1.0 is
    perfect). Both are only diagnostics - the caller decides what to do.
    """
    n_samples, n_models = table.shape
    values = table.to_numpy()
    residual = (
        values
        - values.mean(axis=1)[:, None]
        - values.mean(axis=0)[None, :]
        + values.mean()
    ).ravel()
    normality_p = (
        float(stats.shapiro(residual).pvalue) if residual.size >= 3 else 1.0
    )

    epsilon = 1.0
    if n_models >= 2 and n_samples >= 2:
        contrasts = helmert(n_models)
        covariance = np.cov(values, rowvar=False)
        within = contrasts @ covariance @ contrasts.T
        denom = (n_models - 1) * np.trace(within @ within)
        if denom:
            epsilon = float(np.trace(within) ** 2 / denom)
    return {"normality_p": normality_p, "gg_epsilon": epsilon}


def _violations(diag: dict[str, float]) -> list[str]:
    """Human-readable assumptions that the Tukey path would violate."""
    reasons = []
    if diag["gg_epsilon"] < _SPHERICITY_MIN:
        reasons.append(
            f"sphericity {diag['gg_epsilon']:.2f} < {_SPHERICITY_MIN}"
        )
    if diag["normality_p"] < _NORMALITY_MIN:
        reasons.append(
            f"normality p {diag['normality_p']:.3g} < {_NORMALITY_MIN}"
        )
    return reasons


def _compare(
    table: pd.DataFrame, mode: str
) -> tuple[Comparison, dict[str, float]]:
    """Run the chosen within-subjects comparison on a sample x model table."""
    diag = _assumptions(table)
    if mode == "auto":
        reasons = _violations(diag)
        chosen = "nonparametric" if reasons else "parametric"
    else:
        reasons = _violations(diag)
        chosen = mode
    if chosen == "nonparametric":
        comparison = friedman_conover(table)
        comparison.reason = "; ".join(reasons) or "forced"
    else:
        comparison = rm_tukey_hsd(table)
    return comparison, diag


def _log_comparison(comparison: Comparison, diag: dict[str, float]) -> None:
    """Record the assumption checks and the test that was actually used."""
    if comparison.test == "parametric":
        logger.info(
            "Tukey HSD: normality p={:.3g}, GG epsilon={:.3f}; "
            "omnibus p={:.3g}".format(
                diag["normality_p"], diag["gg_epsilon"], comparison.omnibus_p
            )
        )
    else:
        logger.warning(
            "Nonparametric fallback ({}): normality p={:.3g}, GG "
            "epsilon={:.3f}; omnibus p={:.3g}".format(
                comparison.reason,
                diag["normality_p"],
                diag["gg_epsilon"],
                comparison.omnibus_p,
            )
        )


def _model_color(model: str) -> tuple:
    """Colourblind-safe colour for a model."""
    return LEARNER_COLORS.get(_split_model(model)[0], "#999999")


def _plot_leaderboard(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    metric: str,
    table: pd.DataFrame,
    comparison: Comparison,
    suffix: str | None = None,
) -> None:
    """Figures 1 and 3: mean metric per run with one dot per CV sample.

    Colour encodes the model type. Each horizontal bar is the model's
    simultaneous interval; the best model's interval is bounded by two dotted
    lines, so a model outside them is significantly worse than the best.
    """
    _warn_on_incomplete_folds(df, table)
    means = comparison.means.sort_values(ascending=False)
    best = comparison.best()
    half = comparison.ci_half
    tukey = half is not None

    fig, ax = plt.subplots(
        figsize=(10, 0.46 * len(means) + 2.0), constrained_layout=True
    )
    rng = np.random.default_rng(0)
    for y, model in enumerate(means.index):
        color = _model_color(model)
        vals = table[model].to_numpy()
        ax.scatter(
            vals,
            y + rng.uniform(-0.13, 0.13, len(vals)),
            s=22,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )
        mean = means[model]
        if half is not None:
            ax.plot(
                [mean - half, mean + half],
                [y, y],
                color=color,
                lw=1.6,
                alpha=0.85,
                zorder=4,
            )
        ax.plot(
            [mean, mean],
            [y - 0.28, y + 0.28],
            color=color,
            lw=4.5,
            solid_capstyle="butt",
            zorder=5,
        )
        flag = ""
        if not tukey and model != best:
            flag = " *" if comparison.significant(model, best) else " ns"
        ax.text(
            1.005,
            y,
            f"{mean:.3f}{flag}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color="#444444",
        )

    if half is not None:
        best_mean = float(means[best])
        for bound in (best_mean - half, best_mean + half):
            ax.axvline(bound, ls=":", color="#444444", lw=1.3, zorder=1)
        ax.text(
            best_mean + half,
            -0.6,
            " best 95% CI",
            color="#444444",
            fontsize=9,
            va="center",
            ha="left",
        )

    ax.text(
        1.005,
        -0.9,
        "mean AP",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        fontweight="bold",
        color="#444444",
    )

    ax.set_yticks(range(len(means)), means.index)
    ax.set_ylim(len(means) - 0.4, -1.3)
    ax.tick_params(axis="y", length=0)
    decorate_axis(ax)
    if tukey:
        xlabel = (
            f"Average precision per CV sample (dots), mean (vertical bar) "
            f"and 95% Tukey interval (horizontal bar); "
            f"n_samples={table.shape[0]}"
        )
    else:
        xlabel = (
            f"Average precision per CV sample (dots), mean (vertical bar); "
            f"* = worse than best (Conover-Holm); n_samples={table.shape[0]}"
        )
    set_axes(ax, xlabel=xlabel)
    present = sorted({_split_model(m)[0] for m in table.columns} - {"other"})
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LEARNER_COLORS[learner],
                label=LEARNER_LABELS[learner],
            )
            for learner in present
        ],
        loc="lower right",
    )
    # Same quantity as figure 1 but a different run set, and therefore a
    # different omnibus test: mark it so the two are not confused.
    scope = " (representation stage)" if suffix else ""
    if tukey:
        stat_line = (
            f"repeated-measures ANOVA + Tukey HSD, omnibus F-test "
            f"p={comparison.omnibus_p:.2g}"
        )
    else:
        stat_line = f"Friedman + Conover-Holm, p={comparison.omnibus_p:.2g}"
    ax.set_title(
        f"{dataset}: training-set repeated 5x5 CV "
        f"average precision{scope}\n{stat_line}, α={comparison.alpha}",
        fontsize=12,
    )
    path = output_dir / f"{dataset}_cv_{suffix or metric}.svg"
    save_fig(fig, path)
    logger.info(f"{metric.upper()} leaderboard saved to {path}")


def _plot_representation(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    metric: str,
    table: pd.DataFrame,
    comparison: Comparison,
) -> None:
    """Figure 2: metric per representation, paired within each group.

    Representations are nominal, so nothing is connected across them. The
    two learners at the same representation are joined by a segment because
    that *is* a paired comparison; the segment is drawn solid only when the
    pairwise adjusted p-value is significant.
    """
    _warn_on_incomplete_folds(df, table)
    means = comparison.means

    pairs: dict[str, tuple[str, str]] = {}
    for rep in REPRESENTATION_ORDER:
        cols = [
            f"{learner}_{rep}"
            for learner in _FACTORIAL_LEARNERS
            if f"{learner}_{rep}" in table.columns
        ]
        if len(cols) == 2:
            pairs[rep] = (cols[0], cols[1])
    if not pairs:
        logger.warning(
            "No learner x representation pairs found; skipping figure 2"
        )
        return

    reps = list(pairs)
    xs = np.arange(len(reps))
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    n_tied = 0

    for x, rep in zip(xs, reps):
        left, right = pairs[rep]
        lo, hi = float(means[left]), float(means[right])
        color = LEARNER_COLORS[_split_model(left)[0]]
        significant = comparison.significant(left, right)
        n_tied += not significant
        ax.plot(
            [x, x],
            [lo, hi],
            color=color,
            lw=2.4,
            ls="-" if significant else (0, (2, 2)),
            alpha=0.5,
            zorder=2,
        )
        for model, value in ((left, lo), (right, hi)):
            ax.plot(
                x,
                value,
                marker="o",
                markersize=9,
                color=_model_color(model),
                linestyle="none",
                zorder=4,
            )
        ax.text(
            x,
            max(lo, hi) + 0.004,
            f"{lo - hi:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=LEARNER_COLORS[learner],
            label=LEARNER_LABELS[learner],
        )
        for learner in _FACTORIAL_LEARNERS
    ]
    for learner in _REFERENCE_LEARNERS:
        if learner not in table.columns:
            continue
        value = means[learner]
        color = LEARNER_COLORS[learner]
        ax.axhline(value, ls="--", color=color, lw=1.4, zorder=1)
        ax.text(
            len(reps) - 0.6,
            value,
            f" {LEARNER_LABELS[learner]} {value:.3f}",
            color=color,
            fontsize=9,
            va="bottom",
            ha="right",
        )
        handles.append(
            Line2D([0], [0], color=color, ls="--", lw=1.4, label=learner)
        )
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))

    decorate_axis(ax)
    test_name = (
        "Tukey HSD" if comparison.test == "parametric" else "Conover-Holm"
    )
    set_axes(
        ax,
        ylabel="Average precision",
        xlabel=(
            "Molecular representation (segment labelled "
            f"Δ = {LEARNER_LABELS[_FACTORIAL_LEARNERS[0]]} − "
            f"{LEARNER_LABELS[_FACTORIAL_LEARNERS[1]]})"
        ),
    )
    ax.set_xticks(xs, [REPRESENTATION_LABELS.get(rep, rep) for rep in reps])
    ax.set_xlim(-0.4, len(reps) - 0.6)
    ax.set_title(
        f"{dataset}: representation and model, paired within each "
        f"group\n{metric.upper()}, {test_name}, n_samples={table.shape[0]}, "
        f"{len(reps) - n_tied}/{len(reps)} gaps distinguishable at "
        f"α={comparison.alpha}",
        fontsize=12,
    )
    path = output_dir / f"{dataset}_cv_representation.svg"
    save_fig(fig, path)
    logger.info(f"{metric.upper()} representation figure saved to {path}")


def main() -> None:
    """Render the CV figures."""
    setup_logging()
    args = _parse_args()
    df = _load_cv_metrics(Path(args.data_dir), args.dataset, args.models)
    if "ap" not in df.columns:
        raise ValueError("No average precision available in the CV files")
    metric = "ap"
    table = _fold_table(df.dropna(subset=[metric]), metric)
    comparison, diag = _compare(table, args.test)
    _log_comparison(comparison, diag)
    output_dir = Path(args.output_dir or "figures") / args.dataset / "cv"
    if not args.skip_leaderboard:
        _plot_leaderboard(
            df, args.dataset, output_dir, metric, table, comparison
        )
    if not args.skip_representation:
        _plot_representation(
            df, args.dataset, output_dir, metric, table, comparison
        )
        _plot_leaderboard(
            df,
            args.dataset,
            output_dir,
            metric,
            table,
            comparison,
            suffix="representation_ap",
        )


if __name__ == "__main__":
    main()

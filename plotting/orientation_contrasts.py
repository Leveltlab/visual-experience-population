"""Angular distance and exploratory endpoint/contrast panel variants."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from analysis_config import OUT, AREAS, COLORS, LABELS
from plotting.panels_bcd import MARKERS, OFFSETS, draw_panel
from plotting.statistics import horizontal_bracket
from plotting.style import use_myriad_pro

COS = OUT / "direction_cosine_top2"


def figures(mouse, groups, results, summaries):
    use_myriad_pro()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11})
    handles = [
        Line2D([], [], color=c, marker=m, lw=1, ms=5, label=l)
        for c, m, l in zip(COLORS, MARKERS, LABELS)
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.35), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.25, top=0.77, wspace=0.21)
    for ax, area in zip(axes, AREAS):
        for fi, color in enumerate(COLORS):
            g = (
                groups[(groups.area == area) & (groups.fi == fi)]
                .set_index("difference_degrees")
                .loc[np.arange(1, 8) * 45]
            )
            x = g.index.to_numpy() + OFFSETS[fi] * 6
            m = mouse[(mouse.area == area) & (mouse.fi == fi)].pivot(
                index="mouse", columns="difference_degrees", values="value"
            )
            for values in m[g.index].to_numpy():
                ax.plot(x, values, ls="none", color=color, alpha=0.14, marker=MARKERS[fi], ms=2.3)
            y = g["mean"].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=np.maximum(0, [y - g.ci_low, g.ci_high - y]),
                fmt=MARKERS[fi] + "-",
                color=color,
                ms=4.3,
                lw=1.25,
                capsize=2,
            )
        ax.set(
            title=area,
            xlim=(30, 330),
            xticks=np.arange(1, 8) * 45,
            xlabel="Direction difference Δ (°)",
            ylim=(0, max(mouse.value.max(), groups.ci_high.max()) * 1.08),
        )
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Euclidean distance")
    fig.legend(handles=handles, ncol=4, loc="upper center", frameon=False, fontsize=9)
    fig.text(
        0.53,
        0.85,
        "51.79% + 100% contrast pooled · 30 trials per direction · N = 10 · 1,000 neuron draws",
        ha="center",
        fontsize=9,
        color=".35",
    )
    yield "B_angular_distance", fig, True
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 6), sharey="row")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.86, hspace=0.5, wspace=0.2)
    for row, metric in enumerate(["cosine", "euclidean"]):
        selected = summaries[(summaries.metric == metric) & (summaries.endpoint == "contrast")]
        low = min(0, selected.ci_low.min()) - 0.05 * selected.ci_high.max()
        high = selected.ci_high.max() * 1.3
        for ax, area in zip(axes[row], AREAS):
            for fi, color in enumerate(COLORS):
                g = selected[(selected.area == area) & (selected.fi == fi)].iloc[0]
                y = g["mean"]
                ax.errorbar(
                    fi,
                    y,
                    yerr=np.maximum(0, [[y - g.ci_low], [g.ci_high - y]]),
                    fmt=MARKERS[fi],
                    color=color,
                    ms=5,
                    capsize=2,
                )
            ax.axhline(0, color=".6", ls=":", lw=0.7)
            ax.set(
                title=area if row == 0 else "",
                xticks=[0.5, 2.5],
                xticklabels=["Baseline", "Chronic"],
                xlim=(-0.5, 3.5),
                ylim=(low, high),
            )
            for t in results[
                (results.metric == metric)
                & (results.endpoint == "contrast")
                & (results.area == area)
            ].itertuples():
                fi = 0 if t.timepoint == "baseline" else 2
                ceiling = selected[
                    (selected.area == area) & selected.fi.isin([fi, fi + 1])
                ].ci_high.max()
                horizontal_bracket(
                    ax, fi, fi + 1, (ceiling - low) / (high - low) + 0.04, t.stars, size=8
                )
        axes[row, 0].set_ylabel(
            "Cosine: C(180°) − C(90°)" if metric == "cosine" else "Distance: d(90°) − d(180°)"
        )
    fig.legend(handles=handles, ncol=4, loc="upper center", frameon=False)
    yield "orientation_contrasts", fig, True
    # Explicit alternatives; retain the accepted main figures during exploration.
    cosine_mouse = pd.read_csv(COS / "mouse_profiles.csv")
    cosine_groups = pd.read_csv(COS / "group_profiles.csv")
    for metric in ["cosine", "euclidean"]:
        for mode in ["endpoints", "contrast"]:
            fig = plt.figure(figsize=(150 / 72, 100 / 72))
            fig.text(1 / 150, 0.81, "B", fontsize=7)
            title = (
                "NR–DR at 180° and 90°"
                if mode == "endpoints"
                else "NR–DR: " + ("C180 − C90" if metric == "cosine" else "d90 − d180")
            ) + " (base / chronic)"
            fig.text(78 / 150, 0.87, title, ha="center", fontsize=4)
            compact_handles = [
                Line2D([], [], color=c, marker=m, lw=0.6, ms=2.7, label=l)
                for c, m, l in zip(
                    COLORS, MARKERS, ["DR base", "NR base", "DR chronic", "NR chronic"]
                )
            ]
            fig.legend(
                handles=compact_handles,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.51, 0.99),
                frameon=False,
                fontsize=4,
                handlelength=1.1,
                columnspacing=0.7,
                handletextpad=0.3,
                borderaxespad=0,
            )
            for j, area in enumerate(AREAS):
                ax = fig.add_axes([(18 + j * 33) / 150, 0.22, 27 / 150, 0.48])
                if metric == "cosine":
                    draw_panel(ax, area, cosine_mouse, cosine_groups, individual=False, dodge=0.15)
                    for line in ax.lines:
                        line.set_linewidth(line.get_linewidth() * 0.45)
                        line.set_markersize(line.get_markersize() * 0.46)
                        line.set_markeredgewidth(0.4)
                    for col in ax.collections:
                        col.set_linewidth(0.5)
                    ax.set_yticks([-1, 0, 1])
                else:
                    for fi, color in enumerate(COLORS):
                        g = (
                            groups[(groups.area == area) & (groups.fi == fi)]
                            .set_index("difference_degrees")
                            .loc[[180, 90]]
                        )
                        y = g["mean"].to_numpy()
                        ax.errorbar(
                            np.array([0, 1]) + OFFSETS[fi] * 0.15,
                            y,
                            yerr=np.maximum(0, [y - g.ci_low, g.ci_high - y]),
                            color=color,
                            marker=MARKERS[fi],
                            ms=2.85,
                            capsize=0.92,
                            lw=0.585,
                            elinewidth=0.5,
                            mew=0.4,
                        )
                    ax.set(xlim=(-0.3, 1.3), ylim=(0, groups.ci_high.max() * 1.08), xticks=[0, 1])
                ax.set_xticklabels(["180°", "90°"], fontsize=4)
                ax.set_title(area, fontsize=6, pad=10)
                for sp in ax.spines.values():
                    sp.set_linewidth(0.45)
                ax.tick_params(labelsize=4, length=1.6, width=0.45, pad=1, labelleft=j == 0)
                chosen = results[(results.metric == metric) & (results.area == area)]
                if mode == "contrast":
                    r = chosen[chosen.endpoint == "contrast"].set_index("timepoint")
                    ax.text(
                        0.5,
                        1.02,
                        r.loc["baseline", "stars"] + " / " + r.loc["chronic", "stars"],
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=4,
                    )
                else:
                    for x, endpoint in enumerate(["180", "90"]):
                        r = chosen[chosen.endpoint == endpoint].set_index("timepoint")
                        ax.text(
                            x,
                            1.02,
                            r.loc["baseline", "stars"] + " / " + r.loc["chronic", "stars"],
                            transform=ax.get_xaxis_transform(),
                            ha="center",
                            va="bottom",
                            fontsize=4,
                        )
                if j == 0:
                    ax.set_ylabel(
                        "Cosine similarity" if metric == "cosine" else "Euclidean distance",
                        fontsize=5,
                        labelpad=2,
                    )
            fig.text(0.54, 0.1, "Direction difference (∆ / °)", ha="center", fontsize=5)
            yield f"B_{metric}_{mode}", fig, False


def render(panel):
    folder = OUT / "orientation_contrasts"
    tables = [
        pd.read_csv(folder / f"{name}.csv")
        for name in ["angular_mouse", "angular_group", "nr_dr_tests", "endpoint_groups"]
    ]
    for name, fig, tight in figures(*tables):
        if name == panel:
            return fig, tight
        plt.close(fig)
    raise ValueError(f"Unknown orientation panel: {panel}")

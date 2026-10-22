"""Accepted nparLD main effects and exact rank-sum comparisons, at mouse level."""

from itertools import product
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from analysis_config import OUT, ROOT, AREAS
from geometry.statistics import holm

DEST = OUT / "manuscript_statistics"


def stars(p):
    return (
        "****"
        if p < 0.0001
        else "***"
        if p < 0.001
        else "**"
        if p < 0.01
        else "*"
        if p < 0.05
        else "ns"
    )


def inputs():
    b = pd.read_csv(OUT / "direction_cosine_top2/mouse_profiles.csv")
    b = b[b.difference_degrees.isin([90, 180])].rename(
        columns={"difference_degrees": "level", "cosine": "original_response"}
    )
    c = pd.read_csv(OUT / "direction_cosine_top2/supporting/decoding_mice.csv").rename(
        columns={"task": "level", "value": "original_response"}
    )
    f = pd.read_csv(OUT / "contrast_scaling_pca/mouse_peak_radius.csv").rename(
        columns={"step": "level", "value": "original_response"}
    )
    g = pd.read_csv(OUT / "contrast_scaling_pca/mouse_reverse_transfer.csv").rename(
        columns={"step": "level", "value": "original_response"}
    )
    all_mice = b[["mouse", "fi"]].drop_duplicates()
    rearings = {
        name: ("DR" if group.fi.iloc[0] % 2 == 0 else "NR")
        for name, group in all_mice.groupby("mouse")
    }
    tables = {}
    for panel, frame, levels in [
        ("B", b, [90, 180]),
        ("C", c, ["direction", "orientation"]),
        ("F", f, [1, 2]),
        ("G", g, [1, 2, 3]),
    ]:
        frame = frame.copy()
        frame["rearing"] = frame.fi.map({0: "DR", 1: "NR", 2: "DR", 3: "NR"})
        frame["time"] = frame.fi.map({0: "baseline", 1: "baseline", 2: "chronic", 3: "chronic"})
        frame["response"] = frame.original_response
        if panel == "C":
            chance = frame.level.map({"orientation": 0.25, "direction": 0.125})
            frame["response"] = (frame.original_response - chance) / (1 - chance)
        if panel == "F":
            frame["response"] = np.log(frame.original_response)
        keys = ["mouse", "rearing", "area", "time", "level"]
        design = pd.DataFrame(
            [
                (m, rearings[m], a, t, l)
                for m, a, t, l in product(sorted(rearings), AREAS, ["baseline", "chronic"], levels)
            ],
            columns=keys,
        )
        result = design.merge(
            frame[keys + ["response", "original_response"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        result.insert(0, "panel", panel)
        tables[panel] = result
    return tables


def posthocs(tables):
    endpoints = {}
    for panel, level in [("B", 180), ("C", "orientation")]:
        endpoints[panel] = tables[panel].query("level == @level").copy()
    d = pd.read_csv(OUT / "direction_cosine_top2/supporting/radius_mice.csv")
    d["response"] = d.value
    d["rearing"] = d.fi.map({0: "DR", 1: "NR", 2: "DR", 3: "NR"})
    d["time"] = d.fi.map({0: "baseline", 1: "baseline", 2: "chronic", 3: "chronic"})
    d["level"] = 0
    endpoints["D"] = d
    f = tables["F"].pivot(
        index=["mouse", "rearing", "area", "time"], columns="level", values="response"
    )
    f = (f[2] - f[1]).rename("response").reset_index()
    f["level"] = 0
    endpoints["F"] = f
    endpoints["G"] = tables["G"]
    rows = []
    for panel, data in endpoints.items():
        for (area, time, level), group in data.groupby(["area", "time", "level"]):
            nr = group.loc[group.rearing == "NR", "response"].dropna().to_numpy()
            dr = group.loc[group.rearing == "DR", "response"].dropna().to_numpy()
            if len(nr) < 2 or len(dr) < 2:
                p = np.nan
                u = np.nan
            else:
                if len(np.unique(np.r_[nr, dr])) != len(nr) + len(dr):
                    raise ValueError(
                        "Ties require review of the registered exact rank-sum protocol"
                    )
                result = mannwhitneyu(nr, dr, alternative="two-sided", method="exact")
                p = result.pvalue
                u = result.statistic
            rows.append(
                dict(
                    panel=panel,
                    area=area,
                    timepoint=time,
                    step=level if panel == "G" else np.nan,
                    n_NR=len(nr),
                    n_DR=len(dr),
                    U_NR=u,
                    p_wilcoxon=p,
                )
            )
    result = pd.DataFrame(rows)
    for _, index in result.groupby(["panel", "area"]).groups.items():
        result.loc[index, "p_adjusted"] = holm(result.loc[index, "p_wilcoxon"].to_numpy())
        result.loc[index, "family_size"] = len(index)
    result["stars"] = result.p_adjusted.map(stars)
    return result


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    tables = inputs()
    for panel, table in tables.items():
        table.to_csv(DEST / f"input_{panel}.csv", index=False)
    posthocs(tables).to_csv(DEST / "posthoc_results.csv", index=False)
    subprocess.run(
        ["Rscript", str(ROOT / "geometry/manuscript_statistics.R"), str(DEST)], check=True
    )
    raw = pd.read_csv(DEST / "factorial_ats_raw.csv").rename(
        columns={"Statistic": "ATS", "df": "df_num"}
    )
    for panel, index in raw.groupby("panel").groups.items():
        raw.loc[index, "p_holm"] = holm(raw.loc[index, "p_raw"].to_numpy())
    raw["stars"] = raw.p_holm.map(stars)
    raw.to_csv(DEST / "factorial_main_effects.csv", index=False)


if __name__ == "__main__":
    run()

import numpy as np
import pandas as pd
from analysis_config import DATA, OUT, ROOT, AREAS, LABELS, ids_for, manifest
import subprocess
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from geometry.statistics import interval, holm
from geometry.cosine import pooled_centroids, centered_cosines, angular_profile

DEST = OUT / "euclidean_panel_B"
STATS = OUT / "manuscript_statistics"
HERE = ROOT / "geometry"


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


def distance_matrix(vectors):
    # Ordinary full-space norm, evaluated before averaging pairs or draws.
    return np.linalg.norm(vectors[:, :, None, :] - vectors[:, None, :, :], axis=-1)


def analyze():
    DEST.mkdir(exist_ok=True)
    (DEST / "populations").mkdir(exist_ok=True)
    sessions, draws = [], []
    for p in manifest()["populations"]:
        response = DATA / "responses" / f"s{p['sid']:03d}.npz"
        with np.load(response) as z:
            means, _ = pooled_centroids(z["X"], z["stim"])
        vectors = np.moveaxis(means[:, ids_for(p)], 1, 0)
        assert vectors.shape == (1000, 8, 10)
        centered = vectors - vectors.mean(1, keepdims=True)
        matrix = distance_matrix(centered)
        np.testing.assert_allclose(matrix, distance_matrix(vectors), atol=1e-14)
        for d in [0, 999]:
            np.testing.assert_allclose(matrix[d], cdist(vectors[d], vectors[d]), atol=1e-14)
        np.testing.assert_array_equal(np.diagonal(matrix, axis1=1, axis2=2), 0)
        assert np.isfinite(matrix).all() and (matrix >= 0).all()
        cosine, valid = centered_cosines(vectors)
        with np.load(OUT / "direction_cosine_top2/populations" / f"p{p['pid']:03d}.npz") as z:
            np.testing.assert_allclose(angular_profile(cosine), z["angular_cosine"], atol=1e-12)
            np.testing.assert_array_equal(valid, z["valid"])
        # Final B has complete valid draws; enforce identical sample support.
        assert valid.all()
        profile = angular_profile(matrix)[:, [4, 2]]
        common = {k: p[k] for k in ["pid", "sid", "area", "fi", "mouse"]}
        common["condition"] = LABELS[p["fi"]]
        np.savez_compressed(
            DEST / "populations" / f"p{p['pid']:03d}.npz",
            euclidean_distance=profile,
            difference_degrees=[180, 90],
        )
        for k, angle in enumerate([180, 90]):
            sessions.append(dict(**common, difference_degrees=angle, distance=profile[:, k].mean()))
            draws.append(
                pd.DataFrame(
                    dict(
                        **common,
                        difference_degrees=angle,
                        draw_index=np.arange(1000),
                        distance=profile[:, k],
                    )
                )
            )
    session = pd.DataFrame(sessions)
    mouse = session.groupby(
        ["area", "fi", "condition", "mouse", "difference_degrees"], as_index=False
    ).agg(distance=("distance", "mean"), sessions=("pid", "size"))
    groups = []
    for (area, fi, angle), sub in mouse.groupby(["area", "fi", "difference_degrees"]):
        sub = sub.sort_values("mouse")
        low, high = interval(sub.distance.to_numpy(), fi)
        groups.append(
            dict(
                area=area,
                fi=fi,
                condition=LABELS[fi],
                difference_degrees=angle,
                mice=len(sub),
                mean=sub.distance.mean(),
                ci_low=low,
                ci_high=high,
            )
        )
    groups = pd.DataFrame(groups)
    session.to_csv(DEST / "session_values.csv", index=False)
    mouse.to_csv(DEST / "mouse_values.csv", index=False)
    groups.to_csv(DEST / "group_estimates.csv", index=False)
    pd.concat(draws).to_csv(DEST / "draw_values.csv.gz", index=False)
    # Preserve the accepted factorial design, including its explicit missing rows.
    template = pd.read_csv(STATS / "input_B.csv")
    lookup = mouse.assign(
        rearing=mouse.fi.map({0: "DR", 1: "NR", 2: "DR", 3: "NR"}),
        time=mouse.fi.map({0: "baseline", 1: "baseline", 2: "chronic", 3: "chronic"}),
    )
    keys = ["mouse", "rearing", "area", "time", "level"]
    lookup = lookup.rename(columns={"difference_degrees": "level"})
    design = template[keys].merge(
        lookup[keys + ["distance"]], on=keys, how="left", validate="one_to_one"
    )
    np.testing.assert_array_equal(template.response.isna(), design.distance.isna())
    design.rename(columns={"distance": "response"}).to_csv(
        DEST / "input_euclidean.csv", index=False
    )
    template.to_csv(DEST / "input_cosine_validation.csv", index=False)
    subprocess.run(
        ["Rscript", str(HERE / "euclidean_panel_b.R"), str(DEST)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    check = pd.read_csv(DEST / "factorial_cosine_validation.csv").set_index("factor")
    old = (
        pd.read_csv(STATS / "factorial_main_effects.csv")
        .query("panel == 'B'")
        .set_index("factor")
        .loc[check.index]
    )
    np.testing.assert_allclose(check.Statistic, old.ATS, rtol=1e-10)
    np.testing.assert_allclose(check.df, old.df_num, rtol=1e-10)
    np.testing.assert_allclose(check.p_holm, old.p_holm, atol=1e-12)
    tests = []
    for area in AREAS:
        for time, fi in [("baseline", 0), ("chronic", 2)]:
            sub = mouse[(mouse.area == area) & (mouse.difference_degrees == 180)]
            nr = sub[sub.fi == fi + 1].distance.to_numpy()
            dr = sub[sub.fi == fi].distance.to_numpy()
            assert len(np.unique(np.r_[nr, dr])) == len(nr) + len(dr)
            test = mannwhitneyu(nr, dr, alternative="two-sided", method="exact")
            tests.append(
                dict(
                    area=area,
                    timepoint=time,
                    n_NR=len(nr),
                    n_DR=len(dr),
                    mean_NR=nr.mean(),
                    mean_DR=dr.mean(),
                    U_NR=test.statistic,
                    p_raw=test.pvalue,
                )
            )
    tests = pd.DataFrame(tests)
    for area in AREAS:
        ix = tests.area == area
        tests.loc[ix, "p_holm"] = holm(tests.loc[ix, "p_raw"].to_numpy())
    tests["stars"] = tests.p_holm.map(stars)
    tests.to_csv(DEST / "posthoc_results.csv", index=False)
    factorial = pd.read_csv(DEST / "factorial_euclidean.csv")
    factorial["stars"] = factorial.p_holm.map(stars)
    factorial.to_csv(DEST / "factorial_euclidean.csv", index=False)
    return groups, tests, factorial

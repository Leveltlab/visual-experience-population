"""Mouse-level orientation contrasts, endpoint tests and full-angle distances."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cdist
from analysis_config import OUT, DATA, manifest, ids_for
from geometry.cosine import pooled_centroids, angular_profile
from geometry.statistics import holm, interval
from geometry.euclidean import distance_matrix, stars

DEST = OUT / "orientation_contrasts"
EUC = OUT / "euclidean_panel_B"
COS = OUT / "direction_cosine_top2"


def angular():
    sessions = []
    draws = []
    for p in manifest()["populations"]:
        response = DATA / "responses" / f"s{p['sid']:03d}.npz"
        with np.load(response) as z:
            means, _ = pooled_centroids(z["X"], z["stim"])
        v = np.moveaxis(means[:, ids_for(p)], 1, 0)
        assert v.shape == (1000, 8, 10)
        v = v - v.mean(1, keepdims=True)
        matrix = distance_matrix(v)
        profile = angular_profile(matrix)
        for d in [0, 999]:
            np.testing.assert_allclose(matrix[d], cdist(v[d], v[d]), atol=1e-14)
        np.testing.assert_allclose(profile[:, [1, 2, 3]], profile[:, [7, 6, 5]], atol=1e-14)
        np.testing.assert_array_equal(profile[:, 0], 0)
        assert np.isfinite(profile).all()
        with np.load(EUC / "populations" / f"p{p['pid']:03d}.npz") as z:
            np.testing.assert_allclose(
                profile[:, [4, 2]], z["euclidean_distance"], atol=1e-14, rtol=0
            )
        for i, a in enumerate(np.arange(8) * 45):
            common = {k: p[k] for k in ["pid", "sid", "area", "fi", "mouse"]}
            sessions.append(dict(**common, difference_degrees=a, value=profile[:, i].mean()))
            draws.append(
                pd.DataFrame(
                    dict(
                        **common,
                        difference_degrees=a,
                        draw_index=np.arange(1000),
                        value=profile[:, i],
                    )
                )
            )
    session = pd.DataFrame(sessions)
    mouse = session.groupby(
        ["area", "fi", "mouse", "difference_degrees"], as_index=False
    ).value.mean()
    rows = []
    for (area, fi, angle), sub in mouse.groupby(["area", "fi", "difference_degrees"]):
        sub = sub.sort_values("mouse")
        lo, hi = interval(sub.value.to_numpy(), fi)
        rows.append(
            dict(
                area=area,
                fi=fi,
                difference_degrees=angle,
                mice=len(sub),
                mean=sub.value.mean(),
                ci_low=lo,
                ci_high=hi,
            )
        )
    groups = pd.DataFrame(rows)
    for name, table in [
        ("angular_session", session),
        ("angular_mouse", mouse),
        ("angular_group", groups),
    ]:
        table.to_csv(DEST / f"{name}.csv", index=False)
    pd.concat(draws).to_csv(DEST / "angular_draws.csv.gz", index=False)
    return mouse, groups


def endpoint_tests(cosine, distance):
    """Exact NR–DR tests of mouse endpoints and paired angular contrasts."""
    cosine = cosine.rename(columns={"cosine": "value"})
    endpoints = []
    for metric, frame in [("cosine", cosine), ("euclidean", distance)]:
        wide = frame.pivot(
            index=["area", "fi", "mouse"], columns="difference_degrees", values="value"
        )
        contrast = wide[180] - wide[90] if metric == "cosine" else wide[90] - wide[180]
        for endpoint, values in [("180", wide[180]), ("90", wide[90]), ("contrast", contrast)]:
            table = values.rename("value").reset_index()
            table["metric"] = metric
            table["endpoint"] = endpoint
            endpoints.append(table)
    endpoints = pd.concat(endpoints, ignore_index=True)
    results = []
    summaries = []
    for (metric, endpoint, area), group in endpoints.groupby(["metric", "endpoint", "area"]):
        for time, fi in [("baseline", 0), ("chronic", 2)]:
            nr = group[group.fi == fi + 1].sort_values("mouse")
            dr = group[group.fi == fi].sort_values("mouse")
            assert len(np.unique(np.r_[nr.value, dr.value])) == len(nr) + len(dr), (
                "Exact untied rank-sum required"
            )
            test = mannwhitneyu(nr.value, dr.value, method="exact", alternative="two-sided")
            results.append(
                dict(
                    metric=metric,
                    endpoint=endpoint,
                    area=area,
                    timepoint=time,
                    n_NR=len(nr),
                    n_DR=len(dr),
                    mean_NR=nr.value.mean(),
                    mean_DR=dr.value.mean(),
                    NR_minus_DR=nr.value.mean() - dr.value.mean(),
                    U_NR=test.statistic,
                    p_raw=test.pvalue,
                    probability_NR_exceeds_DR=test.statistic / (len(nr) * len(dr)),
                )
            )
        for fi, sub in group.groupby("fi"):
            sub = sub.sort_values("mouse")
            lo, hi = interval(sub.value.to_numpy(), fi)
            summaries.append(
                dict(
                    metric=metric,
                    endpoint=endpoint,
                    area=area,
                    fi=fi,
                    mice=len(sub),
                    mean=sub.value.mean(),
                    ci_low=lo,
                    ci_high=hi,
                )
            )
    result = pd.DataFrame(results)
    for _, ix in result.groupby(["metric", "endpoint", "area"]).groups.items():
        result.loc[ix, "p_holm"] = holm(result.loc[ix, "p_raw"])
    for _, ix in result.groupby(["metric", "area"]).groups.items():
        result.loc[ix, "p_holm6_sensitivity"] = holm(result.loc[ix, "p_raw"])
    result["stars"] = result.p_holm.map(stars)
    summaries = pd.DataFrame(summaries)
    return endpoints, result, summaries


def analyze():
    DEST.mkdir(parents=True, exist_ok=True)
    mouse, groups = angular()
    endpoints, results, summaries = endpoint_tests(pd.read_csv(COS / "mouse_profiles.csv"), mouse)
    for name, table in [
        ("mouse_endpoints", endpoints),
        ("nr_dr_tests", results),
        ("endpoint_groups", summaries),
    ]:
        table.to_csv(DEST / f"{name}.csv", index=False)
    return results, summaries

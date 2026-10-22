"""Paired angular contrasts and published endpoint inference."""

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from geometry.orientation_contrasts import endpoint_tests
from geometry.statistics import interval

DATA = Path(__file__).resolve().parents[1] / "plotting/source_data"


def test_saved_endpoint_tests_and_bootstrap_intervals():
    cosine = pd.read_csv(DATA / "direction_cosine_top2/mouse_profiles.csv")
    distance = pd.read_csv(DATA / "orientation_contrasts/angular_mouse.csv")
    endpoints, tests, groups = endpoint_tests(cosine, distance)
    for actual, filename in [(tests, "nr_dr_tests"), (groups, "endpoint_groups")]:
        expected = pd.read_csv(DATA / f"orientation_contrasts/{filename}.csv")
        assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)
    assert len(tests) == 48
    assert (tests.p_holm6_sensitivity >= tests.p_holm - 1e-15).all()
    # Each endpoint retains its own timepoint's mice, with subtraction before bootstrap.
    for (metric, area, fi), rows in endpoints.groupby(["metric", "area", "fi"]):
        wide = rows.pivot(index="mouse", columns="endpoint", values="value").sort_index()
        difference = wide["180"] - wide["90"] if metric == "cosine" else wide["90"] - wide["180"]
        np.testing.assert_allclose(wide["contrast"], difference, atol=1e-15)
        estimate = groups[
            (groups.metric == metric)
            & (groups.area == area)
            & (groups.fi == fi)
            & (groups.endpoint == "contrast")
        ].iloc[0]
        np.testing.assert_allclose(
            [estimate.ci_low, estimate.ci_high], interval(difference.to_numpy(), fi), atol=1e-14
        )
    # The new 180-degree tests preserve existing main-panel endpoint corrections.
    for metric, file, column in [
        ("cosine", "manuscript_statistics/posthoc_results.csv", "p_adjusted"),
        ("euclidean", "euclidean_panel_B/posthoc_results.csv", "p_holm"),
    ]:
        previous = pd.read_csv(DATA / file)
        if metric == "cosine":
            previous = previous[previous.panel == "B"]
        current = tests[(tests.metric == metric) & (tests.endpoint == "180")].set_index(
            ["area", "timepoint"]
        )
        previous = previous.set_index(["area", "timepoint"]).loc[current.index]
        np.testing.assert_allclose(current.p_holm, previous[column], atol=1e-12)


def test_angular_distance_symmetry_and_existing_endpoints():
    data = pd.read_csv(DATA / "orientation_contrasts/angular_mouse.csv")
    wide = data.pivot(index=["area", "fi", "mouse"], columns="difference_degrees", values="value")
    np.testing.assert_array_equal(wide[0], 0)
    for angle in [45, 90, 135]:
        np.testing.assert_allclose(wide[angle], wide[360 - angle], atol=1e-14)
    actual = pd.read_csv(DATA / "orientation_contrasts/angular_group.csv")
    expected = pd.read_csv(DATA / "euclidean_panel_B/group_estimates.csv")
    keys = ["area", "fi", "difference_degrees"]
    columns = ["mean", "ci_low", "ci_high"]
    actual = actual[actual.difference_degrees.isin([90, 180])].set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    assert_frame_equal(
        actual[columns], expected[columns], check_exact=False, atol=1e-14, rtol=1e-12
    )

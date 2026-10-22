"""Independent peak-order, full-N10 geometry, inference and export checks."""

import numpy as np
import pandas as pd
from geometry.peak import radius_curve, pooled_centers, peak_mouse_curves


def test_radius_matches_explicit_full_n10_and_translation():
    rng = np.random.default_rng(411)
    c = rng.normal(size=(3, 8, 4, 17))
    ids = np.array([rng.choice(17, 10, replace=False) for _ in range(11)])
    direct = []
    for index in ids:
        v = c[..., index]
        v -= v.mean(-2, keepdims=True)
        direct.append(np.mean(np.sum(v * v, -1), -1))
    np.testing.assert_allclose(radius_curve(c, ids), np.mean(direct, 0), atol=1e-13)
    np.testing.assert_allclose(radius_curve(c + 10, ids), radius_curve(c, ids), atol=1e-13)


def test_session_averaging_precedes_mouse_peak():
    rows = []
    for sid, values in enumerate([[10.0, 0.0], [0.0, 10.0]]):
        for frame, value in enumerate(values):
            rows.append(
                dict(
                    area="V1",
                    fi=0,
                    mouse="a",
                    sid=sid,
                    pool="single",
                    step=1,
                    frame=frame,
                    time=frame * 0.1,
                    value=value,
                )
            )
    curves, peaks = peak_mouse_curves(pd.DataFrame(rows))
    assert peaks.value.item() == 5.0
    assert peaks.peak_time.item() == 0.0  # Deterministic earliest frame on a tie.


def test_pooling_centroids_precedes_norm():
    raw = np.zeros((2, 8, 10))
    ci = np.repeat([0, 1], 4)
    ori = np.tile(np.arange(4), 2)
    raw[:, 0] = 1
    raw[:, 4] = -1
    separate = pooled_centers(raw, ci, ori, 2)
    pooled = pooled_centers(raw, ci // 2, ori, 1)
    ids = np.arange(10)[None]
    assert radius_curve(separate, ids).mean() > 0
    assert np.all(radius_curve(pooled, ids) == 0)

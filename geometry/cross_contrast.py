"""B supplement: same-direction cosine across contrast pools with split-half benchmark."""

import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine
from analysis_config import DATA, OUT, ROOT, LABELS, ids_for, manifest, sha
from geometry.statistics import finite_mean, interval, unit_vectors
from geometry.cosine import ANGLES, centered_cosines

DEST = OUT / "direction_cosine_top2/cross_contrast"
PAIR_LABELS = ["1 + 1.93", "3.72 + 7.19", "13.89 + 26.82", "51.79 + 100 (held out)"]
METRICS = ["cosine"]


def pool_vectors(x, stim, ids):
    """Pool 30 raw trials per direction; shape draw × pool × direction × neuron."""
    contrasts = np.unique(stim[:, 1])
    np.testing.assert_array_equal(np.unique(stim[:, 0]), ANGLES)
    assert len(contrasts) == 8 and ids.shape == (1000, 10)
    pools = []
    for k in range(4):
        means = []
        for direction in ANGLES:
            masks = [
                (stim[:, 0] == direction) & (stim[:, 1] == c) for c in contrasts[2 * k : 2 * k + 2]
            ]
            assert [int(m.sum()) for m in masks] == [15, 15]
            means.append(x[masks[0] | masks[1]].mean(0))
        pools.append(means)
    vectors = np.moveaxis(np.asarray(pools)[..., ids], 2, 0)
    assert vectors.shape == (1000, 4, 8, 10)
    return vectors


def split_high_vectors(x, stim, ids, pid):
    """Fresh disjoint 15/15 split per direction and draw, with recorded trial rows."""
    high = np.unique(stim[:, 1])[-2:]
    rows = np.array([np.flatnonzero((stim[:, 0] == d) & np.isin(stim[:, 1], high)) for d in ANGLES])
    assert rows.shape == (8, 30)
    rng = np.random.default_rng([20261020, 50, int(pid)])
    order = np.argsort(rng.random((len(ids), 8, 30)), axis=-1)
    shuffled = np.take_along_axis(np.broadcast_to(rows, order.shape), order, axis=-1)
    reference_rows, heldout_rows = shuffled[:, :, :15], shuffled[:, :, 15:]
    reference = x[reference_rows[..., None], ids[:, None, None, :]].mean(2)
    heldout = x[heldout_rows[..., None], ids[:, None, None, :]].mean(2)
    return reference, heldout, reference_rows.astype(np.int16), heldout_rows.astype(np.int16)


def compare_vectors(targets, reference):
    """Center each pool/half separately, then evaluate full ten-neuron cosines."""
    centered = targets - targets.mean(axis=-2, keepdims=True)
    reference = reference - reference.mean(axis=-2, keepdims=True)
    return np.sum(unit_vectors(centered) * unit_vectors(reference[:, None]), axis=-1)


def analyze():
    (DEST / "populations").mkdir(parents=True, exist_ok=True)
    populations = manifest()["populations"]
    rows, direction_rows, coverage, hashes = [], [], [], {}
    reference_error = 0.0
    for p in populations:
        path = DATA / "responses" / f"s{p['sid']:03d}.npz"
        with np.load(path) as z:
            ids = ids_for(p)
            vectors = pool_vectors(z["X"], z["stim"], ids)
            reference, heldout, reference_rows, heldout_rows = split_high_vectors(
                z["X"], z["stim"], ids, p["pid"]
            )
        np.testing.assert_allclose((reference + heldout) / 2, vectors[:, 3], atol=1e-12)
        assert np.all(
            np.diff(np.sort(np.concatenate([reference_rows, heldout_rows], -1), -1), axis=-1) > 0
        )
        targets = np.concatenate([vectors[:, :3], heldout[:, None]], axis=1)
        cosine = compare_vectors(targets, reference)
        assert np.nanmax(abs(cosine)) <= 1 + 1e-12
        ref, _ = centered_cosines(vectors[:, 3])
        with np.load(OUT / "direction_cosine_top2/populations" / f"p{p['pid']:03d}.npz") as z:
            error = np.nanmax(abs(ref - z["cosine_matrix"]))
            np.testing.assert_allclose(ref, z["cosine_matrix"], atol=1e-12, equal_nan=True)
        reference_error = max(reference_error, float(error))
        centered = targets - targets.mean(-2, keepdims=True)
        ref_centered = reference - reference.mean(-2, keepdims=True)
        for draw, pair, direction in [(0, 0, 0), (999, 2, 7), (500, 3, 4)]:
            a, b = centered[draw, pair, direction], ref_centered[draw, direction]
            if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                np.testing.assert_allclose(
                    cosine[draw, pair, direction], 1 - scipy_cosine(a, b), atol=1e-13
                )
        common = dict(pid=p["pid"], sid=p["sid"], mouse=p["mouse"], area=p["area"], fi=p["fi"])
        for metric, values in zip(METRICS, [cosine]):
            # All eight directions required for a draw's primary endpoint.
            summary = values.mean(-1)
            for pair in range(4):
                rows.append(
                    dict(
                        **common,
                        metric=metric,
                        pair=pair,
                        value=float(finite_mean(summary[:, pair], 0)),
                    )
                )
                coverage.append(
                    dict(
                        **common,
                        metric=metric,
                        pair=pair,
                        valid_draws=int(np.isfinite(summary[:, pair]).sum()),
                        requested_draws=1000,
                    )
                )
                for direction in range(8):
                    direction_rows.append(
                        dict(
                            **common,
                            metric=metric,
                            pair=pair,
                            direction=direction * 45,
                            value=float(finite_mean(values[:, pair, direction], 0)),
                        )
                    )
        np.savez_compressed(
            DEST / "populations" / f"p{p['pid']:03d}.npz",
            cosine=cosine,
            reference_trial_rows=reference_rows,
            heldout_trial_rows=heldout_rows,
        )
        hashes[str(p["pid"])] = dict(
            response=sha(path), selection=sha(DATA / "selections" / f"p{p['pid']:03d}.npy")
        )
    sessions = pd.DataFrame(rows)
    mouse = sessions.groupby(["area", "fi", "mouse", "metric", "pair"], as_index=False).value.agg(
        lambda x: np.mean(x.to_numpy())
    )
    assert np.isfinite(mouse.value).all()
    groups = []
    for keys, sub in mouse.groupby(["area", "fi", "metric", "pair"]):
        area, fi, metric, pair = keys
        x = sub.sort_values("mouse").value.to_numpy()
        low, high = interval(x, fi)
        groups.append(
            dict(
                area=area,
                fi=fi,
                condition=LABELS[fi],
                metric=metric,
                pair=pair,
                contrast_pair=PAIR_LABELS[pair],
                mice=len(x),
                mean=x.mean(),
                ci_low=low,
                ci_high=high,
            )
        )
    groups = pd.DataFrame(groups)
    for name, frame in [
        ("session_metrics.csv", sessions),
        ("mouse_metrics.csv", mouse),
        ("group_metrics.csv", groups),
        ("coverage.csv", pd.DataFrame(coverage)),
        ("individual_directions_session.csv", pd.DataFrame(direction_rows)),
    ]:
        frame.to_csv(DEST / name, index=False)
    validation = dict(
        validated=True,
        populations=len(populations),
        unique_mice=int(mouse.mouse.nunique()),
        n=10,
        draws=1000,
        lower_trials_per_direction=30,
        reference_trials_per_direction=15,
        high_heldout_trials_per_direction=15,
        split_seed=[20261020, 50, "population_id"],
        split="Uniform random 15/15 partition of the 30 high-pool trials per direction and draw; not stratified by contrast. Same reference half for all four endpoints.",
        lower_pairs_percent=(np.array(manifest()["contrasts"])[:6].reshape(3, 2) * 100).tolist(),
        reference_pair_percent=(np.array(manifest()["contrasts"])[6:] * 100).tolist(),
        response_window_seconds=[0, 0.5],
        centering="Subtract the mean of eight direction centroids separately in each contrast pool or high-contrast half.",
        cosine="Full N10 cosine of the same direction in each target and the shared 15-trial reference, within each draw.",
        aggregation="Equal eight directions per draw, valid draws per session, equal sessions within mouse, equal mice.",
        intervals="Pointwise 95% whole-mouse bootstrap intervals, 5000 resamples. Descriptive; no new significance tests.",
        missing="Cosine with a zero-length vector is undefined. All eight directions must be finite for a draw.",
        min_cosine_valid_draws=min(r["valid_draws"] for r in coverage if r["metric"] == "cosine"),
        max_reference_cosine_matrix_error=reference_error,
        numerical_checks=[
            "15+15 trials per direction per pool",
            "no shared trials between lower and reference pools",
            "reference matrix matches flagship",
            "disjoint 15/15 halves reconstruct full centroid",
            "SciPy cosine references in every population",
            "bounded cosine",
        ],
        source_hashes=hashes,
        code_sha256=sha(__file__),
        helpers_sha256={
            name: sha(ROOT / name)
            for name in ["geometry/statistics.py", "geometry/cosine.py", "analysis_config.py"]
        },
    )
    (DEST / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return groups, validation

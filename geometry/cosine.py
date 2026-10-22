import json
import numpy as np
import pandas as pd
from analysis_config import DATA, OUT, ROOT, AREAS, LABELS, ids_for, manifest, sha
from scipy import stats
from geometry.statistics import finite_mean, holm, interval, signflip_p, unit_vectors

DEST = OUT / "direction_cosine_top2"
ANGLES = np.arange(8) * 45


def centered_cosines(centroids):
    """draw × 8 directions × 10 neurons -> draw × direction × direction."""
    centered = centroids - centroids.mean(axis=-2, keepdims=True)
    unit = unit_vectors(centered)
    matrix = unit @ unit.swapaxes(-1, -2)
    valid = np.isfinite(matrix).all(axis=(-1, -2))
    matrix[~valid] = np.nan  # Common draw set for all angular separations.
    return matrix, valid


def angular_profile(matrix):
    """All eight ordered pairs at each directed separation modulo 360 degrees."""
    i = np.arange(8)
    return np.stack([matrix[..., i, (i + k) % 8].mean(axis=-1) for k in range(8)], -1)


def pooled_centroids(x, stim):
    """Pool raw trial responses before centering and cosine normalization."""
    contrasts = np.unique(stim[:, 1])
    np.testing.assert_array_equal(np.unique(stim[:, 0]), ANGLES)
    per_contrast = []
    pooled = []
    for direction in ANGLES:
        masks = [(stim[:, 0] == direction) & (stim[:, 1] == c) for c in contrasts[-2:]]
        assert [int(m.sum()) for m in masks] == [15, 15]
        combined = masks[0] | masks[1]
        assert combined.sum() == 30
        pooled.append(x[combined].mean(0))
        per_contrast.append([x[m].mean(0) for m in masks])
    pooled = np.asarray(pooled)
    per_contrast = np.moveaxis(np.asarray(per_contrast), 1, 0)
    np.testing.assert_allclose(pooled, per_contrast.mean(0), rtol=1e-12, atol=1e-14)
    return pooled, per_contrast


def analyze():
    (DEST / "populations").mkdir(parents=True, exist_ok=True)
    populations = manifest()["populations"]
    data, pair_rows, coverage, comparisons, hashes = {}, [], [], [], {}
    for p in populations:
        response_path = DATA / "responses" / f"s{p['sid']:03d}.npz"
        with np.load(response_path) as z:
            means, separate = pooled_centroids(z["X"], z["stim"])
        ids = ids_for(p)
        assert ids.shape == (1000, 10) and np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
        vectors = np.moveaxis(means[:, ids], 1, 0)
        assert vectors.shape == (1000, 8, 10)
        matrix, valid = centered_cosines(vectors)
        profile = angular_profile(matrix)
        assert valid.any()
        np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2), atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(profile[:, 0][valid], 1, atol=1e-14)
        for k in [1, 2, 3]:
            np.testing.assert_allclose(profile[:, k], profile[:, 8 - k], atol=1e-14, equal_nan=True)
        assert np.nanmax(abs(matrix)) <= 1 + 1e-12
        # Independent SciPy reference for the first and last complete draw.
        from scipy.spatial.distance import cosine

        centered = vectors - vectors.mean(1, keepdims=True)
        for d in np.flatnonzero(valid)[[0, -1]]:
            for i, j in [(0, 1), (0, 2), (0, 4), (7, 3)]:
                np.testing.assert_allclose(
                    matrix[d, i, j], 1 - cosine(centered[d, i], centered[d, j]), atol=1e-13
                )
        # Preserve the distinction between pooling centroids and averaging cosines.
        individual = np.array(
            [angular_profile(centered_cosines(np.moveaxis(a[:, ids], 1, 0))[0]) for a in separate]
        ).mean(0)
        common = dict(
            pid=p["pid"],
            sid=p["sid"],
            area=p["area"],
            fi=p["fi"],
            condition=LABELS[p["fi"]],
            mouse=p["mouse"],
        )
        data[p["pid"]] = dict(profile=finite_mean(profile, 0), matrix=finite_mean(matrix, 0))
        for k, angle in enumerate(ANGLES):
            comparisons.append(
                dict(
                    **common,
                    difference_degrees=int(angle),
                    pooled_30_trial_cosine=data[p["pid"]]["profile"][k],
                    average_of_two_15_trial_cosines=finite_mean(individual, 0)[k],
                )
            )
        for i in range(8):
            for j in range(8):
                pair_rows.append(
                    dict(
                        **common,
                        direction_i=int(ANGLES[i]),
                        direction_j=int(ANGLES[j]),
                        difference_degrees=int((ANGLES[j] - ANGLES[i]) % 360),
                        cosine=data[p["pid"]]["matrix"][i, j],
                    )
                )
        coverage.append(
            dict(
                **common,
                requested_draws=1000,
                valid_draws=int(valid.sum()),
                trials_per_direction=30,
            )
        )
        hashes[str(p["pid"])] = dict(
            responses=sha(response_path),
            selections=sha(DATA / "selections" / f"p{p['pid']:03d}.npy"),
        )
        np.savez_compressed(
            DEST / "populations" / f"p{p['pid']:03d}.npz",
            cosine_matrix=matrix,
            angular_cosine=profile,
            valid=valid,
            selection_sha256=hashes[str(p["pid"])]["selections"],
        )
    mouse_rows, mouse_pairs = [], []
    for area in AREAS:
        for fi in range(4):
            pops = [p for p in populations if p["area"] == area and p["fi"] == fi]
            for mouse in sorted({p["mouse"] for p in pops}):
                chosen = [p for p in pops if p["mouse"] == mouse]
                common = dict(
                    area=area, fi=fi, condition=LABELS[fi], mouse=mouse, sessions=len(chosen)
                )
                profile = np.mean([data[p["pid"]]["profile"] for p in chosen], 0)
                matrix = np.mean([data[p["pid"]]["matrix"] for p in chosen], 0)
                for k, angle in enumerate(ANGLES):
                    mouse_rows.append(
                        dict(**common, difference_degrees=int(angle), cosine=profile[k])
                    )
                for i in range(8):
                    for j in range(8):
                        mouse_pairs.append(
                            dict(
                                **common,
                                direction_i=int(ANGLES[i]),
                                direction_j=int(ANGLES[j]),
                                difference_degrees=int((ANGLES[j] - ANGLES[i]) % 360),
                                cosine=matrix[i, j],
                            )
                        )
    mouse = pd.DataFrame(mouse_rows)
    groups, tests = [], []
    for (area, fi), sub in mouse.groupby(["area", "fi"]):
        values = sub.pivot(index="mouse", columns="difference_degrees", values="cosine")
        for angle in ANGLES:
            low, high = interval(values[angle].to_numpy(), fi)
            groups.append(
                dict(
                    area=area,
                    fi=fi,
                    condition=LABELS[fi],
                    difference_degrees=angle,
                    mice=len(values),
                    mean=values[angle].mean(),
                    ci_low=low,
                    ci_high=high,
                )
            )
        delta = (values[180] - values[90]).to_numpy()
        low, high = interval(delta, fi)
        result = stats.ttest_1samp(delta, 0)
        tests.append(
            dict(
                area=area,
                fi=fi,
                condition=LABELS[fi],
                mice=len(values),
                opposite_minus_orthogonal=delta.mean(),
                ci_low=low,
                ci_high=high,
                positive_mice=int((delta > 0).sum()),
                p=result.pvalue,
                p_signflip=signflip_p(delta),
            )
        )
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p.to_numpy())
    tests["p_signflip_holm"] = holm(tests.p_signflip.to_numpy())
    groups = pd.DataFrame(groups)
    for name, frame in [
        ("mouse_profiles.csv", mouse),
        ("group_profiles.csv", groups),
        ("individual_pairs_session.csv", pd.DataFrame(pair_rows)),
        ("individual_pairs_mouse.csv", pd.DataFrame(mouse_pairs)),
        ("coverage.csv", pd.DataFrame(coverage)),
        ("paired_tests.csv", tests),
        ("pooling_comparison.csv", pd.DataFrame(comparisons)),
    ]:
        frame.to_csv(DEST / name, index=False)
    validation = dict(
        validated=True,
        populations=len(populations),
        unique_mice=int(mouse.mouse.nunique()),
        mouse_area_condition_profiles=len(mouse) // 8,
        n=10,
        draws=1000,
        contrasts_percent=(np.array(manifest()["contrasts"])[-2:] * 100).tolist(),
        trials_per_contrast_direction=15,
        pooled_trials_per_direction=30,
        total_valid_draws=sum(r["valid_draws"] for r in coverage),
        total_requested_draws=len(populations) * 1000,
        directions_degrees=ANGLES.tolist(),
        direction_difference_degrees=ANGLES.tolist(),
        centering="Subtract the mean of eight pooled direction centroids per neuron.",
        normalization="Cosine of the complete 10-neuron vectors, separately in each draw.",
        pooling="Combine raw trials from the two highest contrasts before constructing centroids; never average already normalized contrast-specific cosines.",
        pairs="Eight ordered pairs at each directed angular difference. The 180-degree bin contains four unique undirected pairs, each twice. The 0-degree bin contains eight self-pairs.",
        aggregation="Equal pairs within draw, equal valid draws within session-area population, equal sessions within mouse, equal mice within area-condition.",
        symmetry="C(delta)=C(360-delta) is guaranteed by symmetric cosine and complete pair averaging; it is not evidence of correct neural encoding. A second peak at 180 degrees and minima near 90/270 degrees are orientation-like.",
        missing="A zero-length centered vector invalidates its whole draw for every bin; no epsilon or zero replacement.",
        intervals="Pointwise 95% intervals from 5000 whole-mouse bootstrap resamples.",
        inference="Exploratory paired mouse t tests, opposite minus orthogonal, Holm across 16 area-condition groups. Exact two-sided sign flips with the same family are a sensitivity check; no tests on mirrored bins or the identity diagonal.",
        limitation="All-trial centered cosine can have noise and shared-centering bias. Self-similarity equals one by construction. Previous independent-fold study remains separately available.",
        numerical_checks=[
            "30 raw trials per direction",
            "direct pooling equals mean of equal-sized centroids",
            "SciPy cosine spot checks in every population",
            "full matrix symmetry",
            "self-pair cosine one",
            "full-circle reflection symmetry",
            "bounds [-1,1]",
        ],
        source_hashes=hashes,
        code_sha256=sha(__file__),
        helper_sha256=sha(ROOT / "geometry/statistics.py"),
        protocol_sha256=sha(ROOT / "protocol.json"),
    )
    (DEST / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return mouse, groups, tests, validation

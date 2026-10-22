import json
import numpy as np
import pandas as pd
from analysis_config import DATA, OUT, LABELS, ids_for, manifest, sha
from geometry.cosine import pooled_centroids
from geometry.statistics import interval
from decoding.pooled import source

DEST = OUT / "direction_cosine_top2/supporting"


def geometry(x, ids):
    """Translation-centered sample cloud: squared radius and top-three fraction per draw."""
    x = x - x.mean(0)
    cov = x.T @ x / len(x)
    selected = cov[ids[:, :, None], ids[:, None, :]]
    eigen = np.linalg.eigvalsh(selected)
    radius = np.trace(selected, axis1=-2, axis2=-1)
    retained = np.divide(
        eigen[:, -3:].sum(-1), radius, out=np.full_like(radius, np.nan), where=radius > 0
    )
    return radius, retained


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []
    inputs = {}
    for p in manifest()["populations"]:
        response = DATA / "responses" / f"s{p['sid']:03d}.npz"
        with np.load(response) as z:
            means, separate = pooled_centroids(z["X"], z["stim"])
        ids = ids_for(p)
        x, di, folds = source(p)
        radius, retained = geometry(means, ids)
        # Independent direct N-dimensional norm calculation.
        vectors = np.moveaxis(means[:, ids], 1, 0)
        vectors -= vectors.mean(1, keepdims=True)
        np.testing.assert_allclose(
            radius, np.mean(np.sum(vectors**2, -1), -1), rtol=1e-12, atol=1e-14
        )
        common = {k: p[k] for k in ["pid", "sid", "area", "fi", "mouse"]}
        rows.append(dict(**common, value=float(radius.mean())))
        levels = {
            "pooled_30_trial_centroids": retained,
            "pooled_individual_trials": geometry(x, ids)[1],
            "single_15_trial_centroids": np.mean([geometry(s, ids)[1] for s in separate], axis=0),
            "heldout_6_trial_centroids": np.mean(
                [
                    geometry(
                        np.array([x[(di == d) & (folds == f)].mean(0) for d in range(8)]), ids
                    )[1]
                    for f in range(5)
                ],
                axis=0,
            ),
        }
        for level, values in levels.items():
            audit.append(dict(**common, level=level, value=float(np.mean(values))))
        inputs[str(p["pid"])] = dict(
            response=sha(response), selection=sha(DATA / "selections" / f"p{p['pid']:03d}.npy")
        )
    sessions = pd.DataFrame(rows)
    mice = sessions.groupby(["area", "fi", "mouse"], as_index=False).value.mean()
    groups = []
    for (area, fi), g in mice.groupby(["area", "fi"]):
        values = g.sort_values("mouse").value.to_numpy()
        lo, hi = interval(values, fi)
        groups.append(
            dict(
                area=area,
                fi=fi,
                condition=LABELS[fi],
                mean=values.mean(),
                ci_low=lo,
                ci_high=hi,
                mice=len(values),
            )
        )
    sessions.to_csv(DEST / "radius_sessions.csv", index=False)
    mice.to_csv(DEST / "radius_mice.csv", index=False)
    pd.DataFrame(groups).to_csv(DEST / "radius_groups.csv", index=False)
    audit = pd.DataFrame(audit)
    audit.to_csv(DEST / "pca_averaging_audit_sessions.csv", index=False)
    am = audit.groupby(["area", "fi", "mouse", "level"], as_index=False).value.mean()
    ag = am.groupby(["area", "fi", "level"], as_index=False).value.mean()
    ag.to_csv(DEST / "pca_averaging_audit_groups.csv", index=False)
    (DEST / "radius_validation.json").write_text(
        json.dumps(
            dict(
                validated=True,
                populations=len(sessions),
                draws=1000,
                n=10,
                trials_per_direction=30,
                directions=8,
                definition="Mean over eight directions of squared Euclidean norm of each centered 30-trial centroid in full N10 space; compute before averaging draws, sessions and mice. No division by neuron count. Equal-mouse group means and 95% whole-mouse bootstrap intervals.",
                caveat="All-trial centroid radius includes finite-trial noise; it is not a noise-unbiased estimate or the radius of the averaged PCA overlay.",
                pca_audit="Top-three fraction fitted separately to each N10 draw/cloud, then averaged draws, sessions, mice. These are individual-cloud upper bounds, not variance retained by the displayed shared group PCA axes.",
                inputs=inputs,
                source=sha(__file__),
            ),
            indent=2,
        )
        + "\n"
    )
    print(ag.groupby("level").value.agg(["min", "median", "max"]).to_string())

import json
import numpy as np
import pandas as pd
from analysis_config import DATA, OUT, ROOT, AREAS, LABELS, ids_for, manifest, sha, mouse_average
from collections import defaultdict
from dimensionality_reduction.pooled import orthogonal_map

DEST = OUT / "contrast_scaling_pca"
D_SOURCE = OUT / "direction_cosine_top2/supporting"
DISPLAY_AREAS = ["AM-PM", "A-AL-RL"]
SOURCES = [
    "dimensionality-reduction/contrast.py",
    "dimensionality-reduction/pooled.py",
    "analysis_config.py",
]


def direction_centers(x, stim, folds, fold):
    ci = np.searchsorted(np.unique(stim[:, 1]), stim[:, 1])
    di = (stim[:, 0] / 45).astype(int)
    result = []
    for train in [True, False]:
        mask = folds != fold if train else folds == fold
        c = np.array(
            [
                [x[mask & (ci // 4 == pool) & (di == d)].mean(0) for d in range(8)]
                for pool in range(2)
            ]
        )
        assert all(
            (mask & (ci // 4 == pool) & (di == d)).sum() == (48 if train else 12)
            for pool in range(2)
            for d in range(8)
        )
        c -= c.mean(-2, keepdims=True)
        result.append(c)
    return tuple(result)


def gram_target(group_grams):
    """D's empirical training Gram target extended to 16 labels in N10 space."""
    gram = np.mean(group_grams, 0)
    eig, vec = np.linalg.eigh(gram)
    # Best rank-10 embedding of the 16-label mean Gram. Only the target is
    # truncated; rotations preserve all original N10 centroid distances.
    target = vec[:, -10:] * np.sqrt(np.maximum(eig[-10:], 0))
    anchor = np.random.default_rng(20261020).normal(size=(16, 10))
    anchor = anchor.reshape(2, 8, 10)
    anchor -= anchor.mean(1, keepdims=True)
    anchor = anchor.reshape(16, 10)
    target = target @ orthogonal_map(target, anchor)
    return target, float(np.maximum(eig[-10:], 0).sum() / np.maximum(eig, 0).sum())


def register_all_areas(configurations):
    """D's rigid registration and labeled PC sign rule, with a pool axis."""
    a = np.asarray(configurations)
    assert a.shape[:2] == (4, 4) and a.shape[-2:] == (8, 10)
    flat = a.reshape(4, -1, 10)
    target = flat[0].copy()
    for iteration in range(200):
        rotations = orthogonal_map(flat, target)
        aligned = flat @ rotations
        new = aligned.mean(0)
        if np.linalg.norm(new - target) <= 1e-12 * max(np.linalg.norm(target), 1):
            break
        target = new
    rotations = orthogonal_map(flat, target)
    aligned = (flat @ rotations).reshape(a.shape)
    _, s, vt = np.linalg.svd(aligned.reshape(-1, 10), full_matrices=False)
    axes = vt[:3].T
    coords = aligned @ axes
    corners = ((coords[..., :4, :] + coords[..., 4:, :]) / 2).mean((0, 1, 2))
    comparisons = [
        corners[0, 0] - corners[2, 0],
        corners[1, 1] - corners[3, 1],
        corners[0, 2] - corners[1, 2],
    ]
    for k, value in enumerate(comparisons):
        if value < 0:
            axes[:, k] *= -1
    return aligned, aligned @ axes, rotations, axes, s * s / (s * s).sum(), iteration + 1


def run():
    # Preserve every quantitative table and standalone F/G export byte for byte.
    protected = {
        p.name: sha(p)
        for p in DEST.iterdir()
        if p.is_file()
        and (
            p.name.startswith(
                (
                    "quantitative_FG.",
                    "quantitative_FG_",
                    "group_peak",
                    "mouse_peak",
                    "session_radius",
                    "mouse_radius",
                    "group_reverse",
                    "mouse_reverse",
                    "peak_expansion",
                    "reverse_transfer",
                )
            )
        )
    }
    (DEST / "complete.json").unlink(missing_ok=True)
    pops = manifest()["populations"]
    inputs = {}
    by_session = defaultdict(list)
    centers = {}
    selections = {}

    def record(path):
        inputs[str(path.resolve())] = sha(path)

    record(DATA / "manifest.json")
    record(D_SOURCE / "shared_pca_validation.json")
    record(D_SOURCE / "shared_pca.npz")
    for p in pops:
        by_session[p["sid"]].append(p)
    for sid, ps in by_session.items():
        path = DATA / "responses" / f"s{sid:03d}.npz"
        record(path)
        with np.load(path) as z:
            x = z["X"]
            stim = z["stim"]
        path = DATA / "decoding_folds" / f"s{sid:03d}.npy"
        record(path)
        folds = np.load(path)
        centers[sid] = [direction_centers(x, stim, folds, f) for f in range(5)]
        for p in ps:
            selections[p["pid"]] = ids_for(p)
            record(DATA / "selections" / f"p{p['pid']:03d}.npy")
    means = []
    mouse_means = []
    raw_grams = []
    orth_error = 0.0
    gram_error = 0.0
    target_retention = []
    for area in AREAS:
        ps = [p for p in pops if p["area"] == area]
        sums = {p["pid"]: np.zeros((16, 10)) for p in ps}
        gramsums = {p["pid"]: np.zeros((16, 16)) for p in ps}
        for f in range(5):
            arrays = {}
            for p in ps:
                c = centers[p["sid"]][f]
                ids = selections[p["pid"]]
                arrays[p["pid"]] = tuple(
                    np.transpose(v[:, :, ids], (2, 0, 1, 3)).reshape(1000, 16, 10) for v in c
                )
            grams = {pid: np.mean(a @ a.swapaxes(-1, -2), 0) for pid, (a, b) in arrays.items()}
            groups, _ = mouse_average(ps, grams)
            target, retained = gram_target(groups)
            target_retention.append(dict(area=area, fold=f, retained_rank10=retained))
            for p in ps:
                pid = p["pid"]
                a, b = arrays[pid]
                r = orthogonal_map(a, target)
                rb = b @ r
                orth_error = max(orth_error, float(abs(r.swapaxes(-1, -2) @ r - np.eye(10)).max()))
                before = b @ b.swapaxes(-1, -2)
                after = rb @ rb.swapaxes(-1, -2)
                gram_error = max(gram_error, float(abs(after - before).max()))
                np.testing.assert_allclose(after, before, atol=1e-11, rtol=1e-11)
                sums[pid] += rb.mean(0) / 5
                gramsums[pid] += before.mean(0) / 5
            print(f"{area}: D-style fold {f + 1}/5 complete", flush=True)
        avg, mice = mouse_average(ps, sums)
        means.append(avg.reshape(4, 2, 8, 10))
        raw, _ = mouse_average(ps, gramsums)
        raw_grams.append(raw)
        mouse_means.extend(
            dict(area=area, fi=m["fi"], mouse=m["mouse"], values=m["values"].reshape(2, 8, 10))
            for m in mice
        )
    before = np.array(means)
    aligned, directions, rotations, axes, explained, iters = register_all_areas(before)
    flat = before.reshape(4, -1, 10)
    after = aligned.reshape(4, -1, 10)
    np.testing.assert_allclose(
        flat @ flat.swapaxes(-1, -2), after @ after.swapaxes(-1, -2), atol=1e-12
    )
    # Only now collapse antipodal directions for display.
    orientations = (directions[..., :4, :] + directions[..., 4:, :]) / 2
    orientation_full = (aligned[..., :4, :] + aligned[..., 4:, :]) / 2
    displayed = [AREAS.index(a) for a in DISPLAY_AREAS]
    np.savez_compressed(
        DEST / "pca_geometry.npz",
        coordinates=orientations[displayed],
        aligned_means=orientation_full[displayed],
        direction_coordinates=directions,
        direction_aligned_means=aligned,
        pre_registration_means=before,
        all_orientation_coordinates=orientations,
        all_orientation_aligned_means=orientation_full,
        raw_grams=np.array(raw_grams),
        rotations=rotations,
        axes=axes,
        explained=explained,
        areas=np.array(AREAS),
        display_areas=np.array(DISPLAY_AREAS),
    )
    coordinate_rows = []
    direction_rows = []
    mouse_rows = []
    audits = []
    for ai, area in enumerate(AREAS):
        for fi in range(4):
            for pool in range(2):
                meta = dict(area=area, fi=fi, condition=LABELS[fi], pool=["low", "high"][pool])
                for d in range(8):
                    direction_rows.append(
                        dict(
                            **meta,
                            direction=d * 45,
                            **{f"PC{k + 1}": directions[ai, fi, pool, d, k] for k in range(3)},
                        )
                    )
                for o in range(4):
                    coordinate_rows.append(
                        dict(
                            **meta,
                            orientation=o * 45,
                            **{f"PC{k + 1}": orientations[ai, fi, pool, o, k] for k in range(3)},
                        )
                    )
                full = np.mean(np.sum(orientation_full[ai, fi, pool] ** 2, -1))
                projected = np.mean(np.sum(orientations[ai, fi, pool] ** 2, -1))
                g = raw_grams[ai][fi, pool * 8 : (pool + 1) * 8, pool * 8 : (pool + 1) * 8]
                pair = np.concatenate([np.eye(4), np.eye(4)], axis=1) / 2
                raw = np.trace(pair @ g @ pair.T) / 4
                audits.append(
                    dict(
                        **meta,
                        mean_individual_radius=raw,
                        aligned_mean_radius=full,
                        projected_mean_radius=projected,
                        retained_3d=projected / full,
                        direction_full_radius=np.mean(np.sum(aligned[ai, fi, pool] ** 2, -1)),
                        direction_projected_radius=np.mean(
                            np.sum(directions[ai, fi, pool] ** 2, -1)
                        ),
                    )
                )
        for m in [m for m in mouse_means if m["area"] == area]:
            d = m["values"] @ rotations[ai] @ axes
            c = (d[:, :4] + d[:, 4:]) / 2
            for pool in range(2):
                for o in range(4):
                    mouse_rows.append(
                        dict(
                            area=area,
                            fi=m["fi"],
                            mouse=m["mouse"],
                            pool=["low", "high"][pool],
                            orientation=o * 45,
                            **{f"PC{k + 1}": c[pool, o, k] for k in range(3)},
                        )
                    )
    for name, rows in [
        ("pca_coordinates", coordinate_rows),
        ("pca_direction_coordinates", direction_rows),
        ("pca_mouse_coordinates", mouse_rows),
        ("pca_radius_audit", audits),
    ]:
        pd.DataFrame(rows).to_csv(DEST / f"{name}.csv", index=False)
    camera = json.loads((D_SOURCE / "shared_pca_validation.json").read_text())["camera"]
    for p, h in inputs.items():
        assert sha(p) == h
    for p, h in protected.items():
        assert sha(DEST / p) == h
    v = dict(
        validated=True,
        populations=len(pops),
        areas=AREAS,
        display_areas=DISPLAY_AREAS,
        pools_original_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        trials_per_direction_per_pool=dict(all=60, train=48, heldout=12),
        draws=1000,
        folds=5,
        fit_rows=256,
        fit_labels="eight directions",
        display_labels="four opposite-direction coordinate means after PCA",
        centering="Independently subtract the eight-direction mean within each pool, training/heldout split and draw.",
        alignment="D-style training-only empirical target within area/fold, equal conditions and mice. Rank-10 Gram embedding of 16 labels. One orthogonal map jointly acts on both pools and all eight directions; apply to held-out centroids. Average folds, draws, sessions, mice.",
        pca="D-style equal-area generalized orthogonal Procrustes across all 64 condition/pool/direction rows per area, then one SVD of all 256 rows. D labeled-corner PC sign convention. No scaling. Average opposing coordinates only after projection; display AM-PM and A-AL-RL.",
        camera=camera,
        iterations=iters,
        pca_retained_3d=float(explained[:3].sum()),
        rotation_orthogonality_error=orth_error,
        rotation_gram_error=gram_error,
        target_rank10_retention=target_retention,
        protected_FG=protected,
        inputs=inputs,
        sources={n: sha(ROOT / n) for n in SOURCES},
    )
    (DEST / "pca_validation.json").write_text(json.dumps(v, indent=2) + "\n")
    print(
        "Validated all-area direction PCA; retained 3D:",
        v["pca_retained_3d"],
        "camera:",
        camera,
        flush=True,
    )

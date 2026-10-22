"""Panels B, D inset and F, with the full-circle B supplement."""

import numpy as np
import pandas as pd
from analysis_config import DATA, OUT, manifest, ids_for
from geometry import cosine, radius
from geometry.peak import pooled_centers, radius_curve, peak_mouse_curves
from decoding.run import summarize


def peak():
    dest = OUT / "contrast_scaling_pca"
    dest.mkdir(parents=True, exist_ok=True)
    times = np.load(DATA / "temporal/time.npy")
    frames = np.flatnonzero((times >= 0) & (times <= 0.5))
    rows = []
    populations = manifest()["populations"]
    for sid in sorted({p["sid"] for p in populations}):
        with np.load(DATA / "responses" / f"s{sid:03d}.npz") as z:
            stim = z["stim"]
        ci = np.searchsorted(np.unique(stim[:, 1]), stim[:, 1])
        orientation = (stim[:, 0] / 45).astype(int) % 4
        keep = (ci < 2) | (ci >= 6)
        pools = (ci[keep] >= 6).astype(int)
        raw = np.load(DATA / "temporal" / f"s{sid:03d}.npy", mmap_mode="r")[frames][:, keep].astype(
            float
        )
        centers = pooled_centers(raw, pools, orientation[keep], 2)
        for p in [p for p in populations if p["sid"] == sid]:
            values = radius_curve(centers, ids_for(p))
            meta = {k: p[k] for k in ["pid", "sid", "area", "fi", "mouse"]}
            for ti, frame in enumerate(frames):
                for b in range(2):
                    rows.append(
                        dict(
                            **meta,
                            pool="coarse",
                            step=b + 1,
                            frame=int(frame),
                            time=float(times[frame]),
                            value=values[ti, b],
                        )
                    )
    sessions = pd.DataFrame(rows)
    curves, peaks = peak_mouse_curves(sessions)
    groups = summarize(peaks, ["area", "fi", "step"])
    for name, frame in [
        ("session_radius_timecourses", sessions),
        ("mouse_radius_timecourses", curves),
        ("mouse_peak_radius", peaks),
        ("group_peak_radius", groups),
    ]:
        frame.to_csv(dest / f"{name}.csv", index=False)


def run():
    cosine.analyze()
    radius.run()
    peak()


if __name__ == "__main__":
    run()

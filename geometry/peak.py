import numpy as np
import pandas as pd


def radius_curve(centers, ids):
    """time x contrast x orientation x neuron -> mean-draw full-N10 radius."""
    centered = centers - centers.mean(-2, keepdims=True)
    weights = np.bincount(ids.ravel(), minlength=centers.shape[-1]) / len(ids)
    return np.einsum("tcon,n->tc", centered**2, weights) / centers.shape[-2]


def pooled_centers(raw, ci, ori, count):
    return np.stack(
        [
            np.stack(
                [raw[:, (ci == c) & (ori == o)].mean(1, dtype=np.float64) for o in range(4)], axis=1
            )
            for c in range(count)
        ],
        axis=1,
    )


def peak_mouse_curves(sessions):
    """Keep the same mouse's full temporal profile together before maximizing."""
    curves = sessions.groupby(
        ["area", "fi", "mouse", "pool", "step", "frame", "time"], as_index=False
    ).value.mean()
    rows = []
    for key, sub in curves.groupby(["area", "fi", "mouse", "pool", "step"]):
        sub = sub.sort_values("frame")
        peak = sub.iloc[np.argmax(sub.value.to_numpy())]
        rows.append(
            dict(
                zip(["area", "fi", "mouse", "pool", "step"], key),
                value=float(peak.value),
                peak_time=float(peak.time),
                peak_frame=int(peak.frame),
            )
        )
    return curves, pd.DataFrame(rows)

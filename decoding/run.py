"""Run C (pooled) and G (adjacent-bin transfer); retain draw-level checkpoints."""

import json
import numpy as np
import pandas as pd
from analysis_config import ROOT, OUT, manifest, sha, LABELS, bootstrap_weights
from decoding import pooled, transfer


def summarize(mice, keys):
    rows = []
    for key, group in mice.groupby(keys):
        meta = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        values = group.sort_values("mouse").value.to_numpy()
        low, high = np.quantile(
            bootstrap_weights(len(values), int(meta["fi"])) @ values, [0.025, 0.975]
        )
        rows.append(
            dict(
                **meta,
                condition=LABELS[int(meta["fi"])],
                mean=values.mean(),
                ci_low=low,
                ci_high=high,
                mice=len(values),
            )
        )
    return pd.DataFrame(rows)


def run():
    code = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ["decoding", "preprocess"]
        for p in sorted((ROOT / folder).glob("*.py"))
    }
    code.update(config=sha(ROOT / "analysis_config.py"), protocol=sha(ROOT / "protocol.json"))
    crows, grows = [], []
    for p in manifest()["populations"]:
        stamp = json.dumps(dict(code=code, inputs=transfer.input_signature(p)), sort_keys=True)
        meta = {k: p[k] for k in ["pid", "sid", "area", "fi", "mouse"]}
        for module in [pooled, transfer]:
            folder = module.DEST / "draws"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"p{p['pid']:03d}.npz"
            cached = False
            if path.exists():
                with np.load(path) as z:
                    cached = str(z["signature"]) == stamp
            if not cached:
                if module is pooled:
                    accuracy, confusion, error = pooled.decode(p)
                    result = dict(accuracy=accuracy, confusion=confusion, score_error=error)
                else:
                    result = transfer.compute(p)
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **result, signature=stamp)
                tmp.replace(path)
            with np.load(path) as z:
                if module is pooled:
                    for i, task in enumerate(pooled.TASKS):
                        crows.append(
                            dict(**meta, task=task, value=float(z["accuracy"][:, i].mean()))
                        )
                else:
                    matrix = z["transfer"].mean((0, 3)) * 100
                    for step, (source, target) in enumerate([(3, 2), (2, 1), (1, 0)], 1):
                        grows.append(
                            dict(
                                **meta,
                                step=step,
                                source_bin=source + 1,
                                target_bin=target + 1,
                                value=matrix[source, target],
                            )
                        )
        print(f"LDA population {p['pid']} complete", flush=True)
    for rows, dest, prefix, keys in [
        (crows, pooled.DEST, "decoding", ["task"]),
        (grows, OUT / "contrast_scaling_pca", "reverse_transfer", ["step"]),
    ]:
        dest.mkdir(parents=True, exist_ok=True)
        sessions = pd.DataFrame(rows)
        mice = sessions.groupby(["area", "fi", "mouse"] + keys, as_index=False).value.mean()
        groups = summarize(mice, ["area", "fi"] + keys)
        if prefix == "decoding":
            names = ["decoding_sessions", "decoding_mice", "decoding_groups"]
        else:
            names = ["session_reverse_transfer", "mouse_reverse_transfer", "group_reverse_transfer"]
        for name, frame in zip(names, [sessions, mice, groups]):
            frame.to_csv(dest / f"{name}.csv", index=False)
    (OUT / "decoding_complete.json").write_text(
        json.dumps(
            dict(populations=len(manifest()["populations"]), validated=True, code=code), indent=2
        )
    )


if __name__ == "__main__":
    run()

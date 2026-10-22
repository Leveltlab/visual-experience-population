"""Prepare raw deconvolved activity, the requested averaging window and N10 subsets."""

import json
import numpy as np
from analysis_config import ROOT, DATA, OUT, AREAS, N, SIGNAL, protocol, sha
from preprocess.folds import make_folds
import os
from pathlib import Path

RAW = Path(os.environ.get("VEP_PYDATA_DIR", ROOT / "preprocess/pydata")) / "pooled"
REGIONS = {"V1": [6], "AM-PM": [0, 1], "A-AL-RL": [2, 3, 4], "LM": [5]}


def raw_trials(signal, time):
    """Retain finite neurons without centering or scaling any trial."""
    good = np.isfinite(signal).all(axis=(0, 1))
    baseline = (time >= -1) & (time < 0)
    mu = signal[baseline].mean(0, dtype=np.float64)
    sd = signal[baseline].std(0, dtype=np.float64, ddof=0)
    return signal, good, mu, sd


def main():
    for folder in [DATA / "responses", DATA / "temporal", DATA / "selections", OUT]:
        folder.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "preprocess/registry.json").read_text())
    sources = []
    pops = []
    audit = []
    hashes = {}
    cached = {}
    for s in old["sources"]:
        key = f"{s['group']}_{s['timepoint']}"
        if key not in cached:
            path = RAW / f"{SIGNAL}_{key}.npy"
            meta = json.loads((RAW / f"metadata_{key}.json").read_text())
            cached[key] = (np.load(path, mmap_mode="r"), meta)
            hashes[str(path)] = sha(path)
        raw, meta = cached[key]
        sess = next(v for v in meta["sessions"] if v["session_key"] == s["session"])
        assert sess["source_file"] == s["source"] and sess["neuron_start"] == s["global_start"]
        start, stop = sess["neuron_start"], sess["neuron_stop"]
        time = np.array(meta["numeric"]["time"])
        frames = (time >= -0.5) & (time <= 1.5)
        window = (time >= protocol()["averaged_window"][0]) & (
            time < protocol()["averaged_window"][1]
        )
        stim = np.array(meta["numeric"]["stim"])[sess["behavior_index"]]
        folds = make_folds(stim, s["sid"])
        signal = np.asarray(raw[:, :, start:stop], dtype=np.float64)
        z, good, mu, sd = raw_trials(signal, time)
        regions = np.array(meta["numeric"]["region"])[start:stop]
        neurons = {a: np.flatnonzero(good & np.isin(regions, REGIONS[a])).tolist() for a in AREAS}
        neurons["ALL"] = sorted(sum(neurons.values(), []))
        source = {k: v for k, v in s.items() if k != "neurons"}
        source["neurons"] = neurons
        sources.append(source)
        X = z[window].mean(0)
        np.savez_compressed(
            DATA / "responses" / f"s{s['sid']:03d}.npz",
            X=X,
            stim=stim,
            folds=folds,
            baseline_mean=mu,
            baseline_sd=sd,
            eligible=good,
        )
        np.save(DATA / "temporal" / f"s{s['sid']:03d}.npy", z[frames].astype("float32"))
        np.save(DATA / "temporal/time.npy", time[frames])
        for ai, area in enumerate(AREAS):
            eligible = neurons[area]
            if len(eligible) < N:
                continue
            pid = len(pops)
            p = {k: v for k, v in source.items() if k not in ["neurons", "global_start", "source"]}
            p.update(pid=pid, n=N, mode="session", area=area, eligible=len(eligible))
            pops.append(p)
            rng = np.random.default_rng([protocol()["selection_seed"], s["sid"], ai, N])
            selected = np.array([rng.choice(eligible, N, replace=False) for _ in range(1000)])
            np.save(DATA / "selections" / f"p{pid:03d}.npy", selected)
        audit.append(
            dict(
                sid=s["sid"],
                neurons=stop - start,
                eligible=int(good.sum()),
                excluded=int((~good).sum()),
                zero_baseline_trial_neurons=int((sd == 0).sum()),
                raw_response_reconstruction_error=float(abs(X - signal[window].mean(0)).max()),
                baseline_min_sd=float(sd[:, good].min()),
                response_max_abs=float(abs(X[:, good]).max()),
            )
        )
        print(f"{SIGNAL} s{s['sid']:03d}: {good.sum()}/{len(good)} valid neurons", flush=True)
    result = dict(
        populations=pops,
        sources=sources,
        contrasts=old["contrasts"],
        draws=1000,
        smoke=False,
        input_hashes=hashes,
    )
    (DATA / "manifest.json").write_text(json.dumps(result, indent=2))
    (OUT / "input_audit.json").write_text(
        json.dumps(
            dict(
                protocol=protocol(),
                sessions=audit,
                populations=len(pops),
                baseline_frames=time[(time >= -1) & (time < 0)].tolist(),
                response_frames=time[window].tolist(),
                temporal_frames=time[frames].tolist(),
            ),
            indent=2,
        )
    )
    print("Prepared", len(pops), "N10 populations", flush=True)


if __name__ == "__main__":
    main()

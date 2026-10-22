# Input preparation

`matlab2py.py` reads the original four-folder MATLAB hierarchy:
`darkReared_baseline`, `darkReared_chronic`, `normReared_baseline`, and `normReared_chronic`.
It writes pooled and per-mouse NumPy arrays plus JSON metadata, preserving session,
neuron, stimulus and time indexing. Run `--dry-run` to inspect the conversion plan.
`--overwrite` explicitly replaces an existing conversion directory.

`prepare.py` uses `registry.json` to preserve the published session IDs and ordering.
It matches each session against metadata, selects finite atlas-assigned neurons,
retains raw signal values, averages [0, 0.5) s, and saves native temporal frames
[-0.5, 1.5] s. Region groups are V1, AM-PM, A-AL-RL and LM.

Each eligible session–area receives 1,000 seeded subsets of ten distinct neurons.
`folds.py` assigns three of the 15 trials in every direction × contrast cell to each
of five folds. No held-out information enters classifier fitting or alignment.

The registry is for the registered experiment, not an automatic session discovery
interface for a new study. Raw recordings are not distributed in this repository.

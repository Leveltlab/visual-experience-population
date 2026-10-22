# Prepare recordings

```sh
uv run python -m preprocess.prepare --input /path/to/recordings
```

The input directory contains `darkReared_baseline`, `normReared_baseline`,
`darkReared_chronic`, and `normReared_chronic`, each holding `.mat` files. The mouse
ID is the filename prefix before the first underscore. All files are discovered
in that condition order, with lexicographically sorted filenames within each
folder. Keep names and ordering stable to preserve session IDs and random draws.

The loader reads only these MATLAB v7.3/HDF5 fields:

| Field | MATLAB shape / meaning |
| --- | --- |
| `Res/CaDeconCorrected` | time × trial × neuron, raw deconvolved activity |
| `Res/ax` | sample times in seconds relative to stimulus onset |
| `info/Stim/log` | trial × 2, direction in degrees and contrast |
| `ABAroi/region` | one region ID per neuron: 1 AM, 2 PM, 3 A, 4 AL, 5 RL, 6 LM, 7 V1; 0 unassigned |

`--signal calcium` reads `Res/CaSigCorrected` instead. A missing region field
leaves all neurons in that recording unassigned. Other fields are not required.
The loader reverses the HDF5 dimension order to recover the MATLAB array shapes.
Signals, times and labels are read as float32, with response averaging in float64.
All recordings must share time and contrast grids. Each must have 960 trials:
eight directions (0°, 45°, …, 315°), eight contrasts and 15 repeats per cell.

Preparation processes one file at a time. It excludes nonfinite neurons, groups
regions into V1, AM-PM, A-AL-RL and LM, saves raw [0, 0.5) s averages and native
[-0.5, 1.5] s frames, then draws 1,000 ten-neuron subsets without replacement.
Each direction–contrast cell contributes three trials to each of five folds.
No centering or scaling is applied to activity.

`--output /path/to/prepared` selects a destination. Defaults are `data/spikes` or
`data/calcium`; use the corresponding `VEP_DATA_DIR` when running later stages
with a custom destination. `--overwrite` explicitly rebuilds prepared data. A
failed preparation leaves the existing destination intact. Generated data and
metadata are local run products and are excluded from Git.

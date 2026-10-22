# Visual experience shapes population representations

Population geometry, linear decoding and shared PCA views of visual cortical responses.

[Preprint · bioRxiv](https://doi.org/10.1101/2024.08.31.610645) · [Figure notebook](plotting/figures.ipynb) · [Analysis protocol](protocol.json)

## Population geometry

<table>
<tr>
<td width="52%" valign="top"><img src="plotting/previews/panel_D.png" alt="D: population orientation representation across four visual areas, in a 2×2 layout"></td>
<td width="48%" valign="top"><img src="plotting/previews/panel_H.png" alt="H: contrast-dependent population spread in medial and lateral dorsal streams"></td>
</tr>
<tr>
<td valign="top"><strong>D.</strong> Dark rearing collapses population orientation representation in dorsal stream higher visual areas. Lateral dorsal stream recovers with visual experience, while medial dorsal stream does not.</td>
<td valign="top"><strong>H.</strong> Stimulus contrast controls spread of population orientation representation. Medial dorsal stream fails to recover this mechanism with visual experience while lateral dorsal stream succeeds.</td>
</tr>
</table>

NR / DR (normally / dark-reared), baseline / chronic (P90 / P120 = P90+30 days of visual experience). V1: Primary visual cortex; AM-PM: Medial dorsal stream; A-AL-RL: Lateral dorsal stream; LM: Ventral stream.

The README layouts and their [rendering script](plotting/previews/render.py) are in `plotting/previews/`; regenerate them with `uv run python -m plotting.previews.render`.

## Reproduce the figures

```sh
uv sync --locked
uv run jupyter lab plotting/figures.ipynb
```

Run the notebook to export **B, C, D, F, G and H**, plus the B supplements and D-radius barplots listed below. Each panel is saved independently as **PNG, PDF, editable SVG and outlined SVG** in `plotting/exports/`.

The included [`plotting/source_data/`](plotting/source_data/) contains only the 19 numerical inputs needed to draw these panels. No recordings or R installation are needed to reproduce the figures. For a noninteractive export:

```sh
uv run python -m plotting
```

Myriad Pro is used when installed; otherwise the exporter uses Matplotlib's bundled DejaVu Sans. Set `VEP_FONT_DIR` to a directory containing licensed Myriad Pro fonts, or `VEP_FONT` to an installed family. The previews use Myriad Pro. A font change affects typography, not plotted values or physical panel scale.

## Analyze recordings

Place MATLAB v7.3/HDF5 recordings in this structure:

```text
recordings/
├── darkReared_baseline/    Mouse_session.mat
├── normReared_baseline/    Mouse_session.mat
├── darkReared_chronic/     Mouse_session.mat
└── normReared_chronic/     Mouse_session.mat
```

The filename prefix before the first underscore identifies the mouse; use the same prefix across timepoints. Files are discovered automatically, in the condition order above and then filename order. That order defines deterministic session IDs and sampling seeds.

Each recording supplies `Res/CaDeconCorrected` (time × trial × neuron in MATLAB), `Res/ax` (seconds relative to stimulus onset), `info/Stim/log` (trial × 2: direction in degrees, contrast), and `ABAroi/region` (one atlas ID per neuron). The design has eight directions, eight contrast levels and 15 trials per direction–contrast cell. Atlas IDs 1–7 map to AM, PM, A, AL, RL, LM and V1; zero or a missing region field is unassigned. Missing behavior, anatomical coordinate and calcium fields do not prevent deconvolved analysis. See [`preprocess/README.md`](preprocess/README.md).

For the statistical analyses, install R and nparLD **2.3.3** with its dependencies. Use the source package because an older R binary repository may offer an older version:

```sh
Rscript -e 'install.packages("nparLD", repos="https://cloud.r-project.org", type="source"); stopifnot(as.character(packageVersion("nparLD")) == "2.3.3")'
```

Then run the complete pipeline:

```sh
uv run python run.py --input /path/to/recordings
```

This prepares the signals, computes geometry, LDA and PCA, runs mouse-level inference, and exports all panels to `results/spikes/panels/`. Preparation also works independently:

```sh
uv run python -m preprocess.prepare --input /path/to/recordings
uv run python run.py --stage geometry
uv run python run.py --stage decoding
uv run python run.py --stage pca
uv run python run.py --stage statistics
uv run python run.py --stage supplement
uv run python run.py --stage plot
```

Preparation refuses to replace existing data unless `--overwrite` is supplied. LDA reuses checkpoints only when their inputs and code match. To use raw calcium instead, pass `--signal calcium`; its input field is `Res/CaSigCorrected`, and its data/results are written separately. Only deconvolved results are included here.

### Numerical protocol

- **10 neurons**, **1,000 shared draws** and **five fixed stratified folds** per session–area population. Neurons must have finite values across the recording. Activity is used without baseline subtraction, variance normalization or additional scaling.
- B/C/D/G/H average **0 ≤ t < 0.5 s**. F takes the peak over native frames in **0 ≤ t ≤ 0.5 s**, after averaging draws and sessions within each mouse.
- B/C/D pool the **top two contrasts**. F compares **bottom two versus top two**. G uses four adjacent pairs and displays **4→3, 3→2, 2→1**. H pools **bottom four versus top four**, preserving all eight directions until after PCA.
- LDA uses analytic Ledoit–Wolf shrinkage of pooled training residuals and class-frequency priors. Held-out trials do not enter fitting. The estimator differs from sklearn LDA's classwise `shrinkage="auto"`.
- Sessions are averaged within mouse; mice receive equal weight. Intervals use 5,000 mouse-bootstrap resamples. Draws and folds are technical repeats.
- Within-area alignment uses training centroids and rotates held-out centroids without scaling. Subsequent cross-area registration and PCA use group means and are descriptive. Ordinary cosine, distance and radius retain finite-trial noise; F also includes peak-selection bias.

Factorial inference uses nparLD 2.3.3 with rearing × area × time × panel-specific factor, incomplete records retained, and Holm correction across four main effects within each panel. C uses chance-normalized accuracy; F uses log radius. Two-sided exact NR–DR rank-sum tests use 180° similarity (B), orientation accuracy (C), full-space radius (D), log(high/low) peak radius (F), and transfer accuracy (G). Holm families contain two timepoints per area/endpoint for B–F and six timepoint × transition comparisons per area for G. The included figures cover 116 session–area populations from 72 contributing sessions and 27 mice; small groups limit inference.

### Supplementary panels

The notebook exports each supplement separately:

| Export name | Content |
| --- | --- |
| `B_angular`, `B_euclidean`, `B_cross_contrast` | Angular cosine, endpoint Euclidean distance, and cross-contrast cosine |
| `B_angular_distance` | Full-angle Euclidean distance, including individual mouse values |
| `orientation_contrasts` | Cosine and distance contrasts across areas |
| `cosine_contrast`, `euclidean_contrast` | Compact 150 × 100 pt contrast panels |
| `B_cosine_endpoints`, `B_euclidean_endpoints` | B curves with NR–DR tests at 180° and 90° |
| `B_cosine_contrast`, `B_euclidean_contrast` | B curves annotated with tests of the within-mouse contrast |
| `radius_bars` | D's full-space squared radius across areas, on a shared scale (150 × 100 pt) |
| `radius_V1`, `radius_AM_PM`, `radius_A_AL_RL`, `radius_LM` | Individual D-radius barplots with area-specific scales (55 × 100 pt) |

Orientation contrasts subtract angles **within each mouse and timepoint**: C(180°) − C(90°) for cosine, d(90°) − d(180°) for distance. Pointwise 95% intervals bootstrap these mouse differences, preserving their covariance. Baseline and chronic retain every eligible mouse independently; longitudinal pairing is not required.

These exploratory follow-ups use two-sided exact NR–DR rank-sum tests, with Holm correction over two timepoints separately for each area, metric and endpoint. The numerical table also reports the stricter six-test endpoint × timepoint correction (`p_holm6_sensitivity`); plotted stars use the two-test family. No correction spans areas or metrics. Tests concern distributions of mouse values, not specifically their means. The alternative annotations do not change the main B panel's tests.

Angular distances use the same pooled centroids and all 1,000 ten-neuron draws, taking distances before averaging pairs, draws, sessions within mouse, and mice. Zero self-distance is omitted from the plot; mirrored angles are identical and are not tested separately. Euclidean differences retain response scale and finite-trial noise. D-radius bars use the eight-direction squared radius in full ten-neuron space, displayed ×1,000, with the existing D tests and intervals.

## Organization

| Folder | Analyses |
| --- | --- |
| `preprocess/` | MATLAB loading, response preparation, eligibility, neuron sampling and trial folds |
| `geometry/` | B and its supplements, D's radius inset, F's peak radius and inference |
| `decoding/` | Ledoit–Wolf LDA for C and G |
| `dimensionality-reduction/` | Alignment and shared PCA for D/H; imported as `dimensionality_reduction` |
| `plotting/` | Panel renderers, source data, README previews and the final notebook |
| `tests/` | Numerical checks, MATLAB ingestion and full pipeline/export tests |

Prepared data defaults to `data/<signal>/`, results to `results/<signal>/`. Set `VEP_DATA_DIR` or `VEP_RESULTS_DIR` before starting Python to change those locations. For another results folder, use `uv run python -m plotting --source /path/to/results --output /path/to/panels`, or set `RESULTS` in the notebook and restart its kernel.

Raw recordings, intermediate arrays, caches, environments and generated exports are excluded from Git. The only bundled analysis data are the figure inputs in `source_data`; D/H preview images support this README. Notebook outputs are cleared to keep the repository small.

## Tests

```sh
uv run pytest -q
uv run pytest -q -m integration tests/test_exports.py
# Requires R with nparLD 2.3.3; generates its own MATLAB recordings:
uv run pytest -q -m integration tests/test_pipeline.py
```

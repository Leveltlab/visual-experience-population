# Validation record — 22 October 2026

This is a local extraction of the active flagship implementation. No GitHub repository,
remote push, website update, or publication was performed.

## Checked

- 23 portable tests cover cosine and radius invariants, trial pooling, disjoint split halves,
  peak ordering, held-out isolation, rigid PCA registration, LDA versus an independent
  sklearn covariance reference, silent-data ties, snapshot hashes, and accepted inference.
- The notebook executed in the repository's locked environment and exported all nine
  individual panels in PNG, PDF, editable SVG and outlined SVG. D/H previews were
  regenerated from that notebook. SVGs contain no raster images; outlined variants
  contain no text nodes. The exported panel layouts were visually inspected.
- Regenerated all 73 session fold arrays and all 116 × 1,000 neuron selections identically
  to the active prepared inputs.
- Recomputed B, the full-space D radius inset and F for every registered population from
  the existing prepared responses. Means and confidence limits match the frozen plotting
  inputs to an absolute tolerance of 1e-12.
- Recomputed D and H for the complete dataset using the extracted PCA pipeline. Both
  exported coordinate arrays match the frozen arrays exactly (maximum absolute error 0).
- Recomputed all 1,000 C draws and the full 4 × 4 × 5 G transfer array for population 0.
  Accuracy/confusion outputs match the source at tolerance 1e-7; independent score
  reference errors were below 3e-14. This is one complete population, not a complete
  rerun of all 116 decoding populations.
- Reconstructed every B/C/F/G factorial input and all corrected B/C/D/F/G rank-sum
  comparisons from the bundled mouse summaries. Inputs agree to floating-point
  roundoff and adjusted rank-sum p-values to 1e-12.
- Ran the extracted R script with nparLD 2.3.3. All 16 accepted main-effect ATS statistics,
  degrees of freedom and within-panel Holm-adjusted p-values reproduce the saved tables
  (relative tolerance 1e-10, absolute tolerance 1e-12).
- Ruff's undefined-name/unused-import checks pass. The Python environment is pinned
  in `uv.lock`; the R scripts enforce nparLD 2.3.3.

## Limits of this check

Original MATLAB conversion and the full raw-recording preparation were not rerun.
Geometry/PCA validation used the existing prepared deconvolved responses read-only.
The all-population LDA production run and calcium path were not run. Supplementary
cosine kernels have synthetic tests; their entire production run was not repeated.
The bundled figures use the exact saved accepted tables, with a separate hash manifest.
Historical validation JSON files inside the snapshot retain their original-run context;
they do not certify a newly computed run in this repository.

The accepted font is Myriad Pro Regular. It must be available locally and is not included.
Public data access and a redistribution license are not established by this local extraction.

"""Training-only alignment, followed by descriptive shared PCA for D and H."""

from analysis_config import AREAS, OUT, manifest
from dimensionality_reduction import pooled, shared, contrast
import pandas as pd


def run():
    pooled.DEST.mkdir(parents=True, exist_ok=True)
    (OUT / "contrast_scaling_pca").mkdir(parents=True, exist_ok=True)
    rows = []
    for area in AREAS:
        audits, _ = pooled.embed(area, [p for p in manifest()["populations"] if p["area"] == area])
        rows.extend(audits)
    pd.DataFrame(rows).to_csv(pooled.DEST / "pca_audit.csv", index=False)
    shared.build()
    contrast.run()


if __name__ == "__main__":
    run()

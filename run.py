"""Run only the flagship analyses. Use --stage to resume at an individual family."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
import argparse
import importlib

STAGES = {
    "prepare": [("preprocess.prepare", "main"), ("preprocess.folds", "main")],
    "geometry": [("geometry.run", "run")],
    "decoding": [("decoding.run", "run")],
    "pca": [("dimensionality_reduction.run", "run")],
    "statistics": [("geometry.inference", "run")],
    "supplement": [("geometry.euclidean", "analyze"), ("geometry.cross_contrast", "analyze")],
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    args = parser.parse_args()
    for stage in STAGES if args.stage == "all" else [args.stage]:
        for module, function in STAGES[stage]:
            getattr(importlib.import_module(module), function)()

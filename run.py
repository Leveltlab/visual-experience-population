"""Run preparation, figure analyses and individual panel exports."""

import argparse
import importlib
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
STAGES = {
    "geometry": [("geometry.run", "run")],
    "decoding": [("decoding.run", "run")],
    "pca": [("dimensionality_reduction.run", "run")],
    "statistics": [("geometry.inference", "run")],
    "supplement": [
        ("geometry.euclidean", "analyze"),
        ("geometry.cross_contrast", "analyze"),
        ("geometry.orientation_contrasts", "analyze"),
    ],
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "prepare", *STAGES, "plot"], default="all")
    parser.add_argument(
        "--input", type=Path, help="Root containing the four MATLAB condition folders"
    )
    parser.add_argument(
        "--signal", choices=["spikes", "calcium"], default=os.environ.get("VEP_SIGNAL", "spikes")
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild prepared data")
    args = parser.parse_args()
    if args.stage in ("all", "prepare") and args.input is None:
        parser.error("--input is required for preparation")
    os.environ["VEP_SIGNAL"] = args.signal
    from analysis_config import DATA, OUT

    if args.stage in ("all", "prepare"):
        from preprocess.prepare import prepare

        prepare(args.input, DATA, args.signal, args.overwrite)
    for stage in STAGES if args.stage == "all" else [args.stage]:
        for module, function in STAGES.get(stage, []):
            getattr(importlib.import_module(module), function)()
    if args.stage in ("all", "plot"):
        from plotting.export import export_all

        export_all(OUT / "panels")

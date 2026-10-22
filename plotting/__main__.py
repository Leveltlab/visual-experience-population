"""Export all nine panels without opening Jupyter."""

import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path(__file__).with_name("source_data"))
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("exports"))
args = parser.parse_args()
os.environ["VEP_RESULTS_DIR"] = str(args.source.resolve())
from plotting.export import export_all

export_all(args.output)
print(f"Exported all panels to {args.output.resolve()}")

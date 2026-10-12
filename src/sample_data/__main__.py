"""Command-line entry point: `python -m src.sample_data`."""

import argparse
from pathlib import Path

import yaml

from .builder import build_all


def main() -> None:
    """Parse arguments and build the sample datasets found in the datasets folder."""
    parser = argparse.ArgumentParser(description="Build sample probe/image datasets from public sources.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    parser.add_argument("--only", nargs="+", default=None, help="build only these dataset names")
    args = parser.parse_args()
    build_all(yaml.safe_load(args.config.read_text()), args.only)


if __name__ == "__main__":
    main()

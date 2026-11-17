import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from port_twin_app.services.marine_data.generation import (
    generate_shallow_water_datasets,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated marine shallow-water datasets."
    )
    parser.add_argument(
        "--version",
        choices=("v1", "v2"),
        default="v1",
        help="Dataset version to generate (default: v1).",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Output directory (defaults to the generation service directory).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Optional list of scenario names to generate.",
    )

    args = parser.parse_args()

    kwargs = {"dataset_version": args.version}

    if args.output_directory:
        kwargs["output_directory"] = args.output_directory

    if args.scenarios:
        kwargs["scenario_names"] = tuple(args.scenarios)

    files = generate_shallow_water_datasets(**kwargs)

    print(f"\nGenerated {len(files)} {args.version} scenario(s):")
    for name, path in files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

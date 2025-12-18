"""Utility for staging archived calligraphy data for preprocessing.

This script replaces the legacy web crawler with a simple filesystem
copy that makes the archived dataset available under a user-specified
output directory. The preprocessing and training pipelines can then
consume the staged data without any changes to their workflows.
"""

import argparse
import shutil
from pathlib import Path


def copy_archive(archive_dir: Path, output_dir: Path) -> None:
    """Copy the archived dataset to the desired output directory.

    Args:
        archive_dir: Location of the provided archive containing ``data``
            (train/test splits) and any metadata files.
        output_dir: Destination directory where the dataset should be
            staged for preprocessing and training.
    """

    if not archive_dir.exists():
        raise FileNotFoundError(
            f"Archive directory not found: {archive_dir}. "
            "Ensure the archive folder is available before running this script."
        )

    source_data = archive_dir / "data"
    if not source_data.exists():
        raise FileNotFoundError(
            f"Expected data folder not found in archive: {source_data}. "
            "Verify the archive structure and try again."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the entire data directory (train/test splits). Python 3.8+
    # provides dirs_exist_ok for idempotent sync behavior.
    shutil.copytree(source_data, output_dir / "data", dirs_exist_ok=True)

    summary_file = archive_dir / "Summary.csv"
    if summary_file.exists():
        shutil.copy2(summary_file, output_dir / "Summary.csv")

    print(f"Archived dataset copied to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage the provided archive data for preprocessing and model training. "
            "No web requests are performed."
        )
    )
    default_archive = Path(__file__).resolve().parent.parent / "archive"
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=default_archive,
        help=(
            "Path to the archive folder containing the prepared data. "
            "Defaults to the repository's archive directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prepared_data",
        help=(
            "Directory where the archive contents should be copied. "
            "Use this path when invoking preprocessing scripts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_archive(args.archive_dir, args.output_dir)


if __name__ == "__main__":
    main()

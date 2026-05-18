"""Build the COM3032 / COMM074 submission zip per the Assessment Brief.

Per the brief (page 9, Code 20% row): "Submitted as a single zip file containing
the all Python Notebooks." This script bundles every notebook with its outputs,
the shared `src/` helpers, the dataset, the processed CSVs, the results CSVs,
the figures, the requirements.txt, the README and RUN_GUIDE, and the report
section drafts into a clean zip ready for Surrey Learn upload.

The script also validates the bundle before zipping: every notebook must have
outputs embedded, every results CSV must exist, and the locked-schema columns
must be present in results_<name>.csv files.

Usage from the project root:
    python make_submission.py                # validate then build zip
    python make_submission.py --check        # validate only, do not build
    python make_submission.py --no-data      # skip the 150 MB raw CSV
    python make_submission.py --output FOO.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Files and directories to include in the submission.
INCLUDE_PATTERNS = [
    # Notebooks (mandatory per brief)
    "notebooks/group/01_Group_Preprocessing.ipynb",
    "notebooks/group/02_Group_Visualisation_Recommendations.ipynb",
    "notebooks/individual/*/Modelling_*.ipynb",
    # Shared code (Code 20% modularity)
    "src/__init__.py",
    "src/config.py",
    "src/evaluation.py",
    "src/plotting.py",
    "requirements.txt",
    # Documentation
    "README.md",
    "RUN_GUIDE.md",
    # Outputs (Code 20% evidence, also referenced from the report)
    "outputs/results/*.csv",
    "outputs/figures/*.png",
    # Processed data (so the marker can rerun individual notebooks without
    # rerunning the 5-minute preprocessing step)
    "data/processed/train_processed.csv",
    "data/processed/val_processed.csv",
    "data/processed/test_processed.csv",
    # Report section drafts (transitional — the final submission also wants the
    # combined PDF report, which is built outside this script)
    "reports/sections/*.md",
]

# Optionally include raw CSV (large, but lets the marker reproduce preprocessing).
RAW_DATA_PATTERN = "data/raw/creditcard.csv"

# Required locked-schema columns for any results_<name>.csv to be considered
# rubric-compliant. Legacy CSVs (e.g. Enisa's) flag a warning but are still
# included in the bundle.
LOCKED_SCHEMA_COLUMNS = [
    "member", "model", "split", "imbalance_strategy", "threshold",
    "pr_auc", "roc_auc", "recall_at_90p",
    "f1", "precision", "recall", "accuracy",
    "tp", "fp", "fn", "tn",
]

# Files to always exclude even if matched by an include pattern.
EXCLUDE_NAMES = {".gitkeep", ".DS_Store", ".ipynb_checkpoints"}


def collect_files() -> list[Path]:
    """Resolve every include pattern to a flat sorted list of files."""
    found: set[Path] = set()
    for pattern in INCLUDE_PATTERNS:
        for match in REPO_ROOT.glob(pattern):
            if match.is_file() and match.name not in EXCLUDE_NAMES:
                if any(part in EXCLUDE_NAMES for part in match.parts):
                    continue
                found.add(match)
    return sorted(found)


def validate_notebooks(files: list[Path]) -> list[str]:
    """Each notebook should have at least one code cell with outputs."""
    warnings: list[str] = []
    for path in files:
        if path.suffix != ".ipynb":
            continue
        try:
            nb = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"  Notebook unreadable: {path.relative_to(REPO_ROOT)} ({exc})")
            continue
        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        cells_with_outputs = [c for c in code_cells if c.get("outputs")]
        if not code_cells:
            warnings.append(f"  Notebook has no code cells: {path.relative_to(REPO_ROOT)}")
        elif not cells_with_outputs:
            warnings.append(
                f"  Notebook has no executed outputs: {path.relative_to(REPO_ROOT)} "
                f"(run jupyter nbconvert --to notebook --execute --inplace)"
            )
    return warnings


def validate_results_schemas(files: list[Path]) -> list[str]:
    """Check each results_<name>.csv matches the locked schema columns."""
    warnings: list[str] = []
    for path in files:
        name = path.name
        if not (name.startswith("results_") and name.endswith(".csv")):
            continue
        try:
            header = path.read_text().splitlines()[0]
        except (OSError, IndexError) as exc:
            warnings.append(f"  Results CSV unreadable: {path.relative_to(REPO_ROOT)} ({exc})")
            continue
        cols = [c.strip() for c in header.split(",")]
        missing = [c for c in LOCKED_SCHEMA_COLUMNS if c not in cols]
        if missing:
            warnings.append(
                f"  Results CSV schema mismatch: {path.relative_to(REPO_ROOT)} "
                f"(missing {missing}) — legacy schema, will still be bundled"
            )
    return warnings


def validate_expected_members(files: list[Path]) -> list[str]:
    """Flag missing per-member artefacts so the user can chase teammates."""
    warnings: list[str] = []
    expected_members = ["shom", "abbas", "enisa", "shehab", "alsihamat", "zack"]
    results_files = {p.name for p in files if p.name.startswith("results_") and p.name.endswith(".csv")}
    for m in expected_members:
        expected = f"results_{m}.csv"
        if expected not in results_files:
            warnings.append(f"  Missing: outputs/results/{expected}")
    notebook_dirs = {
        p.parent.name
        for p in files
        if "notebooks/individual/" in str(p) and p.suffix == ".ipynb"
    }
    for m in expected_members:
        if m not in notebook_dirs:
            warnings.append(f"  Missing: notebooks/individual/{m}/Modelling_*.ipynb")
    return warnings


def build_zip(files: list[Path], output_path: Path) -> int:
    """Create the submission zip. Returns the number of files written."""
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(REPO_ROOT)
            zf.write(path, arcname=arcname)
    return len(files)


def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run validation only; do not build the zip.",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Skip the raw 150 MB CSV. Processed CSVs are still included.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zip path (default: submission_group12_<timestamp>.zip).",
    )
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = REPO_ROOT / f"submission_group12_{timestamp}.zip"

    files = collect_files()
    if not args.no_data:
        raw = REPO_ROOT / RAW_DATA_PATTERN
        if raw.is_file():
            files.append(raw)
            files.sort()

    print(f"Collected {len(files)} files for submission:")
    for path in files:
        print(f"  {path.relative_to(REPO_ROOT)}  ({human_size(path.stat().st_size)})")

    print()
    print("=== Validation ===")
    nb_warnings = validate_notebooks(files)
    schema_warnings = validate_results_schemas(files)
    member_warnings = validate_expected_members(files)

    if nb_warnings:
        print("Notebook issues:")
        for w in nb_warnings:
            print(w)
    else:
        print("All notebooks have embedded outputs: OK")

    if schema_warnings:
        print("Results CSV schema notes:")
        for w in schema_warnings:
            print(w)
    else:
        print("All results CSVs match the locked schema: OK")

    if member_warnings:
        print("Missing per-member artefacts (chase teammates):")
        for w in member_warnings:
            print(w)
    else:
        print("All six members have individual notebook + results CSV: OK")

    if args.check:
        print("\n--check specified; not building zip.")
        return 0

    print()
    print(f"=== Building {args.output.name} ===")
    n = build_zip(files, args.output)
    size = args.output.stat().st_size
    try:
        display_path = args.output.relative_to(REPO_ROOT)
    except ValueError:
        display_path = args.output
    print(f"Wrote {n} files to {display_path} ({human_size(size)}).")
    print()
    print("Submission ready. Upload this zip via Surrey Learn alongside the PDF report and dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

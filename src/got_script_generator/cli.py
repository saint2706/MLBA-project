"""Command-line helpers for the Game of Thrones script generator."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Iterable


PROJECT_MODULES: tuple[str, ...] = (
    "got_script_generator.main_modern",
    "got_script_generator.modern_example_usage",
    "got_script_generator.improved_helperAI",
    "got_script_generator.modern_plot",
)


def _ensure_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {path}. Please provide the path to"
            " Game_of_Thrones_Script.csv."
        )


def analyze_data(data_path: Path) -> dict[str, object]:
    """Return basic statistics about the dialogue dataset."""

    import pandas as pd  # Imported lazily to keep CLI responsive without pandas.

    _ensure_dataset(data_path)
    df = pd.read_csv(data_path)
    total_rows = len(df)
    columns = list(df.columns)
    sample = df.head(5).to_dict(orient="records")

    character_column = None
    dialogue_column = None
    for column in columns:
        lower_column = column.lower()
        if "character" in lower_column or "name" in lower_column:
            character_column = column
        if "dialogue" in lower_column or "sentence" in lower_column:
            dialogue_column = column

    unique_characters: int | None = None
    if character_column:
        unique_characters = df[character_column].nunique()

    return {
        "rows": total_rows,
        "columns": columns,
        "character_column": character_column,
        "dialogue_column": dialogue_column,
        "unique_characters": unique_characters,
        "sample_rows": sample,
    }


def smoke_test(modules: Iterable[str], data_path: Path | None = None) -> dict[str, object]:
    """Perform a lightweight project readiness check."""

    status = {
        module: importlib.util.find_spec(module) is not None for module in modules
    }

    dataset_status: dict[str, object] | None = None
    if data_path is not None:
        dataset_path = Path(data_path)
        dataset_status = {
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
        }
        if dataset_path.exists():
            dataset_status["size_bytes"] = dataset_path.stat().st_size

    return {
        "modules": status,
        "dataset": dataset_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for the Game of Thrones AI script generator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze-data", help="Print dataset statistics in JSON format"
    )
    analyze_parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Game_of_Thrones_Script.csv"),
        help="Path to the Game of Thrones dialogue dataset (CSV)",
    )

    smoke_parser = subparsers.add_parser(
        "smoke-test", help="Check module availability and dataset presence"
    )
    smoke_parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional path to dataset for existence check",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze-data":
        result = analyze_data(args.data)
    elif args.command == "smoke-test":
        result = smoke_test(PROJECT_MODULES, args.data)
    else:  # pragma: no cover - argparse enforces command choices
        parser.error(f"Unknown command: {args.command}")
        return

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

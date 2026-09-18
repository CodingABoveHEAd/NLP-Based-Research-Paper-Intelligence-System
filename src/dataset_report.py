"""Create a data-quality report for the merged research-paper dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Title",
    "Author",
    "Institution",
    "Topic",
    "Domain",
    "Field",
    "Concept",
    "Abstract",
    "Keyword",
    "Citation count",
    "Cites",
    "Top 1% cited",
    "Top 10% cited",
    "DOI",
]


def is_empty(value: object) -> bool:
    """Treat nulls and whitespace-only strings as empty values."""
    return pd.isna(value) or (isinstance(value, str) and not value.strip())


def normalise_title(value: object) -> str:
    """Create a comparison key for duplicate-title detection."""
    if is_empty(value):
        return ""
    return re.sub(r"\W+", " ", str(value).casefold(), flags=re.UNICODE).strip()


def format_count(value: int) -> str:
    return f"{value:,}"


def build_report(dataframe: pd.DataFrame, source: Path) -> str:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    unexpected_columns = [column for column in dataframe.columns if column not in REQUIRED_COLUMNS]

    empty_counts = {
        column: int(dataframe[column].map(is_empty).sum())
        for column in dataframe.columns
    }
    duplicate_rows = int(dataframe.duplicated(keep=False).sum())
    title_keys = dataframe["Title"].map(normalise_title) if "Title" in dataframe else pd.Series(dtype="string")
    duplicate_title_rows = int(title_keys[title_keys.ne("")].duplicated(keep=False).sum())
    duplicate_title_groups = int(
        title_keys[title_keys.ne("")].value_counts().gt(1).sum()
    )

    abstract_lengths = (
        dataframe["Abstract"].map(lambda value: len(str(value).strip()) if not is_empty(value) else pd.NA).dropna()
        if "Abstract" in dataframe
        else pd.Series(dtype="int64")
    )

    lines = [
        "# Dataset Report",
        "",
        f"Source: `{source.as_posix()}`",
        "",
        "## Overview",
        "",
        f"- Papers: {format_count(len(dataframe))}",
        f"- Columns: {len(dataframe.columns)}",
        f"- Required columns present: {'Yes' if not missing_columns else 'No'}",
        f"- Unique topics: {format_count(dataframe['Topic'].where(~dataframe['Topic'].map(is_empty)).nunique()) if 'Topic' in dataframe else 'N/A'}",
        f"- Unique domains: {format_count(dataframe['Domain'].where(~dataframe['Domain'].map(is_empty)).nunique()) if 'Domain' in dataframe else 'N/A'}",
        f"- Unique fields: {format_count(dataframe['Field'].where(~dataframe['Field'].map(is_empty)).nunique()) if 'Field' in dataframe else 'N/A'}",
        "",
        "## Schema",
        "",
        f"- Missing required columns: {', '.join(missing_columns) if missing_columns else 'None'}",
        f"- Unexpected columns: {', '.join(unexpected_columns) if unexpected_columns else 'None'}",
        "",
        "## Empty Values",
        "",
        "| Column | Empty values | Percent |",
        "|---|---:|---:|",
    ]
    for column in REQUIRED_COLUMNS:
        count = empty_counts.get(column, len(dataframe))
        percent = (count / len(dataframe) * 100) if len(dataframe) else 0
        lines.append(f"| {column} | {format_count(count)} | {percent:.2f}% |")

    lines.extend(
        [
            "",
            "## Duplicates",
            "",
            f"- Duplicate full rows: {format_count(duplicate_rows)} rows",
            f"- Duplicate normalized titles: {format_count(duplicate_title_rows)} rows in {format_count(duplicate_title_groups)} title groups",
            "",
            "## Abstract Length",
            "",
        ]
    )
    if len(abstract_lengths):
        lines.extend(
            [
                f"- Non-empty abstracts measured: {format_count(len(abstract_lengths))}",
                f"- Average characters (non-empty only): {abstract_lengths.mean():,.2f}",
                f"- Median characters: {abstract_lengths.median():,.2f}",
                f"- Shortest characters: {int(abstract_lengths.min()):,}",
                f"- Longest characters: {int(abstract_lengths.max()):,}",
            ]
        )
    else:
        lines.append("- No Abstract column available.")

    for column in ("Topic", "Domain", "Field"):
        if column not in dataframe:
            continue
        values = dataframe[column].where(~dataframe[column].map(is_empty), "[empty]").fillna("[empty]")
        distribution = values.value_counts(dropna=False)
        lines.extend(["", f"## {column} Distribution", "", "| Value | Papers | Percent |", "|---|---:|---:|"])
        for value, count in distribution.items():
            escaped_value = str(value).replace("|", "\\|")
            percent = count / len(dataframe) * 100 if len(dataframe) else 0
            lines.append(f"| {escaped_value} | {format_count(int(count))} | {percent:.2f}% |")

    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "processed_data" / "merged_research_papers.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "results" / "dataset_report.md",
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(dataframe, args.input), encoding="utf-8")
    print(f"Dataset report written to: {args.output}")


if __name__ == "__main__":
    main()
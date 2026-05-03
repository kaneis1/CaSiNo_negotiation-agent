#!/usr/bin/env python3
"""Validate the paper-facing artifact overlay.

The manifest is stored as JSON-compatible YAML so this checker only needs the
Python standard library. YAML parsers can still read the file as YAML.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper_artifacts" / "manifest.yml"

ALLOWED_ARTIFACT_TYPES = {
    "data_file",
    "result_directory",
    "summary_json",
    "csv",
    "markdown",
    "figure",
    "table_source",
    "script",
    "lsf_log",
    "model_adapter",
    "external_reference",
}

ALLOWED_STATUSES = {
    "existing",
    "symlinked",
    "external",
    "not_found",
    "manual_paper_only",
}

SOURCE_OPTIONAL_STATUSES = {"external", "not_found", "manual_paper_only"}
FORBIDDEN_ALIAS_PARTS = ("day9", "day10", "day11", "neurips2026")
REQUIRED_FIELDS = {
    "paper_section",
    "paper_item",
    "claim",
    "alias_path",
    "source_path",
    "artifact_type",
    "generated_by",
    "reproduction_command",
    "status",
    "notes",
}


def repo_path(path: str) -> Path:
    return ROOT / path


def load_manifest() -> dict[str, Any]:
    try:
        return json.loads(MANIFEST.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{MANIFEST} must be JSON-compatible YAML for this checker: {exc}"
        ) from exc


def iter_symlinks(root: Path) -> list[Path]:
    links: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in dirs + files:
            path = current_path / name
            if path.is_symlink():
                links.append(path)
    return links


def main() -> int:
    errors: list[str] = []
    manifest = load_manifest()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("manifest must contain an object at top-level key 'artifacts'")
        artifacts = {}

    alias_paths: set[str] = set()
    for artifact_id, entry in artifacts.items():
        if not isinstance(entry, dict):
            errors.append(f"{artifact_id}: entry must be an object")
            continue

        missing = sorted(REQUIRED_FIELDS - set(entry))
        if missing:
            errors.append(f"{artifact_id}: missing required fields: {', '.join(missing)}")
            continue

        artifact_type = entry["artifact_type"]
        if artifact_type not in ALLOWED_ARTIFACT_TYPES:
            errors.append(f"{artifact_id}: invalid artifact_type {artifact_type!r}")

        status = entry["status"]
        if status not in ALLOWED_STATUSES:
            errors.append(f"{artifact_id}: invalid status {status!r}")

        if not entry["paper_section"]:
            errors.append(f"{artifact_id}: paper_section must be non-empty")
        if not entry["claim"]:
            errors.append(f"{artifact_id}: claim must be non-empty")
        if not isinstance(entry["generated_by"], list):
            errors.append(f"{artifact_id}: generated_by must be a list")

        alias_path = entry["alias_path"]
        source_path = entry["source_path"]
        alias_paths.add(alias_path)

        alias_lower = alias_path.lower()
        bad_parts = [part for part in FORBIDDEN_ALIAS_PARTS if part in alias_lower]
        if bad_parts:
            errors.append(
                f"{artifact_id}: alias_path contains forbidden paper-facing label(s): "
                + ", ".join(bad_parts)
            )

        if not repo_path(alias_path).exists():
            errors.append(f"{artifact_id}: alias_path does not exist: {alias_path}")

        if status not in SOURCE_OPTIONAL_STATUSES and not repo_path(source_path).exists():
            errors.append(f"{artifact_id}: source_path does not exist: {source_path}")

    figure_paths = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "paper_artifacts" / "figures").glob("figure_*")
    }
    table_paths = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "paper_artifacts" / "tables").glob("table_*")
    }
    for path in sorted(figure_paths | table_paths):
        if path not in alias_paths:
            errors.append(f"{path}: missing manifest entry")

    for link in iter_symlinks(ROOT / "paper_artifacts"):
        if not link.exists():
            errors.append(f"broken symlink: {link.relative_to(ROOT)}")

    if errors:
        print("Paper artifact manifest validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"OK: {len(artifacts)} manifest artifacts validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

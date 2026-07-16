#!/usr/bin/env python3
"""Apply explicit manual painted-line decisions to evidence records."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


ENTRY_KEYS = {"view_path", "camera", "target_id", "decision"}
DECISIONS = {"verified", "rejected", "unreviewed"}


class ReviewError(ValueError):
    pass


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"unable to read {path}: {exc}") from exc


def _resolve_view_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ReviewError("view_path must be a non-empty relative path")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReviewError(f"view_path escapes dataset root: {relative}") from exc
    if candidate.suffix != ".json" or not candidate.is_file():
        raise ReviewError(f"view_path is not an evidence JSON file: {relative}")
    return candidate


def _find_target(record, camera_name, target_id):
    cameras = record.get("cameras")
    if not isinstance(cameras, dict) or camera_name not in cameras:
        raise ReviewError(f"camera {camera_name} is missing")
    targets = cameras[camera_name].get("stop_targets")
    if not isinstance(targets, list):
        raise ReviewError(f"camera {camera_name} has no stop_targets list")
    matches = [
        target
        for target in targets
        if isinstance(target, dict) and target.get("target_id") == target_id
    ]
    if len(matches) != 1:
        raise ReviewError(f"target_id {target_id} must resolve exactly once")
    painted = matches[0].get("painted_line")
    if not isinstance(painted, dict):
        raise ReviewError(f"target_id {target_id} has no painted_line object")
    return painted


def _write_atomic_records(records):
    temporaries = []
    try:
        for path, record in records.items():
            temporary = path.with_suffix(".json.tmp")
            with open(temporary, "w", encoding="utf-8") as file_handle:
                json.dump(record, file_handle, indent=2, sort_keys=True)
                file_handle.write("\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())
            temporaries.append((temporary, path))
        for temporary, path in temporaries:
            os.replace(temporary, path)
    except OSError as exc:
        for temporary, _path in temporaries:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise ReviewError(f"unable to write reviewed evidence: {exc}") from exc


def apply_reviews(root, manifest_path):
    """Validate the complete manifest, then atomically replace changed files."""
    root = Path(root)
    entries = _read_json(Path(manifest_path))
    if not isinstance(entries, list):
        raise ReviewError("review manifest must be a JSON list")

    records = {}
    operations = []
    seen = set()
    counts = Counter()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise ReviewError(
                f"manifest entry {index} must contain exactly {sorted(ENTRY_KEYS)}"
            )
        if entry["decision"] not in DECISIONS:
            raise ReviewError(f"manifest entry {index} has invalid decision")
        if entry["camera"] not in {"front", "left", "right"}:
            raise ReviewError(f"manifest entry {index} has invalid camera")
        key = (entry["view_path"], entry["camera"], entry["target_id"])
        if key in seen:
            raise ReviewError(f"manifest entry {index} duplicates a target decision")
        seen.add(key)

        path = _resolve_view_path(root, entry["view_path"])
        if path not in records:
            records[path] = _read_json(path)
        record = records[path]
        painted = _find_target(record, entry["camera"], entry["target_id"])
        decision = entry["decision"]
        if decision != "unreviewed" and painted.get("status") != "candidate":
            raise ReviewError(
                f"target_id {entry['target_id']} decision requires an existing candidate"
            )
        operations.append((path, painted, decision))
        counts[decision] += 1

    changed_paths = set()
    for path, painted, decision in operations:
        if decision == "unreviewed":
            continue
        painted["status"] = "verified" if decision == "verified" else "unknown"
        painted["review_source"] = "manual_manifest"
        painted["review_decision"] = decision
        changed_paths.add(path)
    _write_atomic_records(
        {path: records[path] for path in changed_paths}
    )
    return {
        "entries": len(entries),
        "verified": counts["verified"],
        "rejected": counts["rejected"],
        "unreviewed": counts["unreviewed"],
        "files_changed": len(changed_paths),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        summary = apply_reviews(args.root, args.manifest)
    except ReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

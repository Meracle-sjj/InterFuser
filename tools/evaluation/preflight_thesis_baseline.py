#!/usr/bin/env python3
"""
[INPUT]: 依赖 configs/thesis/baseline_eval_v1.json、其引用的 checkpoint/路线/场景/agent 文件，以及当前 Git 提交关系。
[OUTPUT]: 对外提供 PreflightError、preflight_baseline 与 CLI，输出哈希、路线分区、场景覆盖和代码锚点的 P0 报告。
[POS]: tools/evaluation 的 M0 启动门禁，在任何 CARLA 批量评测前阻止输入漂移、路线遗漏和空场景配置进入昂贵运行。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CONFIG_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "thesis" / "baseline_eval_v1.json"


class PreflightError(ValueError):
    """Raised when the preflight configuration itself cannot be interpreted."""


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"unable to read JSON {path}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(repo_root, relative, field, errors):
    if not isinstance(relative, str) or not relative:
        errors.append(f"{field}.path must be a non-empty string")
        return None
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        errors.append(f"{field}.path escapes repository root: {relative}")
        return None
    if not candidate.is_file():
        errors.append(f"{field}.path is not a file: {relative}")
        return None
    return candidate


def _verify_file(field, spec, repo_root, errors):
    if not isinstance(spec, dict):
        errors.append(f"{field} must be an object")
        return None
    path = _resolve_repo_path(repo_root, spec.get("path"), field, errors)
    expected = spec.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"{field}.sha256 must be a 64-character digest")
        return path
    if path is not None:
        actual = _sha256(path)
        if actual != expected:
            errors.append(f"{field} sha256 mismatch: expected={expected} actual={actual}")
    return path


def _parse_routes(path, errors):
    if path is None:
        return {}
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        errors.append(f"unable to parse routes XML: {exc}")
        return {}
    routes = {}
    for route in root.findall("route"):
        raw_id = route.attrib.get("id")
        town = route.attrib.get("town")
        try:
            route_id = int(raw_id)
        except (TypeError, ValueError):
            errors.append(f"route id is not an integer: {raw_id}")
            continue
        if route_id in routes:
            errors.append(f"duplicate route id: {route_id}")
            continue
        if not isinstance(town, str) or not town:
            errors.append(f"route {route_id} has no town")
            continue
        if len(route.findall("waypoint")) < 2:
            errors.append(f"route {route_id} has fewer than two waypoints")
        routes[route_id] = town
    if not routes:
        errors.append("routes XML contains no usable routes")
    return routes


def _scenario_events(path, errors):
    if path is None:
        return Counter()
    data = _read_json(path)
    available = data.get("available_scenarios") if isinstance(data, dict) else None
    if not isinstance(available, list):
        errors.append("scenario JSON must contain available_scenarios list")
        return Counter()
    events = Counter()
    for town_block in available:
        if not isinstance(town_block, dict):
            errors.append("each available_scenarios entry must be an object")
            continue
        for town, groups in town_block.items():
            if not isinstance(groups, list):
                errors.append(f"scenario groups for {town} must be a list")
                continue
            for group in groups:
                if not isinstance(group, dict):
                    errors.append(f"scenario group for {town} must be an object")
                    continue
                configurations = group.get("available_event_configurations", [])
                if not isinstance(configurations, list):
                    errors.append(
                        f"available_event_configurations for {town} must be a list"
                    )
                    continue
                events[town] += len(configurations)
    return events


def _int_set(value, field, errors):
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return set()
    result = set()
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            errors.append(f"{field} must contain only integers")
            continue
        if item in result:
            errors.append(f"{field} contains duplicate route {item}")
        result.add(item)
    return result


def _string_set(value, field, errors):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        errors.append(f"{field} must be a list of non-empty strings")
        return set()
    if len(value) != len(set(value)):
        errors.append(f"{field} contains duplicates")
    return set(value)


def _check_git_anchor(repo_root, anchor, runtime_roots, errors):
    if not isinstance(anchor, str) or not anchor:
        errors.append("code_anchor must be a non-empty commit id")
        return None
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", anchor, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or "anchor is not an ancestor of HEAD"
        errors.append(f"code_anchor check failed: {details}")
        return None
    if not isinstance(runtime_roots, list) or not runtime_roots or any(
        not isinstance(item, str)
        or not item
        or Path(item).is_absolute()
        or ".." in Path(item).parts
        for item in runtime_roots
    ):
        errors.append("runtime_code_roots must be non-empty repository-relative paths")
        return None
    runtime_diff = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--quiet",
            anchor,
            "--",
            *runtime_roots,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if runtime_diff.returncode != 0:
        details = runtime_diff.stderr.strip()
        message = "runtime source differs from code_anchor"
        if details:
            message += f": {details}"
        errors.append(message)
        return None
    runtime_status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--", *runtime_roots],
        capture_output=True,
        text=True,
        check=False,
    )
    if runtime_status.returncode != 0 or runtime_status.stdout.strip():
        details = runtime_status.stderr.strip() or runtime_status.stdout.strip()
        errors.append(f"runtime source worktree is not clean: {details}")
        return None
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        errors.append(f"unable to read Git HEAD: {head.stderr.strip()}")
        return None
    return head.stdout.strip()


def preflight_baseline(config_path=DEFAULT_CONFIG, repo_root=REPO_ROOT, check_git=True):
    """Validate every static input needed before starting a thesis baseline run."""
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise PreflightError(f"repository root is not a directory: {repo_root}")
    config_path = Path(config_path)
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise PreflightError("baseline config must be a JSON object")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise PreflightError(
            f"unsupported baseline config schema_version: {config.get('schema_version')}"
        )

    errors = []
    warnings = []
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise PreflightError("inputs must be an object")
    required_inputs = {"routes", "scenarios", "agent", "agent_config"}
    missing_inputs = sorted(required_inputs - set(inputs))
    if missing_inputs:
        errors.append(f"missing required inputs: {missing_inputs}")
    verified_paths = {
        name: _verify_file(f"inputs.{name}", spec, repo_root, errors)
        for name, spec in sorted(inputs.items())
    }

    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise PreflightError("checkpoint must be an object")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_absolute() or not checkpoint_path.is_file():
        errors.append(f"checkpoint path is not an absolute file: {checkpoint_path}")
        checkpoint_actual = None
    else:
        checkpoint_actual = _sha256(checkpoint_path)
        if checkpoint_actual != checkpoint.get("sha256"):
            errors.append(
                "checkpoint sha256 mismatch: "
                f"expected={checkpoint.get('sha256')} actual={checkpoint_actual}"
            )

    routes = _parse_routes(verified_paths.get("routes"), errors)
    scenario_events = _scenario_events(verified_paths.get("scenarios"), errors)
    route_sets = config.get("route_sets")
    if not isinstance(route_sets, dict):
        raise PreflightError("route_sets must be an object")
    development = _int_set(route_sets.get("development_d7"), "development_d7", errors)
    primary = _int_set(route_sets.get("primary_a36"), "primary_a36", errors)
    excluded = _int_set(
        route_sets.get("excluded_until_map_install"),
        "excluded_until_map_install",
        errors,
    )
    route_ids = set(routes)
    if development - primary:
        errors.append(
            f"development_d7 is not a subset of primary_a36: {sorted(development - primary)}"
        )
    if primary & excluded:
        errors.append(f"primary and excluded route sets overlap: {sorted(primary & excluded)}")
    if primary | excluded != route_ids:
        errors.append(
            "primary/excluded route partition mismatch: "
            f"missing={sorted(route_ids - (primary | excluded))} "
            f"extra={sorted((primary | excluded) - route_ids)}"
        )

    map_policy = config.get("map_policy")
    if not isinstance(map_policy, dict):
        raise PreflightError("map_policy must be an object")
    available_towns = _string_set(
        map_policy.get("available_route_towns"), "available_route_towns", errors
    )
    unavailable_towns = _string_set(
        map_policy.get("unavailable_route_towns"), "unavailable_route_towns", errors
    )
    if available_towns & unavailable_towns:
        errors.append(
            f"available and unavailable towns overlap: {sorted(available_towns & unavailable_towns)}"
        )
    for route_id, town in routes.items():
        if route_id in primary and town not in available_towns:
            errors.append(f"primary route {route_id} uses unavailable town {town}")
        if route_id in excluded and town not in unavailable_towns:
            errors.append(f"excluded route {route_id} is not in an unavailable town: {town}")
    for town in sorted(set(routes.values())):
        if scenario_events[town] <= 0:
            errors.append(f"scenario file has no events for route town {town}")

    background = config.get("background_vehicles_by_town")
    if not isinstance(background, dict):
        errors.append("background_vehicles_by_town must be an object")
    else:
        missing_background = sorted(
            {routes[route_id] for route_id in development if routes.get(route_id) not in background}
        )
        if missing_background:
            errors.append(
                f"development route towns missing background counts: {missing_background}"
            )

    seeds = config.get("random_seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
    ):
        errors.append("random_seeds must be a non-empty integer list")
    elif len(seeds) != len(set(seeds)):
        errors.append("random_seeds contains duplicates")

    result_root = config.get("result_root")
    if not isinstance(result_root, str) or not result_root:
        errors.append("result_root must be a non-empty relative path")
    elif Path(result_root).is_absolute() or ".." in Path(result_root).parts:
        errors.append("result_root must stay inside the repository")

    head = None
    if check_git:
        head = _check_git_anchor(
            repo_root,
            config.get("code_anchor"),
            config.get("runtime_code_roots"),
            errors,
        )
    if excluded:
        warnings.append(
            f"{len(excluded)} routes are excluded until their maps are installed; "
            "they must not be counted as zero in A36"
        )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "git_head": head,
        "checkpoint_sha256": checkpoint_actual,
        "route_count": len(routes),
        "routes_by_town": dict(sorted(Counter(routes.values()).items())),
        "scenario_events_by_town": dict(sorted(scenario_events.items())),
        "development_routes": sorted(development),
        "primary_routes": sorted(primary),
        "excluded_routes": sorted(excluded),
        "random_seeds": seeds if isinstance(seeds, list) else [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate thesis baseline inputs before launching CARLA"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = preflight_baseline(args.config, args.repo_root)
    except PreflightError as exc:
        print(f"preflight error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

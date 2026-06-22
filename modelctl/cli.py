from __future__ import annotations

import argparse
import json
import math
import sys

from . import __version__

PRETTY = False

MANIFEST_COMMANDS = {"validate", "preflight", "start", "wait", "stop", "status", "health", "smoke", "soak", "bench", "doctor", "watchdog", "daemon", "report", "cleanup", "service", "rotate", "promote"}
BENCH_PRESETS = {
    "tiny": [128],
    "small": [128, 512, 1024],
    "standard": [128, 512, 1024, 2048, 4096],
}


def _pretty_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _pretty_lines(obj, indent: int = 0) -> list[str]:
    pad = "  " * indent
    if isinstance(obj, dict):
        lines: list[str] = []
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(_pretty_lines(value, indent + 1))
            elif isinstance(value, list):
                if not value:
                    lines.append(f"{pad}{key}: []")
                else:
                    lines.append(f"{pad}{key}:")
                    for item in value:
                        if isinstance(item, (dict, list)):
                            lines.append(f"{pad}  -")
                            lines.extend(_pretty_lines(item, indent + 2))
                        else:
                            lines.append(f"{pad}  - {_pretty_value(item)}")
            else:
                lines.append(f"{pad}{key}: {_pretty_value(value)}")
        return lines
    return [f"{pad}{_pretty_value(obj)}"]


def emit(obj) -> None:
    if PRETTY:
        print("\n".join(_pretty_lines(obj)))
    else:
        print(json.dumps(obj, indent=2, sort_keys=True))


def parse_int_list(value: str) -> list[int]:
    try:
        items = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not items:
        raise argparse.ArgumentTypeError("at least one integer is required")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("values must be positive")
    return items


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative finite number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative finite number")
    return parsed


def add_registry_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_registry = sub.add_parser("registry", help="Manage the manifest registry")
    reg = p_registry.add_subparsers(dest="registry_command", required=True)
    p_reg_list = reg.add_parser("list", help="List registered manifests")
    p_reg_list.add_argument("--registry", action="append", default=[], help="Extra registry directory to scan; can be repeated")
    p_reg_add = reg.add_parser("add", help="Copy a manifest into the registry")
    p_reg_add.add_argument("--source", default=None, help="Manifest path; defaults to global --manifest")
    p_reg_add.add_argument("--name", default=None, help="Registry entry name; defaults to manifest id")
    p_reg_add.add_argument("--registry", default=None, help="Registry directory to write to")
    p_reg_add.add_argument("--overwrite", action="store_true")
    p_reg_show = reg.add_parser("show", help="Show one registered manifest")
    p_reg_show.add_argument("name")
    p_reg_show.add_argument("--registry", action="append", default=[])
    p_reg_show.add_argument("--content", action="store_true", help="Include manifest content")
    p_reg_rm = reg.add_parser("remove", help="Remove one registered manifest")
    p_reg_rm.add_argument("name")
    p_reg_rm.add_argument("--registry", default=None)
    p_reg_rm.add_argument("--missing-ok", action="store_true")
    p_reg_use = reg.add_parser("use", help="Copy or symlink a registered manifest to a working path")
    p_reg_use.add_argument("name")
    p_reg_use.add_argument("--output", "-o", default="modelctl.toml")
    p_reg_use.add_argument("--registry", default=None)
    p_reg_use.add_argument("--overwrite", action="store_true")
    p_reg_use.add_argument("--symlink", action="store_true")


def add_reports_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_reports = sub.add_parser("reports", help="Manage saved model reports")
    reps = p_reports.add_subparsers(dest="reports_command", required=True)
    p_save = reps.add_parser("save", help="Save a model report under the modelctl state directory")
    p_save.add_argument("--format", choices=["json", "md"], default="json")
    p_save.add_argument("--include-smoke", action="store_true")
    p_list = reps.add_parser("list", help="List saved reports")
    p_list.add_argument("--model", default=None, help="Filter by manifest id")
    p_show = reps.add_parser("show", help="Show a saved report by report id or path")
    p_show.add_argument("report_id")


def add_fleet_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_fleet = sub.add_parser("fleet", help="Run operations across registered model manifests")
    fleet = p_fleet.add_subparsers(dest="fleet_command", required=True)
    p_status = fleet.add_parser("status", help="Show operator status across registry entries")
    p_status.add_argument("--registry", action="append", default=[], help="Extra registry directory to scan; can be repeated")
    p_status.add_argument("--limit", type=int, default=None, help="Limit number of registry entries checked")
    p_status.add_argument("--jobs", type=positive_int, default=None, help="Maximum concurrent status probes; defaults to a bounded parallel scan")
    p_status.add_argument("--readiness-timeout", type=float, default=1.0, help="Per-model readiness timeout seconds")
    p_health = fleet.add_parser("health", help="Run health checks across registry entries")
    p_health.add_argument("--registry", action="append", default=[], help="Extra registry directory to scan; can be repeated")
    p_health.add_argument("--max-swap-gib", type=float, default=None, help="Absolute swap ceiling for each model")
    p_health.add_argument("--max-swap-delta-gib", type=float, default=None, help="Maximum allowed swap growth across --sample-sec for each model")
    p_health.add_argument("--sample-sec", type=float, default=None, help="Seconds between swap samples per model")
    p_health.add_argument("--smoke", action="store_true", help="Run each manifest smoke test as part of health")
    p_health.add_argument("--max-latency-sec", type=float, default=None, help="Maximum allowed smoke latency when --smoke is used")
    p_health.add_argument("--limit", type=int, default=None, help="Limit number of registry entries checked")
    p_health.add_argument("--jobs", type=positive_int, default=None, help="Maximum concurrent health probes; defaults to a bounded parallel scan")
    p_recover = fleet.add_parser("recover", help="Plan or execute safe recovery for down registered models")
    p_recover.add_argument("--registry", action="append", default=[], help="Extra registry directory to scan; can be repeated")
    p_recover.add_argument("--limit", type=int, default=None, help="Limit number of registry entries checked")
    p_recover.add_argument("--jobs", type=positive_int, default=None, help="Maximum concurrent recovery probes; dry-run defaults to parallel, execute requires serial jobs")
    p_recover.add_argument("--readiness-timeout", type=float, default=1.0, help="Per-model readiness timeout seconds")
    p_recover.add_argument("--execute", action="store_true", help="Actually start recoverable down manifests; requires --wait; dry-run by default")
    p_recover.add_argument("--wait", action="store_true", help="Wait for readiness after starting each model")


def add_mlx_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_mlx = sub.add_parser("mlx", help="MLX model discovery, inspection, overlays, and manifests")
    mlx = p_mlx.add_subparsers(dest="mlx_command", required=True)
    p_discover = mlx.add_parser("discover", help="Find MLX model directories under a root")
    p_discover.add_argument("--root", default="~/.cache/mlx-models", help="Root to scan for config.json files")
    p_discover.add_argument("--limit", type=int, default=200)
    p_inspect = mlx.add_parser("inspect", help="Inspect an MLX model directory for serving hazards")
    p_inspect.add_argument("model_path")
    p_overlay = mlx.add_parser("overlay", help="Create a reversible -served overlay with patched chat_template.jinja")
    p_overlay.add_argument("model_path")
    p_overlay.add_argument("--output", "-o", default=None, help="Overlay directory; defaults to sibling NAME-served")
    p_overlay.add_argument("--overwrite", action="store_true")
    p_manifest = mlx.add_parser("manifest", help="Generate a modelctl.toml for mlx_lm server")
    p_manifest.add_argument("model_path")
    p_manifest.add_argument("--output", "-o", default="modelctl.toml")
    p_manifest.add_argument("--model-id", default=None, help="Request model id for stock mlx_lm; defaults to default_model. Use --id for friendly names.")
    p_manifest.add_argument("--id", dest="ident", default=None)
    p_manifest.add_argument("--port", type=int, default=8080)
    p_manifest.add_argument("--python", default=None, help="Python executable with mlx_lm installed")
    p_manifest.add_argument("--max-tokens", type=int, default=4096)
    p_manifest.add_argument("--temp", type=float, default=0.23)
    p_manifest.add_argument("--top-p", type=float, default=0.9)
    p_manifest.add_argument("--prompt-cache-gib", type=float, default=4.0)
    p_manifest.add_argument("--prompt-cache-size", type=positive_int, default=4)
    p_manifest.add_argument("--overwrite", action="store_true")


def add_service_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_service = sub.add_parser("service", help="Install and control a macOS launchd service for this manifest")
    svc = p_service.add_subparsers(dest="service_command", required=True)
    p_install = svc.add_parser("install", help="Write a LaunchAgent plist that runs modelctl daemon")
    p_install.add_argument("--label", default=None, help="launchd label; defaults to ai.modelctl.<manifest-id>")
    p_install.add_argument("--restart", action="store_true", help="Daemon may restart/start the model on breach; requires [start]")
    p_install.add_argument("--max-swap-gib", type=float, default=None, help="Override manifest preflight max_swap_gib for the daemon")
    p_install.add_argument("--max-swap-delta-gib", type=float, default=None, help="Health-mode maximum allowed swap growth across --sample-sec")
    p_install.add_argument("--sample-sec", type=float, default=None, help="Health-mode seconds between swap samples")
    p_install.add_argument("--health-mode", action="store_true", help="Run the daemon with modelctl health verdicts instead of legacy watchdog samples")
    p_install.add_argument("--smoke", action="store_true", help="Health-mode daemon includes manifest smoke test")
    p_install.add_argument("--max-latency-sec", type=float, default=None, help="Health-mode maximum allowed smoke latency")
    p_install.add_argument("--interval", type=float, default=30.0, help="Daemon sample interval seconds")
    p_install.add_argument("--python", default=None, help="Python executable to run modelctl from; defaults to current interpreter")
    p_install.add_argument("--service-log", default=None, help="LaunchAgent stdout log path")
    p_install.add_argument("--run-at-load", action="store_true", help="Start daemon when launchd loads the plist")
    p_install.add_argument("--no-keepalive", action="store_true", help="Do not ask launchd to keep the daemon alive")
    p_install.add_argument("--no-wait", action="store_true", help="Pass --no-wait to daemon restarts")
    p_install.add_argument("--overwrite", action="store_true")
    p_install.add_argument("--dry-run", action="store_true")
    p_diff = svc.add_parser("diff", help="Compare installed LaunchAgent plist with desired manifest/service options")
    p_diff.add_argument("--label", default=None, help="launchd label; defaults to ai.modelctl.<manifest-id>")
    p_diff.add_argument("--restart", action="store_true", help="Desired daemon may restart/start the model on breach")
    p_diff.add_argument("--max-swap-gib", type=float, default=None, help="Desired absolute swap ceiling")
    p_diff.add_argument("--max-swap-delta-gib", type=float, default=None, help="Desired health-mode maximum allowed swap growth")
    p_diff.add_argument("--sample-sec", type=float, default=None, help="Desired health-mode seconds between swap samples")
    p_diff.add_argument("--health-mode", action="store_true", help="Desired daemon uses modelctl health verdicts")
    p_diff.add_argument("--smoke", action="store_true", help="Desired health-mode daemon includes manifest smoke test")
    p_diff.add_argument("--max-latency-sec", type=float, default=None, help="Desired health-mode maximum allowed smoke latency")
    p_diff.add_argument("--interval", type=float, default=30.0, help="Desired daemon sample interval seconds")
    p_diff.add_argument("--python", default=None, help="Desired Python executable to run modelctl from")
    p_diff.add_argument("--service-log", default=None, help="Desired LaunchAgent stdout log path")
    p_diff.add_argument("--run-at-load", action="store_true", help="Desired RunAtLoad value")
    p_diff.add_argument("--no-keepalive", action="store_true", help="Desired KeepAlive=false")
    p_diff.add_argument("--no-wait", action="store_true", help="Desired daemon restarts pass --no-wait")
    p_diff.add_argument("--content", action="store_true", help="Include full desired and installed plist dictionaries")
    for name in ("start", "stop", "restart", "status", "uninstall"):
        p = svc.add_parser(name, help=f"{name} the launchd service")
        p.add_argument("--label", default=None)
        p.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelctl", description="Manifest-driven lifecycle control for local LLM servers.")
    parser.add_argument("-m", "--manifest", default="modelctl.toml", help="Path to model manifest TOML")
    parser.add_argument("--pretty", action="store_true", help="Print human-readable output instead of JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="Print modelctl version")
    p_init = sub.add_parser("init", help="Write a starter modelctl.toml manifest")
    p_init.add_argument("--output", "-o", default="modelctl.toml")
    p_init.add_argument("--template", choices=["minimal", "llama-cpp"], default="minimal")
    p_init.add_argument("--model-id", default="local-model")
    p_init.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    p_init.add_argument("--id", dest="ident", default=None)
    p_init.add_argument("--port", type=int, default=8080)
    p_init.add_argument("--overwrite", action="store_true")
    sub.add_parser("validate", help="Parse manifest and print resolved summary")
    sub.add_parser("preflight", help="Run required path, port, disk, and swap checks")
    p_list = sub.add_parser("list", help="List manifests in registry directories")
    p_list.add_argument("--registry", action="append", default=[], help="Extra registry directory to scan; can be repeated")
    add_registry_parser(sub)
    add_reports_parser(sub)
    add_fleet_parser(sub)
    add_mlx_parser(sub)
    add_service_parser(sub)
    p_ingest = sub.add_parser("ingest", help="Generate a starter manifest from an OpenAI-compatible /v1 endpoint")
    p_ingest.add_argument("--endpoint", required=True, help="Endpoint base URL, e.g. http://127.0.0.1:8080/v1")
    p_ingest.add_argument("--output", "-o", default=None, help="Write manifest to this path; omit to print manifest JSON payload")
    p_ingest.add_argument("--model-id", default=None, help="Model id to use; defaults to the first /models entry")
    p_ingest.add_argument("--id", dest="ident", default=None, help="Manifest [model].id; defaults to sanitized model id")
    p_ingest.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    p_start = sub.add_parser("start", help="Start configured model server")
    p_start.add_argument("--wait", action="store_true", help="Wait for readiness after start")
    p_wait = sub.add_parser("wait", help="Wait for readiness")
    p_wait.add_argument("--timeout", type=int, default=None, help="Override startup timeout seconds")
    p_stop = sub.add_parser("stop", help="Stop configured model server")
    p_stop.add_argument("--timeout", type=int, default=10, help="Grace period before SIGKILL")
    p_rotate = sub.add_parser("rotate", help="Readiness-gated process rotation to another manifest with atomic PID-state handoff")
    p_rotate.add_argument("--to", required=True, help="Target manifest to start after stopping the current manifest")
    p_rotate.add_argument("--readiness-timeout", type=positive_float, default=None, help="Seconds to wait for the target readiness gate")
    p_rotate.add_argument("--stop-timeout", type=positive_int, default=10, help="Grace period before SIGKILL when stopping old/failed processes")
    p_rotate.add_argument("--no-rollback", action="store_true", help="Do not restart the current manifest if target readiness fails")
    p_rotate.add_argument("--dry-run", action="store_true", help="Print the rotation plan without side effects")
    p_promote = sub.add_parser("promote", help="Plan or execute candidate promotion through preflight, rotate, and post-health gates")
    p_promote.add_argument("--candidate", required=True, help="Candidate manifest to promote onto the current stable endpoint/model")
    p_promote.add_argument("--execute", action="store_true", help="Perform the promotion; default is a no-side-effect plan")
    p_promote.add_argument("--readiness-timeout", type=positive_float, default=None, help="Seconds to wait for candidate readiness")
    p_promote.add_argument("--stop-timeout", type=positive_int, default=10, help="Grace period before SIGKILL when stopping old/failed processes")
    p_promote.add_argument("--no-rollback", action="store_true", help="Do not restart the current manifest if candidate rotation or post-health fails")
    p_promote.add_argument("--max-swap-gib", type=nonnegative_float, default=None, help="Post-promotion absolute swap ceiling")
    p_promote.add_argument("--max-swap-delta-gib", type=nonnegative_float, default=None, help="Post-promotion maximum allowed swap growth across --sample-sec")
    p_promote.add_argument("--sample-sec", type=nonnegative_float, default=None, help="Post-promotion swap sample seconds; 0 takes back-to-back samples")
    p_promote.add_argument("--smoke", action="store_true", help="Run manifest smoke during post-promotion health")
    p_promote.add_argument("--max-latency-sec", type=nonnegative_float, default=None, help="Maximum allowed post-promotion smoke latency")
    sub.add_parser("status", help="Print process/readiness status")
    p_health = sub.add_parser("health", help="Run high-signal readiness, pid, swap, and optional smoke health checks")
    p_health.add_argument("--max-swap-gib", type=float, default=None, help="Absolute swap ceiling; defaults to manifest preflight max_swap_gib")
    p_health.add_argument("--max-swap-delta-gib", type=float, default=None, help="Maximum allowed swap growth across --sample-sec")
    p_health.add_argument("--sample-sec", type=float, default=None, help="Seconds between swap samples; 0 takes back-to-back samples")
    p_health.add_argument("--smoke", action="store_true", help="Run the manifest smoke test as part of health")
    p_health.add_argument("--max-latency-sec", type=float, default=None, help="Maximum allowed smoke latency when --smoke is used")
    p_doctor = sub.add_parser("doctor", help="Run preflight, status, cleanup review, and stale-state diagnostics")
    p_doctor.add_argument("--fix", action="store_true", help="Apply safe local repairs such as stale PID removal and state-dir creation")
    p_report = sub.add_parser("report", help="Write a JSON or markdown model state report")
    p_report.add_argument("--output", "-o", default=None)
    p_report.add_argument("--format", choices=["json", "md"], default="json")
    p_report.add_argument("--include-smoke", action="store_true", help="Run and include smoke test result")
    p_smoke = sub.add_parser("smoke", help="Run OpenAI-compatible chat completion smoke")
    p_smoke.add_argument("--prompt", default=None)
    p_smoke.add_argument("--expect", default=None)
    p_smoke.add_argument("--max-tokens", type=int, default=None)
    p_smoke.add_argument("--temperature", type=float, default=None)
    p_soak = sub.add_parser("soak", help="Run repeated smoke tests with timing and swap sampling")
    p_soak.add_argument("--count", type=int, default=3)
    p_soak.add_argument("--delay", type=float, default=0.0, help="Delay between runs in seconds")
    p_soak.add_argument("--no-fail-fast", action="store_true", help="Continue after a failed run")
    p_bench = sub.add_parser("bench", help="Run synthetic prompt-size benchmarks with timing and swap sampling")
    p_bench.add_argument("--preset", choices=sorted(BENCH_PRESETS), default="small", help="Prompt-size preset; ignored when --prompt-chars is set")
    p_bench.add_argument("--prompt-chars", type=parse_int_list, default=None, help="Comma-separated synthetic prompt sizes in characters")
    p_bench.add_argument("--repeats", type=int, default=1)
    p_bench.add_argument("--max-tokens", type=int, default=16)
    p_bench.add_argument("--output", "-o", default=None, help="Write benchmark artifact to this path")
    p_bench.add_argument("--format", choices=["json", "md"], default="json")
    p_watchdog = sub.add_parser("watchdog", help="Sample readiness/swap and optionally stop the model on breach")
    p_watchdog.add_argument("--max-swap-gib", type=float, default=None)
    p_watchdog.add_argument("--duration", type=float, default=0.0, help="Seconds to watch; 0 means one sample")
    p_watchdog.add_argument("--interval", type=float, default=10.0, help="Seconds between samples")
    p_watchdog.add_argument("--stop-on-breach", action="store_true")
    p_daemon = sub.add_parser("daemon", help="Run a foreground readiness/swap supervisor loop")
    p_daemon.add_argument("--max-swap-gib", type=float, default=None)
    p_daemon.add_argument("--max-swap-delta-gib", type=float, default=None, help="Health-mode maximum allowed swap growth across --sample-sec")
    p_daemon.add_argument("--sample-sec", type=float, default=None, help="Health-mode seconds between swap samples")
    p_daemon.add_argument("--smoke", action="store_true", help="Health-mode: include manifest smoke test")
    p_daemon.add_argument("--max-latency-sec", type=float, default=None, help="Health-mode maximum allowed smoke latency")
    p_daemon.add_argument("--health-mode", action="store_true", help="Use modelctl health verdicts instead of legacy watchdog samples")
    p_daemon.add_argument("--interval", type=float, default=30.0, help="Seconds between supervisor samples")
    p_daemon.add_argument("--iterations", type=int, default=None, help="Stop after N samples; omit to run until interrupted")
    p_daemon.add_argument("--restart", action="store_true", help="On breach, stop/start the configured manifest process. Explicit for a reason.")
    p_daemon.add_argument("--no-wait", action="store_true", help="Do not wait for readiness after an explicit restart")
    p_cleanup = sub.add_parser("cleanup", help="Plan or execute cleanup candidates")
    p_cleanup.add_argument("--execute", action="store_true", help="Actually delete safe cleanup candidates")
    p_cleanup.add_argument("--force", action="store_true", help="Allow deleting unsafe cleanup candidates too")
    return parser


def main(argv: list[str] | None = None) -> int:
    global PRETTY
    parser = build_parser()
    args = parser.parse_args(argv)
    PRETTY = bool(args.pretty)
    try:
        if args.command == "version":
            emit({"version": __version__}); return 0
        if args.command == "init":
            from .init import init_manifest
            result = init_manifest(output=args.output, template=args.template, model_id=args.model_id, endpoint=args.endpoint, ident=args.ident, port=args.port, overwrite=args.overwrite); emit(result); return 0 if result.get("ok") else 2
        if args.command == "list":
            from .registry import list_registry
            emit(list_registry(args.registry)); return 0
        if args.command == "registry":
            from .registry import add_registry, list_registry, remove_registry, show_registry, use_registry
            if args.registry_command == "list":
                emit(list_registry(args.registry)); return 0
            if args.registry_command == "add":
                result = add_registry(args.source or args.manifest, name=args.name, registry_dir=args.registry, overwrite=args.overwrite); emit(result); return 0 if result.get("ok") else 2
            if args.registry_command == "show":
                result = show_registry(args.name, extra_dirs=args.registry, include_content=args.content); emit(result); return 0 if result.get("ok") else 2
            if args.registry_command == "remove":
                result = remove_registry(args.name, registry_dir=args.registry, missing_ok=args.missing_ok); emit(result); return 0 if result.get("ok") else 2
            if args.registry_command == "use":
                result = use_registry(args.name, output=args.output, registry_dir=args.registry, overwrite=args.overwrite, symlink=args.symlink); emit(result); return 0 if result.get("ok") else 2
        if args.command == "reports":
            from .report_store import list_reports, save_report, show_report
            if args.reports_command == "list":
                result = list_reports(model=args.model); emit(result); return 0 if result.get("ok") else 2
            if args.reports_command == "show":
                result = show_report(args.report_id); emit(result); return 0 if result.get("ok") else 2
            if args.reports_command == "save":
                from .manifest import load_manifest
                manifest = load_manifest(args.manifest)
                result = save_report(manifest, fmt=args.format, include_smoke=args.include_smoke); emit(result); return 0 if result.get("ok") else 2
        if args.command == "fleet":
            from .fleet import fleet_health, fleet_recover, fleet_status
            if args.fleet_command == "status":
                result = fleet_status(registries=args.registry, limit=args.limit, readiness_timeout=args.readiness_timeout, jobs=args.jobs); emit(result); return 0
            if args.fleet_command == "health":
                result = fleet_health(registries=args.registry, max_swap_gib=args.max_swap_gib, max_swap_delta_gib=args.max_swap_delta_gib, sample_sec=args.sample_sec, include_smoke=args.smoke, max_latency_sec=args.max_latency_sec, limit=args.limit, jobs=args.jobs); emit(result); return 0 if result.get("ok") else 2
            if args.fleet_command == "recover":
                result = fleet_recover(registries=args.registry, limit=args.limit, readiness_timeout=args.readiness_timeout, execute=args.execute, wait=args.wait, jobs=args.jobs); emit(result); return 0 if result.get("ok") else 2
        if args.command == "mlx":
            from .mlx import create_overlay, discover_mlx_models, inspect_mlx_model, write_mlx_manifest
            if args.mlx_command == "discover":
                result = discover_mlx_models(root=args.root, limit=args.limit); emit(result); return 0 if result.get("ok") else 2
            if args.mlx_command == "inspect":
                result = inspect_mlx_model(args.model_path); emit(result); return 0 if result.get("ok") else 2
            if args.mlx_command == "overlay":
                result = create_overlay(args.model_path, output=args.output, overwrite=args.overwrite); emit(result); return 0 if result.get("ok") else 2
            if args.mlx_command == "manifest":
                result = write_mlx_manifest(args.model_path, output=args.output, overwrite=args.overwrite, model_id=args.model_id, ident=args.ident, port=args.port, python=args.python, max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p, prompt_cache_gib=args.prompt_cache_gib, prompt_cache_size=args.prompt_cache_size); emit(result); return 0 if result.get("ok") else 2
        if args.command == "ingest":
            from .ingest import ingest
            result = ingest(args.endpoint, output=args.output, model_id=args.model_id, ident=args.ident, overwrite=args.overwrite); emit(result); return 0 if result.get("ok") else 2
        if args.command not in MANIFEST_COMMANDS:
            parser.error("unknown command")
        from .manifest import load_manifest
        manifest = load_manifest(args.manifest)
        from .ops import bench, cleanup_execute, cleanup_plan, daemon, doctor, doctor_fix, health, preflight, smoke, soak, status, validate, watchdog
        from .runner import rotate, start, stop, wait_ready
        if args.command == "validate":
            emit(validate(manifest)); return 0
        if args.command == "preflight":
            result = preflight(manifest); emit(result); return 0 if result.get("ok") else 2
        if args.command == "start":
            emit(start(manifest, wait=args.wait)); return 0
        if args.command == "wait":
            result = wait_ready(manifest, timeout_sec=args.timeout); emit(result); return 0 if result.get("ready") else 2
        if args.command == "stop":
            emit(stop(manifest, timeout_sec=args.timeout)); return 0
        if args.command == "rotate":
            target = load_manifest(args.to)
            result = rotate(manifest, target, readiness_timeout_sec=args.readiness_timeout, stop_timeout_sec=args.stop_timeout, rollback=not args.no_rollback, dry_run=args.dry_run); emit(result); return 0 if result.get("ok") else 2
        if args.command == "promote":
            from .promote import promote
            candidate = load_manifest(args.candidate)
            result = promote(manifest, candidate, execute=args.execute, readiness_timeout_sec=args.readiness_timeout, stop_timeout_sec=args.stop_timeout, rollback=not args.no_rollback, max_swap_gib=args.max_swap_gib, max_swap_delta_gib=args.max_swap_delta_gib, sample_sec=args.sample_sec, include_smoke=args.smoke, max_latency_sec=args.max_latency_sec); emit(result); return 0 if result.get("ok") else 2
        if args.command == "status":
            emit(status(manifest)); return 0
        if args.command == "health":
            result = health(manifest, max_swap_gib=args.max_swap_gib, max_swap_delta_gib=args.max_swap_delta_gib, sample_sec=args.sample_sec, include_smoke=args.smoke, max_latency_sec=args.max_latency_sec); emit(result); return 0 if result.get("ok") else 2
        if args.command == "doctor":
            result = doctor_fix(manifest) if args.fix else doctor(manifest); emit(result); return 0 if result.get("ok") else 2
        if args.command == "report":
            from .report import write_report
            result = write_report(manifest, output=args.output, fmt=args.format, include_smoke=args.include_smoke); emit(result); return 0 if result.get("ok") else 2
        if args.command == "smoke":
            result = smoke(manifest, prompt=args.prompt, expect=args.expect, max_tokens=args.max_tokens, temperature=args.temperature); emit(result); return 0 if result.get("ok") else 2
        if args.command == "soak":
            result = soak(manifest, count=args.count, delay_sec=args.delay, fail_fast=not args.no_fail_fast); emit(result); return 0 if result.get("ok") else 2
        if args.command == "bench":
            from .bench_artifacts import write_bench_artifact
            prompt_chars = args.prompt_chars or BENCH_PRESETS[args.preset]
            result = bench(manifest, prompt_chars=prompt_chars, repeats=args.repeats, max_tokens=args.max_tokens)
            artifact = write_bench_artifact(manifest, result, output=args.output, fmt=args.format)
            emit(artifact if args.output else result)
            return 0 if result.get("ok") else 2
        if args.command == "watchdog":
            result = watchdog(manifest, max_swap_gib=args.max_swap_gib, duration_sec=args.duration, interval_sec=args.interval, stop_on_breach=args.stop_on_breach); emit(result); return 0 if result.get("ok") else 2
        if args.command == "daemon":
            daemon_health_mode = bool(args.health_mode or args.max_swap_delta_gib is not None or args.sample_sec is not None or args.smoke or args.max_latency_sec is not None)
            result = daemon(manifest, max_swap_gib=args.max_swap_gib, max_swap_delta_gib=args.max_swap_delta_gib, sample_sec=args.sample_sec, include_smoke=args.smoke, max_latency_sec=args.max_latency_sec, health_mode=daemon_health_mode, interval_sec=args.interval, iterations=args.iterations, restart=args.restart, wait=not args.no_wait); emit(result); return 0 if result.get("ok") else 2
        if args.command == "service":
            from .service import diff_service, install_service, service_action
            if args.service_command == "install":
                service_health_mode = bool(args.health_mode or args.max_swap_delta_gib is not None or args.sample_sec is not None or args.smoke or args.max_latency_sec is not None)
                result = install_service(manifest, label=args.label, restart=args.restart, max_swap_gib=args.max_swap_gib, max_swap_delta_gib=args.max_swap_delta_gib, sample_sec=args.sample_sec, include_smoke=args.smoke, max_latency_sec=args.max_latency_sec, health_mode=service_health_mode, interval_sec=args.interval, python=args.python, keep_alive=not args.no_keepalive, run_at_load=args.run_at_load, service_log_path=args.service_log, overwrite=args.overwrite, dry_run=args.dry_run, wait=not args.no_wait); emit(result); return 0 if result.get("ok") else 2
            if args.service_command == "diff":
                service_health_mode = bool(args.health_mode or args.max_swap_delta_gib is not None or args.sample_sec is not None or args.smoke or args.max_latency_sec is not None)
                result = diff_service(manifest, label=args.label, restart=args.restart, max_swap_gib=args.max_swap_gib, max_swap_delta_gib=args.max_swap_delta_gib, sample_sec=args.sample_sec, include_smoke=args.smoke, max_latency_sec=args.max_latency_sec, health_mode=service_health_mode, interval_sec=args.interval, python=args.python, keep_alive=not args.no_keepalive, run_at_load=args.run_at_load, service_log_path=args.service_log, wait=not args.no_wait, include_content=args.content); emit(result); return 0 if result.get("ok") else 2
            result = service_action(manifest, args.service_command, label=args.label, dry_run=args.dry_run); emit(result); return 0 if result.get("ok") else 2
        if args.command == "cleanup":
            emit(cleanup_execute(manifest, force=args.force) if args.execute else cleanup_plan(manifest)); return 0
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr); return 130
    except Exception as exc:
        if exc.__class__.__name__ == "ManifestError" and exc.__class__.__module__ == "modelctl.manifest":
            print(f"manifest error: {exc}", file=sys.stderr); return 2
        if exc.__class__.__name__ == "ServiceError" and exc.__class__.__module__ == "modelctl.service":
            print(f"service error: {exc}", file=sys.stderr); return 2
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr); return 1
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys

from .ingest import ingest
from .manifest import ManifestError, load_manifest
from .ops import bench, cleanup_execute, cleanup_plan, doctor, preflight, smoke, soak, status, validate, watchdog
from .registry import list_registry
from .runner import start, stop, wait_ready

MANIFEST_COMMANDS = {"validate", "preflight", "start", "wait", "stop", "status", "smoke", "soak", "bench", "doctor", "watchdog", "cleanup"}


def emit(obj) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelctl", description="Manifest-driven lifecycle control for local LLM servers.")
    parser.add_argument("-m", "--manifest", default="modelctl.toml", help="Path to model manifest TOML")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="Parse manifest and print resolved summary")
    sub.add_parser("preflight", help="Run required path, port, disk, and swap checks")
    p_list = sub.add_parser("list", help="List manifests in registry directories")
    p_list.add_argument("--registry", action="append", default=[], help="Extra registry directory to scan; can be repeated")
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
    sub.add_parser("status", help="Print process/readiness status")
    sub.add_parser("doctor", help="Run preflight, status, cleanup review, and stale-state diagnostics")
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
    p_bench.add_argument("--prompt-chars", type=parse_int_list, default=[128, 1024], help="Comma-separated synthetic prompt sizes in characters")
    p_bench.add_argument("--repeats", type=int, default=1)
    p_bench.add_argument("--max-tokens", type=int, default=16)
    p_watchdog = sub.add_parser("watchdog", help="Sample readiness/swap and optionally stop the model on breach")
    p_watchdog.add_argument("--max-swap-gib", type=float, default=None)
    p_watchdog.add_argument("--duration", type=float, default=0.0, help="Seconds to watch; 0 means one sample")
    p_watchdog.add_argument("--interval", type=float, default=10.0, help="Seconds between samples")
    p_watchdog.add_argument("--stop-on-breach", action="store_true")
    p_cleanup = sub.add_parser("cleanup", help="Plan or execute cleanup candidates")
    p_cleanup.add_argument("--execute", action="store_true", help="Actually delete safe cleanup candidates")
    p_cleanup.add_argument("--force", action="store_true", help="Allow deleting unsafe cleanup candidates too")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            emit(list_registry(args.registry)); return 0
        if args.command == "ingest":
            result = ingest(args.endpoint, output=args.output, model_id=args.model_id, ident=args.ident, overwrite=args.overwrite); emit(result); return 0 if result.get("ok") else 2
        if args.command not in MANIFEST_COMMANDS:
            parser.error("unknown command")
        manifest = load_manifest(args.manifest)
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
        if args.command == "status":
            emit(status(manifest)); return 0
        if args.command == "doctor":
            result = doctor(manifest); emit(result); return 0 if result.get("ok") else 2
        if args.command == "smoke":
            result = smoke(manifest, prompt=args.prompt, expect=args.expect, max_tokens=args.max_tokens, temperature=args.temperature); emit(result); return 0 if result.get("ok") else 2
        if args.command == "soak":
            result = soak(manifest, count=args.count, delay_sec=args.delay, fail_fast=not args.no_fail_fast); emit(result); return 0 if result.get("ok") else 2
        if args.command == "bench":
            result = bench(manifest, prompt_chars=args.prompt_chars, repeats=args.repeats, max_tokens=args.max_tokens); emit(result); return 0 if result.get("ok") else 2
        if args.command == "watchdog":
            result = watchdog(manifest, max_swap_gib=args.max_swap_gib, duration_sec=args.duration, interval_sec=args.interval, stop_on_breach=args.stop_on_breach); emit(result); return 0 if result.get("ok") else 2
        if args.command == "cleanup":
            emit(cleanup_execute(manifest, force=args.force) if args.execute else cleanup_plan(manifest)); return 0
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr); return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr); return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr); return 1
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

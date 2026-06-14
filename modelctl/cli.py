from __future__ import annotations

import argparse
import json
import sys

from .manifest import ManifestError, load_manifest
from .ops import cleanup_execute, cleanup_plan, doctor, preflight, smoke, soak, status, validate
from .registry import list_registry
from .runner import start, stop, wait_ready

MANIFEST_COMMANDS = {"validate", "preflight", "start", "wait", "stop", "status", "smoke", "soak", "doctor", "cleanup"}


def emit(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelctl", description="Manifest-driven lifecycle control for local LLM servers.")
    parser.add_argument("-m", "--manifest", default="modelctl.toml", help="Path to model manifest TOML")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="Parse manifest and print resolved summary")
    sub.add_parser("preflight", help="Run required path, port, disk, and swap checks")
    p_list = sub.add_parser("list", help="List manifests in registry directories")
    p_list.add_argument("--registry", action="append", default=[], help="Extra registry directory to scan; can be repeated")
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

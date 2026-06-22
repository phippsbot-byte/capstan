from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from .manifest import ManifestError, load_manifest


def default_registry_dirs(extra: list[str] | None = None) -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("MODELCTL_REGISTRY")
    if env:
        dirs.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p)
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    dirs.append(config_home / "modelctl" / "models")
    if extra:
        dirs.extend(Path(p).expanduser() for p in extra)
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        resolved = d.resolve() if d.exists() else d
        if resolved not in seen:
            unique.append(d)
            seen.add(resolved)
    return unique


def selected_registry_dir(registry_dir: str | None = None) -> Path:
    if registry_dir:
        return Path(registry_dir).expanduser()
    env = os.environ.get("MODELCTL_REGISTRY")
    if env:
        first = next((p for p in env.split(os.pathsep) if p), None)
        if first:
            return Path(first).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "modelctl" / "models"


def safe_stem(name: str) -> str:
    stem = name.strip().replace("/", "-").replace(":", "-")
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in stem)
    stem = stem.strip(".-_")
    if not stem:
        raise ManifestError("registry name cannot be empty")
    return stem[:-5] if stem.endswith(".toml") else stem


def iter_manifest_files(registry_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in registry_dirs:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob("*.toml")))
    return files


def _entry_for(path: Path) -> dict[str, Any]:
    try:
        manifest = load_manifest(path)
        return {
            "ok": True,
            "path": str(path),
            "name": path.stem,
            "id": manifest.id,
            "model_id": manifest.model_id,
            "endpoint": manifest.endpoint,
            "description": manifest.description,
            "fleet": {"enabled": manifest.fleet.enabled, "reason": manifest.fleet.reason},
        }
    except ManifestError as exc:
        return {"ok": False, "path": str(path), "name": path.stem, "error": str(exc)}


def list_registry(extra_dirs: list[str] | None = None) -> dict[str, Any]:
    dirs = default_registry_dirs(extra_dirs)
    entries = [_entry_for(path) for path in iter_manifest_files(dirs)]
    return {"registry_dirs": [str(d) for d in dirs], "count": len(entries), "entries": entries}


def find_registry_entry(name: str, extra_dirs: list[str] | None = None) -> dict[str, Any] | None:
    target = safe_stem(name)
    for path in iter_manifest_files(default_registry_dirs(extra_dirs)):
        entry = _entry_for(path)
        if path.stem == target or entry.get("id") == name or entry.get("model_id") == name:
            return entry
    return None


def show_registry(name: str, extra_dirs: list[str] | None = None, include_content: bool = False) -> dict[str, Any]:
    entry = find_registry_entry(name, extra_dirs)
    if not entry:
        return {"ok": False, "error": f"registry entry not found: {name}"}
    if include_content and entry.get("path"):
        entry = dict(entry)
        entry["content"] = Path(entry["path"]).read_text(encoding="utf-8")
    return {"ok": bool(entry.get("ok")), "entry": entry}


def add_registry(source: str, name: str | None = None, registry_dir: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    manifest = load_manifest(source)
    stem = safe_stem(name or manifest.id)
    directory = selected_registry_dir(registry_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{stem}.toml"
    if target.exists() and not overwrite:
        return {"ok": False, "error": f"registry entry exists: {target}; pass --overwrite", "path": str(target)}
    target.write_text(Path(source).expanduser().read_text(encoding="utf-8"), encoding="utf-8")
    # Validate the copy.
    copied = load_manifest(target)
    return {"ok": True, "path": str(target), "name": target.stem, "id": copied.id, "model_id": copied.model_id, "endpoint": copied.endpoint, "fleet": {"enabled": copied.fleet.enabled, "reason": copied.fleet.reason}}


def remove_registry(name: str, registry_dir: str | None = None, missing_ok: bool = False) -> dict[str, Any]:
    dirs = [selected_registry_dir(registry_dir)] if registry_dir else default_registry_dirs([])
    entry = find_registry_entry(name, [str(d) for d in dirs])
    if not entry:
        if missing_ok:
            return {"ok": True, "removed": False, "missing": True, "name": name}
        return {"ok": False, "error": f"registry entry not found: {name}"}
    path = Path(str(entry["path"]))
    path.unlink()
    return {"ok": True, "removed": True, "path": str(path), "name": path.stem}


def use_registry(name: str, output: str = "modelctl.toml", registry_dir: str | None = None, overwrite: bool = False, symlink: bool = False) -> dict[str, Any]:
    dirs = [selected_registry_dir(registry_dir)] if registry_dir else default_registry_dirs([])
    entry = find_registry_entry(name, [str(d) for d in dirs])
    if not entry:
        return {"ok": False, "error": f"registry entry not found: {name}"}
    if not entry.get("ok"):
        return {"ok": False, "error": f"registry entry is invalid: {entry.get('error')}", "entry": entry}
    source = Path(str(entry["path"])).resolve()
    target = Path(output).expanduser()
    if target.exists() or target.is_symlink():
        if not overwrite:
            return {"ok": False, "error": f"output exists: {target}; pass --overwrite", "output": str(target)}
        if target.is_dir() and not target.is_symlink():
            return {"ok": False, "error": f"output is a directory: {target}", "output": str(target)}
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        target.symlink_to(source)
        mode = "symlink"
    else:
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        mode = "copy"
    # Validate the materialized manifest, not just the registry copy.
    materialized = load_manifest(target)
    return {"ok": True, "mode": mode, "source": str(source), "output": str(target), "id": materialized.id, "model_id": materialized.model_id, "endpoint": materialized.endpoint, "fleet": {"enabled": materialized.fleet.enabled, "reason": materialized.fleet.reason}}

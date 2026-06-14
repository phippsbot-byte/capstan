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
    # Preserve order while removing duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        resolved = d.resolve() if d.exists() else d
        if resolved not in seen:
            unique.append(d)
            seen.add(resolved)
    return unique


def iter_manifest_files(registry_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in registry_dirs:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob("*.toml")))
    return files


def list_registry(extra_dirs: list[str] | None = None) -> dict[str, Any]:
    dirs = default_registry_dirs(extra_dirs)
    entries: list[dict[str, Any]] = []
    for path in iter_manifest_files(dirs):
        try:
            manifest = load_manifest(path)
            entries.append({
                "ok": True,
                "path": str(path),
                "id": manifest.id,
                "model_id": manifest.model_id,
                "endpoint": manifest.endpoint,
                "description": manifest.description,
            })
        except ManifestError as exc:
            entries.append({"ok": False, "path": str(path), "error": str(exc)})
    return {"registry_dirs": [str(d) for d in dirs], "count": len(entries), "entries": entries}

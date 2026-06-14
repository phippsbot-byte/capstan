from __future__ import annotations

from typing import Any
import json
import urllib.error
import urllib.request


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> tuple[int, Any, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = text
            return resp.status, body, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = text
        return exc.code, body, text

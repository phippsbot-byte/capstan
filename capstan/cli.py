from __future__ import annotations

from modelctl.cli import main as _modelctl_main


def main(argv: list[str] | None = None) -> int:
    return _modelctl_main(argv, prog="capstan")


if __name__ == "__main__":
    raise SystemExit(main())

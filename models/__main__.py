#!/usr/bin/env python3
"""
Package entrypoint for `python -m models`.

Default runs the BYND pipeline. You can also explicitly run:
  python -m models bynd
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Models package entrypoint")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = False

    subparsers.add_parser("bynd", help="Run BYND pipeline")

    args = parser.parse_args()

    if args.command in (None, "bynd"):
        from .bynd_pipeline import main as bynd_main
        bynd_main()
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()




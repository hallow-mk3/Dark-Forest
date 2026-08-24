from __future__ import annotations

import argparse

import darkforest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="darkforest", description="Dark Forest developer CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print the Dark Forest package version")
    subparsers.add_parser("status", help="Print the runtime status message")
    subparsers.add_parser("smoke-test", help="Run the autograd smoke test")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "version"):
        print(darkforest.version())
        return 0

    if args.command == "status":
        print(darkforest.status())
        return 0

    if args.command == "smoke-test":
        print(darkforest.smoke_test())
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

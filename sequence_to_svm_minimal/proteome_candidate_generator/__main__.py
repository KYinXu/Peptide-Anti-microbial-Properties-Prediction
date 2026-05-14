"""Entrypoint for `python -m proteome_candidate_generator`."""

from __future__ import annotations

from proteome_candidate_generator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import subprocess
from pathlib import Path

from peptide_pipeline.env import subprocess_env


def run_command(cmd: list[str], *, root: Path, dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(root), check=True, env=subprocess_env())

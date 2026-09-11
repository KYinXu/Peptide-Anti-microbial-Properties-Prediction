"""Uniform run directories and provenance manifests for CSV analyses."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RunConfiguration


RUNS_DIRECTORY = "runs"
MANIFEST_FILENAME = "run_manifest.json"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"


@dataclass(frozen=True)
class RunSpec:
    output_root: Path
    output_filename: str
    runner: str
    model: str
    config: Mapping[str, Any] | RunConfiguration = field(default_factory=dict)
    inputs: Mapping[str, Path] = field(default_factory=dict)
    model_files: Mapping[str, Path] = field(default_factory=dict)
    command: Sequence[str] | None = None
    repository_root: Path | None = None


@dataclass
class RunOutput:
    run_id: str
    directory: Path
    csv_path: Path
    manifest_path: Path
    manifest: dict[str, Any]

    def write_manifest(self) -> None:
        _write_json_atomic(self.manifest_path, self.manifest)

    def complete(self, row_count: int) -> None:
        self.manifest["status"] = "completed"
        self.manifest["completed_at_utc"] = _utc_now().isoformat()
        self.manifest["output"] = _file_record(self.csv_path, relative_to=self.directory)
        self.manifest["output"]["row_count"] = row_count
        self.write_manifest()

    def fail(self, error: BaseException) -> None:
        self.manifest["status"] = "failed"
        self.manifest["completed_at_utc"] = _utc_now().isoformat()
        self.manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        if self.csv_path.is_file():
            self.manifest["output"] = _file_record(self.csv_path, relative_to=self.directory)
        self.write_manifest()


def execute_csv_run(
    spec: RunSpec,
    write_csv: Callable[[RunOutput], int],
    *,
    now: datetime | None = None,
) -> RunOutput:
    run = _create_run_output(spec, now=now)
    run.write_manifest()
    try:
        row_count = write_csv(run)
        run.complete(row_count)
    except BaseException as error:
        run.fail(error)
        raise
    return run


def _create_run_output(spec: RunSpec, *, now: datetime | None) -> RunOutput:
    _validate_output_filename(spec.output_filename)
    created_at = _as_utc(now or _utc_now())
    directory = _create_unique_run_directory(spec.output_root, created_at)
    csv_path = directory / spec.output_filename
    manifest = _build_manifest(spec, directory.name, created_at, csv_path)
    return RunOutput(
        run_id=directory.name,
        directory=directory,
        csv_path=csv_path,
        manifest_path=directory / MANIFEST_FILENAME,
        manifest=manifest,
    )


def _create_unique_run_directory(output_root: Path, created_at: datetime) -> Path:
    runs_root = output_root.expanduser().resolve() / RUNS_DIRECTORY
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = created_at.strftime(TIMESTAMP_FORMAT)
    for attempt in range(1, 10_000):
        suffix = "" if attempt == 1 else f"_{attempt:02d}"
        candidate = runs_root / f"{timestamp}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Could not allocate a unique run directory under {runs_root}")


def _build_manifest(spec: RunSpec, run_id: str, created_at: datetime, csv_path: Path) -> dict[str, Any]:
    repository_root = spec.repository_root or _find_repository_root(Path.cwd())
    config = _json_value(spec.config)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "created_at_utc": created_at.isoformat(),
        "runner": spec.runner,
        "model": {
            "name": spec.model,
            "files": {name: _file_record(path) for name, path in spec.model_files.items()},
        },
        "config": config,
        "config_sha256": _value_sha256(config),
        "inputs": {name: _file_record(path) for name, path in spec.inputs.items()},
        "command": list(spec.command) if spec.command is not None else list(sys.argv),
        "repository": _repository_state(repository_root),
        "runtime": {"python": sys.version.split()[0]},
        "output": {"path": csv_path.name},
    }


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    displayed_path = resolved.relative_to(relative_to.resolve()) if relative_to is not None else resolved
    record: dict[str, Any] = {"path": str(displayed_path)}
    if resolved.is_file():
        record.update({"sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size})
    else:
        record["exists"] = False
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _validate_output_filename(filename: str) -> None:
    if not filename or Path(filename).name != filename:
        raise ValueError("output_filename must be a filename without directory components.")
    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("output_filename must use the .csv extension.")


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _repository_state(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(root, "status", "--porcelain")
    return {"root": str(root.resolve()), "commit": commit, "dirty": bool(status)}


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _find_repository_root(start: Path) -> Path | None:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

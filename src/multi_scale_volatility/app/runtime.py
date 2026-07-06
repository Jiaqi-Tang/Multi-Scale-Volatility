"""Terminal logging and append-only runtime tracking."""

from __future__ import annotations

import csv
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

RUNTIME_LOG_COLUMNS = [
    "run_id",
    "stage",
    "operation",
    "baseline_type",
    "simulation_id",
    "started_at_utc",
    "ended_at_utc",
    "elapsed_seconds",
    "status",
    "rows_in",
    "rows_out",
    "output_path",
    "error_message",
]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RuntimeTracker:
    path: Path
    run_id: str = field(default_factory=new_run_id)
    flush_every: int = 10
    pending_rows: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        stage: str,
        operation: str,
        started_at_utc: datetime,
        elapsed_seconds: float,
        baseline_type: str = "",
        simulation_id: int | str = "",
        status: str = "success",
        rows_in: int = 0,
        rows_out: int = 0,
        output_path: str = "",
        error_message: str = "",
        flush: bool = False,
    ) -> None:
        self.pending_rows.append(
            runtime_row(
                run_id=self.run_id,
                stage=stage,
                operation=operation,
                started_at_utc=started_at_utc,
                elapsed_seconds=elapsed_seconds,
                baseline_type=baseline_type,
                simulation_id=simulation_id,
                status=status,
                rows_in=rows_in,
                rows_out=rows_out,
                output_path=output_path,
                error_message=error_message,
            )
        )
        if flush or len(self.pending_rows) >= self.flush_every or status != "success":
            self.flush()

    def extend(self, rows: list[dict[str, Any]], flush: bool = False) -> None:
        for row in rows:
            output = dict(row)
            output["run_id"] = self.run_id
            self.pending_rows.append(output)
        if flush or len(self.pending_rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending_rows:
            return
        append_runtime_rows(self.path, self.pending_rows)
        self.pending_rows.clear()


@dataclass(frozen=True)
class TimedOperation:
    started_at_utc: datetime
    started_at_counter: float

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at_counter


def start_timer() -> TimedOperation:
    return TimedOperation(
        started_at_utc=utc_now(),
        started_at_counter=time.perf_counter(),
    )


@contextmanager
def logged_stage(logger: logging.Logger, name: str) -> Iterator[TimedOperation]:
    timer = start_timer()
    logger.info("Starting %s", name)
    try:
        yield timer
    except Exception:
        logger.exception("Failed %s after %.2fs", name, timer.elapsed_seconds)
        raise
    else:
        logger.info("Finished %s in %.2fs", name, timer.elapsed_seconds)


def runtime_row(
    run_id: str,
    stage: str,
    operation: str,
    started_at_utc: datetime,
    elapsed_seconds: float,
    baseline_type: str,
    simulation_id: int | str,
    status: str,
    rows_in: int,
    rows_out: int,
    output_path: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        "operation": operation,
        "baseline_type": baseline_type,
        "simulation_id": simulation_id,
        "started_at_utc": started_at_utc.isoformat(),
        "ended_at_utc": utc_now().isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "status": status,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "output_path": output_path,
        "error_message": error_message,
    }


def append_runtime_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    try:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUNTIME_LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in RUNTIME_LOG_COLUMNS})
    except PermissionError:
        append_runtime_rows(path.with_name(f"{path.stem}_latest{path.suffix}"), rows)

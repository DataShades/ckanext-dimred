"""Shared deterministic workloads for manual DimRed performance measurements."""

from __future__ import annotations

import json
import multiprocessing
import platform
import resource
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

import ckan

from ckanext.dimred.logic.action import _prepare_feature_matrix
from ckanext.dimred.methods import get_projection_method


@dataclass(frozen=True)
class BenchmarkCase:
    """One deterministic projection workload."""

    method: str
    rows: int
    features: int
    seed: int = 42


@dataclass(frozen=True)
class BenchmarkResult:
    """Measurements emitted by one projection workload."""

    method: str
    rows: int
    features: int
    seed: int
    method_params: dict[str, Any]
    wall_seconds: float
    peak_rss_mb: float
    preview_payload_bytes: int


SMOKE_CASES = (
    BenchmarkCase("pca", rows=1_000, features=10),
    BenchmarkCase("umap", rows=1_000, features=10),
    BenchmarkCase("tsne", rows=500, features=10),
)

FULL_CASES = (
    BenchmarkCase("pca", rows=1_000, features=10),
    BenchmarkCase("pca", rows=5_000, features=10),
    BenchmarkCase("pca", rows=10_000, features=10),
    BenchmarkCase("pca", rows=50_000, features=10),
    BenchmarkCase("pca", rows=10_000, features=50),
    BenchmarkCase("umap", rows=1_000, features=10),
    BenchmarkCase("umap", rows=5_000, features=10),
    BenchmarkCase("umap", rows=10_000, features=10),
    BenchmarkCase("umap", rows=10_000, features=50),
    BenchmarkCase("tsne", rows=500, features=10),
    BenchmarkCase("tsne", rows=1_000, features=10),
    BenchmarkCase("tsne", rows=2_000, features=10),
    BenchmarkCase("tsne", rows=2_000, features=50),
)

CALIBRATION_CASES = {
    "pca-limit": (BenchmarkCase("pca", rows=50_000, features=10),),
    "umap-limit": (BenchmarkCase("umap", rows=10_000, features=10),),
    "tsne-limit": (BenchmarkCase("tsne", rows=2_000, features=10),),
}


def cases_for_profile(profile: str) -> tuple[BenchmarkCase, ...]:
    """Return the bounded benchmark matrix selected by a CLI/test profile."""
    profiles = {"smoke": SMOKE_CASES, "full": FULL_CASES, **CALIBRATION_CASES}
    try:
        return profiles[profile]
    except KeyError as err:
        raise ValueError(profile) from err


def benchmark_case(case: BenchmarkCase) -> BenchmarkResult:
    """Run one case in a forked CKAN process for an isolated peak RSS value."""
    context = multiprocessing.get_context("fork")
    results: multiprocessing.Queue[dict[str, Any]] = context.Queue()
    process = context.Process(target=_benchmark_case_worker, args=(case, results))
    process.start()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(case.method)
    result = results.get_nowait()
    return BenchmarkResult(**result)


def _benchmark_case_worker(case: BenchmarkCase, results: multiprocessing.Queue[dict[str, Any]]) -> None:
    """Measure one case after inheriting the configured CKAN process by fork."""
    matrix = deterministic_matrix(case)
    columns = [f"feature_{index}" for index in range(case.features)]
    dataframe = pd.DataFrame(matrix, columns=columns)
    started = time.perf_counter()
    scaled = _prepare_feature_matrix(dataframe, columns, [])
    method_params = _method_params(case.method)
    method = get_projection_method(case.method)(**method_params)
    embedding = method.fit_transform(scaled)
    wall_seconds = time.perf_counter() - started

    payload = _compact_preview_payload(case, embedding, method.params)
    result = BenchmarkResult(
        **asdict(case),
        method_params=method.params,
        wall_seconds=wall_seconds,
        peak_rss_mb=_peak_rss_mb(),
        preview_payload_bytes=len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
    )
    results.put(asdict(result))


def deterministic_matrix(case: BenchmarkCase) -> np.ndarray:
    """Generate a repeatable numeric resource without external IO or fixtures."""
    generator = np.random.default_rng(case.seed)
    return generator.normal(size=(case.rows, case.features))


def environment_metadata() -> dict[str, str]:
    """Return the versions required to interpret benchmark measurements."""
    return {
        "ckan": ckan.__version__,
        "numpy": metadata.version("numpy"),
        "python": platform.python_version(),
        "scikit-learn": metadata.version("scikit-learn"),
        "umap-learn": metadata.version("umap-learn"),
    }


def result_as_json(result: BenchmarkResult) -> str:
    """Serialize a result as one stable JSON line for manual benchmark output."""
    return json.dumps(asdict(result), sort_keys=True)


def metadata_as_json(metadata_: dict[str, str]) -> str:
    """Serialize benchmark environment metadata as one stable JSON line."""
    return json.dumps(metadata_, sort_keys=True)


def _method_params(method_name: str) -> dict[str, Any]:
    """Read the active CKAN defaults once, then make each run explicit."""
    return get_projection_method(method_name).default_params()


def _compact_preview_payload(
    case: BenchmarkCase,
    embedding: np.ndarray,
    method_params: dict[str, Any],
) -> dict[str, Any]:
    """Model the compact numeric-only preview contract for payload comparison."""
    return {
        "embedding": np.round(embedding, decimals=3).tolist(),
        "meta": {
            "method": case.method,
            "method_params": method_params,
            "prepare_info": {"source_row_ids": list(range(1, case.rows + 1))},
        },
    }


def _peak_rss_mb() -> float:
    """Return the process high-water RSS in MiB on the Linux CKAN runtime."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

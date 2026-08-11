"""Manual benchmark entry point for projection workload calibration."""

from __future__ import annotations

import json
import os

import pytest
from click import echo

from ckanext.dimred.tests.benchmarks.workloads import (
    BenchmarkResult,
    benchmark_case,
    cases_for_profile,
    deterministic_matrix,
    environment_metadata,
    metadata_as_json,
    result_as_json,
)


def test_benchmark_matrix_is_deterministic():
    """Keep benchmark data stable independently from timing-sensitive runs."""
    case = cases_for_profile("smoke")[0]

    assert (deterministic_matrix(case) == deterministic_matrix(case)).all()


def test_benchmark_result_records_effective_method_params():
    """Keep the stored calibration baseline reproducible after config changes."""
    result = BenchmarkResult(
        method="pca",
        rows=1_000,
        features=10,
        seed=42,
        method_params={"n_components": 2, "random_state": 42, "whiten": False},
        wall_seconds=0.1,
        peak_rss_mb=100.0,
        preview_payload_bytes=1_000,
    )

    assert json.loads(result_as_json(result))["method_params"] == result.method_params


@pytest.mark.benchmark
@pytest.mark.usefixtures("with_plugins")
def test_projection_workload_profile():
    """Run a manually selected workload profile and emit machine-readable results."""
    profile = os.environ.get("DIMRED_BENCHMARK_PROFILE", "smoke")
    echo(metadata_as_json(environment_metadata()))

    for case in cases_for_profile(profile):
        result = benchmark_case(case)
        echo(result_as_json(result))
        assert result.wall_seconds > 0
        assert result.peak_rss_mb > 0
        assert result.preview_payload_bytes > 0

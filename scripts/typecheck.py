"""Run BasedPyright against the Python environment that runs this project."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _package_root(package: str) -> str:
    """Return the import root for an installed source or wheel package."""
    spec = find_spec(package)
    if spec is None or spec.origin is None:
        msg = f"Cannot locate the {package!r} package in {sys.executable}"
        raise RuntimeError(msg)
    return str(Path(spec.origin).resolve().parent.parent)


def main() -> int:
    """Configure BasedPyright with the active interpreter's import paths."""
    extra_paths = list(
        dict.fromkeys(
            [
                str(PROJECT_ROOT),
                _package_root("ckan"),
                *(path for path in sys.path if path),
            ]
        )
    )
    config = {
        "include": [str(PROJECT_ROOT / "ckanext")],
        "exclude": [str(PROJECT_ROOT / "ckanext" / "dimred" / "tests")],
        "extraPaths": extra_paths,
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
        "typeCheckingMode": "standard",
        "reportMissingImports": True,
        "reportMissingModuleSource": True,
        "reportUnknownArgumentType": False,
        "reportUnknownLambdaType": False,
        "reportUnknownMemberType": False,
        "reportUnknownParameterType": False,
        "reportUnknownVariableType": False,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
        json.dump(config, config_file)
        config_path = Path(config_file.name)

    try:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "basedpyright", "--project", str(config_path)],
            check=False,
        ).returncode
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

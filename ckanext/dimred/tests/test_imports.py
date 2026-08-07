from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_only_selected_projection_and_matplotlib_backend_are_imported_lazily():
    package_root = Path(__file__).resolve().parents[3] / "ckanext"
    code = f"""
import sys
import ckanext
ckanext.__path__.append({str(package_root)!r})
import ckanext.dimred.utils.core
from ckanext.dimred.methods import get_projection_method

assert 'matplotlib.pyplot' not in sys.modules
assert 'umap' not in sys.modules
assert 'sklearn.decomposition' not in sys.modules
get_projection_method('pca')
assert 'ckanext.dimred.methods.pca' in sys.modules
assert 'sklearn.decomposition' not in sys.modules
get_projection_method('umap')
assert 'ckanext.dimred.methods.umap' in sys.modules
assert 'umap' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

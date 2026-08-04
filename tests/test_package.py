import importlib.metadata
import os

import pytest

import magpylib as m


@pytest.mark.skipif(
    not os.environ.get("CI"),
    reason=(
        "Installed package metadata only matches the source version in a clean "
        "install (e.g. CI). In a local editable checkout the installed version "
        "can lag behind the source tree, so this check is gated to CI."
    ),
)
def test_version() -> None:
    assert importlib.metadata.version("magpylib") == m.__version__

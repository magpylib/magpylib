from __future__ import annotations

import warnings

import pytest


@pytest.fixture
def image_regression_helper(image_regression):
    """Extended image_regression fixture to include helper functionality."""

    def check_image(fig, basename, diff_threshold=0.1, scale=1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            image_bytes = fig.to_image(format="png", scale=scale)
            image_regression.check(
                image_data=image_bytes,
                diff_threshold=diff_threshold,
                basename=basename,
            )

    return check_image

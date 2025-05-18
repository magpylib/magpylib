from __future__ import annotations

import base64
import re
import warnings

import numpy as np
import pytest


def _convert_ndarray_to_list(obj):
    """Recursively convert numpy arrays in dicts/lists to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_ndarray_to_list(i) for i in obj]
    return obj


def _sanitize_ids(obj):
    """Recursively replace random id fields (e.g., id=12345678) with a constant value."""
    if isinstance(obj, dict):
        return {k: _sanitize_ids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_ids(i) for i in obj]
    if isinstance(obj, str):
        # Replace patterns like 'id=12345678' or 'Sensor(id=12345678)'
        return re.sub(r"id=\d+", "id=ID", obj)
    return obj


def _normalize_bdata(obj):
    """Recursively decode bdata fields and convert to lists with canonical endianness."""
    if isinstance(obj, dict):
        # If both bdata and dtype are present, decode and replace with list
        if "bdata" in obj and "dtype" in obj:
            b = base64.b64decode(obj["bdata"])
            dtype = np.dtype(obj["dtype"])
            # Always use little-endian for comparison
            dtype_le = dtype.newbyteorder("<")
            arr = np.frombuffer(b, dtype=dtype_le)
            # Re-encode to bdata for platform-independent storage
            bdata_str = base64.b64encode(arr.astype(dtype_le).tobytes()).decode("ascii")
            return {"bdata": bdata_str, "dtype": obj["dtype"]}
        return {k: _normalize_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_bdata(i) for i in obj]
    return obj


@pytest.fixture
def fig_regression_helper(data_regression, image_regression):
    """Regression helper for Plotly figures using to_plotly_json()."""

    def check_fig(fig, mode="data"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if mode == "data":
                fig_data = fig.to_plotly_json()
                fig_data = _convert_ndarray_to_list(fig_data)
                fig_data = _sanitize_ids(fig_data)
                fig_data = _normalize_bdata(fig_data)
                data_regression.check(fig_data)
            elif mode == "image":
                image_bytes = fig.to_image(format="png", scale=1)
                image_regression.check(image_data=image_bytes, diff_threshold=0.1)

    return check_fig

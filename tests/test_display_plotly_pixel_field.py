from __future__ import annotations

import warnings

import numpy as np
import pytest

import magpylib as magpy

# pylint: disable=assignment-from-no-return
# pylint: disable=no-member


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


@pytest.mark.parametrize(
    "vectorsource",
    ["B", "H", "J", "M"],
)
def test_display_vector_fields(vectorsource, image_regression_helper):
    """Test displaying vector fields."""
    c1 = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), style_opacity=0.2
    )
    ls = np.linspace(-1, 1, 10)
    s0 = magpy.Sensor(pixel=[[x, y, 0] for x in ls for y in ls], position=(0, 0, 0))
    s1 = s0.copy(style_pixel_field_vectorsource=vectorsource)
    fig = magpy.show(c1, s1, backend="plotly", return_fig=True)

    image_regression_helper(fig, f"vector_field_{vectorsource}")


@pytest.mark.parametrize("colorsource", ["H", "Jxy", "Bz", False])
def test_field_coloring(colorsource, image_regression_helper):
    """Test field coloring."""
    c1 = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), style_opacity=0.2
    )
    ls = np.linspace(-1, 1, 10)
    s0 = magpy.Sensor(pixel=[[x, y, 0] for x in ls for y in ls], position=(0, 0, 0))
    s1 = s0.copy(
        style_pixel_field_vectorsource="B", style_pixel_field_colorsource=colorsource
    )
    fig = magpy.show(c1, s1, backend="plotly", return_fig=True)

    image_regression_helper(fig, f"field_coloring_{colorsource}")


@pytest.mark.parametrize("symbol", ["cone", "arrow3d", "arrow"])
def test_directional_symbols(symbol, image_regression_helper):
    """Test different directional symbols."""
    c1 = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), style_opacity=0.2
    )
    ls = np.linspace(-1, 1, 10)
    s0 = magpy.Sensor(pixel=[[x, y, 0] for x in ls for y in ls], position=(0, 0, 0))
    s1 = s0.copy(style_pixel_field_vectorsource="B", style_pixel_field_symbol=symbol)
    fig = magpy.show(c1, s1, backend="plotly", return_fig=True)

    image_regression_helper(fig, f"directional_symbol_{symbol}")


@pytest.mark.parametrize("sizemode", ["constant", "log", "linear"])
def test_sizing_modes(sizemode, image_regression_helper):
    """Test sizing modes of directional symbols."""
    c1 = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), style_opacity=0.2
    )
    ls = np.linspace(-1, 1, 10)
    s0 = magpy.Sensor(pixel=[[x, y, 0] for x in ls for y in ls], position=(0, 0, 0))
    s1 = s0.copy(
        style_pixel_field_vectorsource="B", style_pixel_field_sizemode=sizemode
    )
    fig = magpy.show(c1, s1, backend="plotly", return_fig=True)

    image_regression_helper(fig, f"sizing_mode_{sizemode}")


@pytest.mark.parametrize("shownull", [True, False])
def test_null_values(shownull, image_regression_helper):
    """Test handling of null or NaN values."""
    c1 = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), style_opacity=0.2
    )
    ls = np.linspace(-1, 1, 5)
    s0 = magpy.Sensor(pixel=[[x, y, 0] for x in ls for y in ls], position=(0, 0, 0))
    s1 = s0.copy(
        style_pixel_field_vectorsource="B", style_pixel_field_shownull=shownull
    )
    fig = magpy.show(c1, s1, backend="plotly", return_fig=True)

    image_regression_helper(fig, f"sizing_null_values_{shownull}")


@pytest.mark.parametrize("colorscale", ["Viridis", "Inferno", "Oranges", "RdPu"])
def test_color_scales(colorscale, image_regression_helper):
    """Test different color scales."""
    c1 = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), style_opacity=0.2
    )
    ls = np.linspace(-1, 1, 10)
    s0 = magpy.Sensor(pixel=[[x, y, 1] for x in ls for y in ls], position=(0, 0, 0))
    s1 = s0.copy(
        style_pixel_field_vectorsource="B", style_pixel_field_colorscale=colorscale
    )
    fig = magpy.show(c1, s1, backend="plotly", return_fig=True)

    image_regression_helper(fig, f"color_scales_{colorscale}")

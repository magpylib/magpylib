"""Helpers a display backend may reuse to draw as the built-in backends do.

Exported from `magpylib.graphics.backend`. They take and return plain values,
so a backend drawing sensor pixels its own way -- as instances in a scene
graph, say -- sizes and colours them exactly as the built-ins do.

Nothing from magpylib is imported at module level: the display modules use
these helpers too, and this module must be importable from any of them.
"""

import re

import numpy as np

#: The values ``style.pixel.field.sizescaling`` and ``colorscaling`` take.
_SCALINGS = re.compile(r"uniform|linear|log|log\^[2-9]")


def sensor_pixel_fields(objects):
    """The field each sensor's pixels read, for the sensors that show it.

    Parameters
    ----------
    objects : iterable
        Sources, sensors and collections, as given to `magpylib.show`.

    Returns
    -------
    dict
        ``{sensor: {field: array}}`` for every sensor among `objects`,
        collections searched, whose ``style.pixel.field.source`` is set.
        `field` is that source's letter (``"B"`` for ``"Bxy"``), and the array,
        of shape ``(path_length, n_pixels, 3)``, holds all three components in
        the sensor's own frame, summed over the sources among `objects`. The
        inner dict is empty when there are no sources.
    """
    # pylint: disable=import-outside-toplevel
    from magpylib._src.display.traces_generic import (  # noqa: PLC0415
        get_sensor_pixel_field,
    )

    return get_sensor_pixel_field(list(objects))


def scale_pixel_norms(norms, scaling="linear", *, minimum=None):
    """Factors for pixel sizes or colours, from field norms, as magpylib
    scales them.

    Parameters
    ----------
    norms : array-like
        Field magnitudes, any shape, typically ``(path_length, n_pixels)`` so
        that the scaling spans the whole path, as in the built-in backends. A
        zero or NaN marks a pixel that reads nothing.
    scaling : {'uniform', 'linear', 'log', 'log^2', ..., 'log^9'}
        As ``style.pixel.field.sizescaling`` or ``colorscaling``. The norms are
        first spread over [0, 1], then passed through the logarithm as many
        times as the scaling says.
    minimum : float, optional
        Smallest factor, as ``style.pixel.field.sizemin``.

    Returns
    -------
    numpy.ndarray
        Factors in [`minimum` or 0, 1], of the shape of `norms`; NaN where a
        pixel reads nothing.
    """
    if not isinstance(scaling, str) or not _SCALINGS.fullmatch(scaling):
        msg = (
            "scaling must be one of 'uniform', 'linear', 'log', or 'log^2' to "
            f"'log^9'; instead received {scaling!r}."
        )
        raise ValueError(msg)
    # pylint: disable=import-outside-toplevel
    from magpylib._src.display.traces_core import (  # noqa: PLC0415
        _apply_scaling_transformation,
    )

    norms = np.array(norms, dtype=float)
    null = np.logical_or(norms == 0, np.isnan(norms))
    norms[null] = np.nan
    if null.all():
        return norms
    if scaling == "uniform":
        return np.where(null, np.nan, 1.0)
    low, high = np.nanmin(norms), np.nanmax(norms)
    spread = high - low
    norms = (norms - low) / spread if spread != 0 else norms * 0 + 0.5
    return _apply_scaling_transformation(norms, scaling, null, ..., min_=minimum)


def colors_from_colorscale(
    values, colorscale, *, cmin=None, cmax=None, nan_color="#b2beb5"
):
    """Hex colours for `values` along a colorscale.

    Parameters
    ----------
    values : array-like, shape (n,)
        The values to colour.
    colorscale : str | sequence
        A Plotly colorscale name such as ``"Viridis"``, or ``[(0, color), ...,
        (1, color)]``, as ``style.pixel.field.colormap`` takes.
    cmin, cmax : float, optional
        The values mapped to the two ends of the scale; by default the
        smallest and largest of `values`.
    nan_color : str, default "#b2beb5"
        The colour of a NaN value.

    Returns
    -------
    numpy.ndarray
        One ``"#rrggbb"`` string per value.
    """
    # pylint: disable=import-outside-toplevel
    from magpylib._src.display.traces_utility import (  # noqa: PLC0415
        get_hexcolors_from_colormap,
    )

    return get_hexcolors_from_colormap(
        values, colorscale, cmin=cmin, cmax=cmax, nan_color=nan_color
    )

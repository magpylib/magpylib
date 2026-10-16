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
        ``{sensor: array}`` for every sensor among `objects`, collections
        searched, whose ``style.pixel.field.source`` is set and that has
        sources to read. The array, of shape ``(path_length, n_pixels, 3)``,
        is the field in the sensor's own frame, summed over the sources among
        `objects`, with the components the source does not name set to 0:
        ``"Bxy"`` keeps x and y.
    """
    # pylint: disable=import-outside-toplevel
    from magpylib._src.display.traces_generic import (  # noqa: PLC0415
        get_sensor_pixel_field,
    )

    fields = {}
    for sensor, read in get_sensor_pixel_field(list(objects)).items():
        if not read:  # no sources among `objects`
            continue
        ((_, values),) = read.items()
        named = sensor.style.pixel.field.source[1:] or "xyz"
        values = np.array(values, dtype=float)  # a copy: the cache is shared
        values[..., [i for i, axis in enumerate("xyz") if axis not in named]] = 0
        fields[sensor] = values
    return fields


def scale_pixel_norms(norms, scaling, *, minimum=None):
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
        Smallest factor, in [0, 1], as ``style.pixel.field.sizemin``.

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
    if minimum is not None and not 0 <= minimum <= 1:
        msg = f"minimum must be in [0, 1]; instead received {minimum!r}."
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
        A Plotly colorscale name such as ``"Viridis"``, or ``[(stop, colour),
        ...]`` with stops from 0 to 1 and any CSS name or hex colour, as a
        mesh trace's own ``colorscale``.
    cmin, cmax : float, optional
        The values mapped to the two ends of the scale; by default the
        smallest and largest of `values`, a single value mapping to the
        middle. For factors from `scale_pixel_norms`, pass ``cmin=0, cmax=1``,
        as the built-in backends do.
    nan_color : str, default "#b2beb5"
        The colour of a NaN value, returned as given.

    Returns
    -------
    list of str
        One colour per value: ``"#rrggbb"``, or `nan_color` for NaN.
    """
    # pylint: disable=import-outside-toplevel
    from matplotlib.colors import rgb2hex, to_rgb  # noqa: PLC0415
    from plotly.colors import sample_colorscale  # noqa: PLC0415
    from plotly.exceptions import PlotlyError  # noqa: PLC0415

    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.ndim != 1:
        # a (path, n) array flattened would colour every path step at once
        msg = f"values must be one-dimensional; instead received shape {values.shape}."
        raise ValueError(msg)
    valid = ~np.isnan(values)
    colors = [nan_color] * len(values)
    if not valid.any():
        return colors
    low = np.min(values[valid]) if cmin is None else cmin
    high = np.max(values[valid]) if cmax is None else cmax
    spread = high - low
    scaled = (values[valid] - low) / spread if spread != 0 else values[valid] * 0 + 0.5
    if not isinstance(colorscale, str):
        # plotly samples only its own colour notation
        stops = []
        for stop, color in colorscale:
            r, g, b = (round(255 * c) for c in to_rgb(color))
            stops.append([float(stop), f"rgb({r}, {g}, {b})"])
        colorscale = stops
    try:
        sampled = sample_colorscale(colorscale, np.clip(scaled, 0, 1), colortype=None)
    except PlotlyError as err:  # an unknown scale name
        msg = (
            "colorscale must be a Plotly colorscale name or (stop, colour) "
            f"pairs; instead received {colorscale!r}."
        )
        raise ValueError(msg) from err
    for index, rgb in zip(np.flatnonzero(valid), sampled, strict=True):
        colors[index] = rgb2hex(rgb)
    return colors

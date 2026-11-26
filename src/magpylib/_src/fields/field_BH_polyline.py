"""Line current segments field implementation."""

# pylint: disable=too-many-positional-arguments

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import (
    array_namespace,
    cross,
    jax_padded,
    xp_promote,
)
from magpylib._src.array_api_utils import dot as dot3
from magpylib._src.input_checks import check_field_input


def _current_vertices_field(
    field: str,
    observers: np.ndarray,
    current: np.ndarray,
    vertices: np.ndarray = None,
    segment_start=None,
    segment_end=None,
) -> np.ndarray:
    """
    This function accepts n vertex sets, creates a single long input array of
    segments for _BHJM_current_polyline(), computes, sums and returns a single
    field for each vertex set at respective n observer positions.

    ### Args:
    - field (str): "B", "H", "J" or "M"
    - observers (ndarray nx3): n observer positions in units (m)
    - current (ndarray n): current on line in units (A)
    - vertices (ndarray nxmx3): n vertex sets in units (m); when their lengths
      mi differ, a list or object array of n arrays of shape (mi, 3)
    - segment_start, segment_end (ndarray nx3): single segments, used when
      vertices is None

    ### Returns:
    - BH (ndarray nx3): field at the observers in units (T) or (A/m)
    """
    if vertices is None:
        return _BHJM_current_polyline(
            field=field,
            observers=observers,
            current=current,
            segment_start=segment_start,
            segment_end=segment_end,
        )

    # getB passes vertex sets of differing lengths as an object array
    if isinstance(vertices, np.ndarray) and vertices.dtype != object:
        n0, n1, *_ = vertices.shape
        BH = _BHJM_current_polyline(
            field=field,
            observers=np.repeat(observers, n1 - 1, axis=0),
            current=np.repeat(current, n1 - 1, axis=0),
            segment_start=vertices[:, :-1].reshape(-1, 3),
            segment_end=vertices[:, 1:].reshape(-1, 3),
        )
        BH = BH.reshape((n0, n1 - 1, 3))
        BH = np.sum(BH, axis=1)
    else:
        nseg = np.fromiter(map(len, vertices), int, len(vertices)) - 1  # per set
        verts = np.concatenate(vertices)
        # a segment joins two consecutive vertices of the same set
        same_set = np.ones(len(verts) - 1, dtype=bool)
        same_set[np.cumsum(nseg + 1)[:-1] - 1] = False
        segment_start, segment_end = verts[:-1][same_set], verts[1:][same_set]
        del verts
        BH_seg = _BHJM_current_polyline(
            field=field,
            observers=np.repeat(observers, nseg, axis=0),
            current=np.repeat(current, nseg, axis=0),
            segment_start=segment_start,
            segment_end=segment_end,
        )
        # sum the segments of each set; a set without segments stays zero
        has_seg = nseg > 0
        first_seg = (np.cumsum(nseg) - nseg)[has_seg]  # of each set in BH_seg
        BH = np.zeros((len(nseg), 3))
        BH[has_seg] = np.add.reduceat(BH_seg, first_seg, axis=0)
    return BH


@jax_padded()
def current_polyline_Hfield(
    observers: np.ndarray,
    segments_start: np.ndarray,
    segments_end: np.ndarray,
    currents: np.ndarray,
) -> np.ndarray:
    """H-field of i straight current segments in Cartesian coordinates.

    The current flows from start to end positions. The field is set to (0, 0, 0) on a
    line segment. The output is proportional to the current and independent of the
    length units chosen for observers and dimensions.

    Parameters
    ----------
    observers : ndarray, shape (i, 3)
        Observer positions (x, y, z) in Cartesian coordinates.
    segments_start : ndarray, shape (i, 3)
        Segment start positions (x, y, z) in Cartesian coordinates.
    segments_end : ndarray, shape (i, 3)
        Segment end positions (x, y, z) in Cartesian coordinates.
    currents : ndarray, shape (i,)
        Electrical currents (A) in the segments.

    Returns
    -------
    ndarray, shape (i, 3)
        H-field in (A/m) at the observer positions.

    Notes
    -----
    Field computation via the Biot-Savart law. See also many resources, e.g.,
    http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf

    Be careful with magnetic fields of isolated, discontinuous segments; they
    are unphysical and can lead to unphysical effects.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> H = magpy.core.current_polyline_Hfield(
    ...     observers=np.array([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]),
    ...     segments_start=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]),
    ...     segments_end=np.array([(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]),
    ...     currents=np.array([100.0, 200.0]),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [[ 0.    -2.297  2.297]
     [ 0.     0.598 -0.598]]
    """
    xp = array_namespace(observers, segments_start, segments_end, currents)
    observers, segments_start, segments_end, currents = xp_promote(
        observers, segments_start, segments_end, currents, xp=xp
    )

    def dot(vectors, others):
        return dot3(vectors, others)[:, xp.newaxis]

    # make dimensionless (avoid all large/small input problems) by introducing
    # the segment length as characteristic length scale.
    segment = segments_end - segments_start
    length = xp.sqrt(dot(segment, segment))
    e = segment / length
    r1 = (observers - segments_start) / length
    r2 = (observers - segments_end) / length

    # field direction, and squared distance of the observers from the line
    cros_ = cross(e, r1)
    d2 = dot(cros_, cros_)

    # separate on-line cases (-> H=0), which are computed with harmless
    # values in place of the distances that are zero there
    on_line = d2 < 1e-30  # account for numerical issues
    one = xp.ones_like(d2)
    d2 = xp.where(on_line, one, d2)
    R1 = xp.sqrt(xp.where(on_line, one, dot(r1, r1)))
    R2 = xp.sqrt(xp.where(on_line, one, dot(r2, r2)))

    # Positions of the observers along the line, from start and from end.
    # They differ by the length, 1. Rounded independently they do not, by
    # more the farther the observer: the smaller one is kept and gives the
    # other.
    t1 = dot(e, r1)
    t2 = dot(e, r2)
    nearer_end = xp.abs(t2) < xp.abs(t1)
    t1 = xp.where(nearer_end, t2 + 1, t1)
    t2 = xp.where(nearer_end, t2, t1 - 1)

    # Biot-Savart: H = I / (4 pi) * k * (e x r1), k = (t1/R1 - t2/R2) / d2.
    # Beyond an end of the segment the two cosines have one sign and cancel,
    # the more the farther along the line (#1097). There k is rationalized
    # to a sum of terms of one sign, with t1 - t2 = 1. Its denominator is
    # divided by in two steps, as the product overflows in float32 from 1e9
    # lengths.
    beyond = t1 * t2 > 0
    k = xp.where(
        beyond,
        (t1 + t2) / (R1 * R2) / xp.where(beyond, t1 * R2 + t2 * R1, one),
        (t1 / R1 - t2 / R2) / d2,
    )

    H = k * cros_ / length * currents[:, xp.newaxis] / (4 * xp.pi)
    return xp.where(on_line, 0.0, H)


def _BHJM_current_polyline(
    field: str,
    observers: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """
    - translate Polyline field to BHJM
    - treat some special cases
    """
    # pylint: disable=too-many-statements

    check_field_input(field)

    xp = array_namespace(observers, segment_start, segment_end, current)
    observers, segment_start, segment_end, current = xp_promote(
        observers, segment_start, segment_end, current, xp=xp
    )

    if field in "MJ":
        return xp.zeros_like(observers)

    # Check for zero-length segments (or discontinuous)
    mask_nan_start = xp.all(xp.isnan(segment_start), axis=-1)
    mask_nan_end = xp.all(xp.isnan(segment_end), axis=-1)
    mask_equal = xp.all(segment_start == segment_end, axis=-1)
    mask0 = (mask_equal | mask_nan_start | mask_nan_end)[:, xp.newaxis]

    # continue only with non-zero segments: the others are computed as a
    # unit segment along x, and return (0, 0, 0)
    unit_x = xp.asarray([1.0, 0.0, 0.0], dtype=observers.dtype)
    BHJM = xp.where(
        mask0,
        0.0,
        current_polyline_Hfield(
            observers=observers,
            segments_start=xp.where(mask0, 0.0, segment_start),
            segments_end=xp.where(mask0, unit_x, segment_end),
            currents=current,
        ),
    )

    if field == "H":
        return BHJM

    if field == "B":
        return BHJM * MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover

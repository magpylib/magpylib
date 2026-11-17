"""Line current segments field implementation."""

# pylint: disable=too-many-positional-arguments

import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import xp_promote
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

    def norm(vectors):
        return xp.linalg.vector_norm(vectors, axis=-1, keepdims=True)

    # rename
    p1, p2, po = segments_start, segments_end, observers

    # make dimensionless (avoid all large/small input problems) by introducing
    # the segment length as characteristic length scale.
    norm_12 = norm(p1 - p2)
    p1 = p1 / norm_12
    p2 = p2 / norm_12
    po = po / norm_12

    # p4 = projection of pos_obs onto line p1-p2
    t = xp.sum((po - p1) * (p1 - p2), axis=-1, keepdims=True)
    p4 = p1 + t * (p1 - p2)

    # distance of observers from line
    norm_o4 = norm(po - p4)

    # separate on-line cases (-> H=0), which are computed with harmless
    # values in place of the distances that are zero there
    mask1 = norm_o4 < 1e-15  # account for numerical issues

    # determine field direction
    cros_ = xp.linalg.cross(p2 - p1, po - p4)
    eB = cros_ / xp.where(mask1, 1.0, norm(cros_))

    # compute angles
    norm_o1 = norm(po - p1)
    norm_o2 = norm(po - p2)
    norm_41 = norm(p4 - p1)
    norm_42 = norm(p4 - p2)
    sinTh1 = norm_41 / xp.where(mask1, 1.0, norm_o1)
    sinTh2 = norm_42 / xp.where(mask1, 1.0, norm_o2)

    # determine how p1, p2, p4 are sorted on the line (to get sinTH signs)
    # both points below
    mask2 = (norm_41 > 1) & (norm_41 > norm_42)
    # both points above
    mask3 = (norm_42 > 1) & (norm_42 > norm_41)
    # else: one above one below or one equals p4
    deltaSin = xp.where(mask2 | mask3, xp.abs(sinTh1 - sinTh2), xp.abs(sinTh1 + sinTh2))

    H = (
        deltaSin
        / xp.where(mask1, 1.0, norm_o4)
        * eB
        / norm_12
        * currents[:, xp.newaxis]
        / (4 * xp.pi)
    )
    return xp.where(mask1, 0.0, H)


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

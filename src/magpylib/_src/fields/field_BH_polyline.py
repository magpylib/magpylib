"""Line current segments field implementation."""

# pylint: disable=too-many-positional-arguments

import numpy as np
from numpy.linalg import norm
from scipy.constants import mu_0 as MU0

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
    # rename
    p1, p2, po = segments_start, segments_end, observers

    # make dimensionless (avoid all large/small input problems) by introducing
    # the segment length as characteristic length scale.
    norm_12 = norm(p1 - p2, axis=1)
    p1 = (p1.T / norm_12).T
    p2 = (p2.T / norm_12).T
    po = (po.T / norm_12).T

    # p4 = projection of pos_obs onto line p1-p2
    t = np.sum((po - p1) * (p1 - p2), axis=1)
    p4 = p1 + (t * (p1 - p2).T).T

    # distance of observers from line
    norm_o4 = norm(po - p4, axis=1)

    # separate on-line cases (-> B=0)
    mask1 = norm_o4 < 1e-15  # account for numerical issues

    # continue only with general off-line cases
    if np.any(mask1):
        not_mask1 = ~mask1
        po = po[not_mask1]
        p1 = p1[not_mask1]
        p2 = p2[not_mask1]
        p4 = p4[not_mask1]
        norm_12 = norm_12[not_mask1]
        norm_o4 = norm_o4[not_mask1]
        currents = currents[not_mask1]

    # determine field direction
    cros_ = np.cross(p2 - p1, po - p4)
    norm_cros = norm(cros_, axis=1)
    eB = (cros_.T / norm_cros).T

    # compute angles
    norm_o1 = norm(
        po - p1, axis=1
    )  # improve performance by computing all norms at once
    norm_o2 = norm(po - p2, axis=1)
    norm_41 = norm(p4 - p1, axis=1)
    norm_42 = norm(p4 - p2, axis=1)
    sinTh1 = norm_41 / norm_o1
    sinTh2 = norm_42 / norm_o2
    deltaSin = np.empty((len(po),))

    # determine how p1, p2, p4 are sorted on the line (to get sinTH signs)
    # both points below
    mask2 = (norm_41 > 1) * (norm_41 > norm_42)
    deltaSin[mask2] = abs(sinTh1[mask2] - sinTh2[mask2])
    # both points above
    mask3 = (norm_42 > 1) * (norm_42 > norm_41)
    deltaSin[mask3] = abs(sinTh2[mask3] - sinTh1[mask3])
    # one above one below or one equals p4
    mask4 = ~mask2 * ~mask3
    deltaSin[mask4] = abs(sinTh1[mask4] + sinTh2[mask4])

    # B = (deltaSin / norm_o4 * eB.T / norm_12 * current * 1e-7).T

    # avoid array creation if possible
    if np.any(mask1):
        H = np.zeros_like(observers, dtype=float)
        H[~mask1] = (deltaSin / norm_o4 * eB.T / norm_12 * currents / (4 * np.pi)).T
        return H
    return (deltaSin / norm_o4 * eB.T / norm_12 * currents / (4 * np.pi)).T


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

    BHJM = np.zeros_like(observers, dtype=float)

    if field in "MJ":
        return BHJM

    # Check for zero-length segments (or discontinuous)
    mask_nan_start = np.isnan(segment_start).all(axis=1)
    mask_nan_end = np.isnan(segment_end).all(axis=1)
    mask_equal = np.all(segment_start == segment_end, axis=1)
    mask0 = mask_equal | mask_nan_start | mask_nan_end
    not_mask0 = ~mask0  # avoid multiple computation of ~mask

    if np.all(mask0):
        return BHJM

    # continue only with non-zero segments
    if np.any(mask0):
        current = current[not_mask0]
        segment_start = segment_start[not_mask0]
        segment_end = segment_end[not_mask0]
        observers = observers[not_mask0]

    BHJM[not_mask0] = current_polyline_Hfield(
        observers=observers,
        segments_start=segment_start,
        segments_end=segment_end,
        currents=current,
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

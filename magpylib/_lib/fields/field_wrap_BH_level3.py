from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2


# ON INTERFACE
def getB(sources, observers, sumup=False, squeeze=True, **specs):
    """
    Compute B-field in [mT] for given sources and observers.

    Parameters
    ----------
    sources: source object, Collection or 1D list thereof
        Sources can be a single source object, a Collection or a 1D list of L source
        objects and/or collections.

    observers: array_like or Sensor or 1D list thereof
        Observers can be array_like positions of shape (N1, N2, ..., 3) where the field
        should be evaluated, can be a Sensor object with pixel shape (N1, N2, ..., 3) or
        a 1D list of K Sensor objects with similar pixel shape. All positions are given
        in units of [mm].

    sumup: bool, default=False
        If True, the field of all sources is summed up.

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    Returns
    -------
    B-field: ndarray, shape squeeze(L, M, K, N1, N2, ..., 3), unit [mT]
        B-field of each source (L) at each path position (M) for each sensor (K) and each
        sensor pixel position (Ni) in units of [mT]. Sensor pixel positions are equivalent
        to simple observer positions. Paths of objects that are shorter than M will be
        considered as static beyond their end.

    Note
    ----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.
    """
    return getBH_level2(True, sources, observers, sumup, squeeze, **specs)


# ON INTERFACE
def getH(sources, observers, sumup=False, squeeze=True, **specs):
    """
    Compute H-field in [kA/m] for given sources and observers.

    Parameters
    ----------
    sources: source object, Collection or 1D list thereof
        Sources can be a single source object, a Collection or a 1D list of L source
        objects and/or collections.

    observers: array_like or Sensor or 1D list thereof
        Observers can be array_like positions of shape (N1, N2, ..., 3) where the field
        should be evaluated, can be a Sensor object with pixel shape (N1, N2, ..., 3) or
        a 1D list of K Sensor objects with similar pixel shape. All positions are given
        in units of [mm].

    sumup: bool, default=False
        If True, the field of all sources is summed up.

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    Returns
    -------
    H-field: ndarray, shape squeeze(L, M, K, N1, N2, ..., 3), unit [kA/m]
        H-field of each source (L) at each path position (M) for each sensor (K) and each
        sensor pixel position (Ni) in units of [mT]. Sensor pixel positions are equivalent
        to simple observer positions. Paths of objects that are shorter than M will be
        considered as static beyond their end.

    Note
    ----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.
    """
    return getBH_level2(False, sources, observers, sumup, squeeze, **specs)

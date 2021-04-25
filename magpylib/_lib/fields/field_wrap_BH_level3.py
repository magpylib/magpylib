from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2


# ON INTERFACE
def getB(sources, observers, sumup=False, squeeze=True, **specs):
    """
    Compute B-field for given sources and observers.

    Parameters
    ----------
    sources: src objects, Collections or arbitrary lists thereof
        Source object or a 1D list of L source objects and/or collections. Pathlength of all
        sources must be M or 1. Sources with Pathlength=1 will be considered as static.

    observers: array_like or Sensor or list of Sensors
        Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
        a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
        of [mm].

    sumup: bool, default=False
        If True, the field of all sources is summed up.

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    Returns
    -------
    B-field: ndarray, shape squeeze(L, M, K, N1, N2, ..., 3), unit [mT]
        B-field of each source (L) at each path position (M) for each sensor (K) and each sensor
        pixel position (N) in units of [mT].
        Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
        single sensor or no sensor or single pixel) is removed.

    Info
    ----
    This function automatically joins all sensor and position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(True, sources, observers, sumup, squeeze, **specs)


# ON INTERFACE
def getH(sources, observers, sumup=False, squeeze=True, **specs):
    """ Compute H-field for given sources and observers.

    Parameters
    ----------
    sources: src_obj, col_obj or list thereof
        Source object or a 1D list of L source objects and/or collections. Pathlength of all
        sources must be M or 1. Sources with Pathlength=1 will be considered as static.

    observers: array_like or sens_obj or list of sens_obj
        Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
        a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
        of [mm].

    sumup: bool, default=False
        If True, the field of all sources is summed up.

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    Returns
    -------
    H-field: ndarray, shape squeeze(L, M, K, N1, N2, ..., 3), unit [kA/m]
        H-field of each source (L) at each path position (M) for each sensor (K) and each sensor
        pixel position (N) in units of [kA/m].
        Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
        single sensor or no sensor or single pixel) is removed.

    Info
    ----
    This function automatically joins all sensor and position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(False, sources, observers, sumup, squeeze, **specs)

from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2

def getB(sources, observers, sumup=False, **specs):
    """ Compute the B-field for given sources and observers

    Parameters
    ----------
    sources: src_obj or list of src_obj
        Source object or a 1D list of L source objects and collections. Pathlength of all sources
        must be the same (or 1). Pathlength=1 sources will be considered as static.

    observers: array_like or sens_obj or list of sens_obj
        Observers can be array_like positions of shape (N1, N2, ..., 3) or a sensor or
        a 1D list of K sensors with pixel position shape of (N1, N2, ..., 3)
        in units of millimeters.

    sumup: bool, default=False
        If true, the field of all sources is summed up.

    Specific kwargs
    ---------------
    niter: int, default=50
        for Cylinder sources diametral iteration (Simpsons formula).

    Returns
    -------
    B-field: ndarray, shape (L, M, K, N1, N2, ..., 3), unit [mT]
        B-field of each source at each path position for each sensor and each sensor pixel
        position in units of mT.
        Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
        single sensor or no sensor) is removed.

    Info
    ----
    This function automatically joins all sensor and position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(True, sources, observers, sumup, **specs)


def getH(sources, observers, sumup=False, **specs):
    """ Compute the H-field for given sources and observers

    Parameters
    ----------
    sources: src_obj or list of src_obj
        Source object or a 1D list of L source objects and collections. Pathlength of all sources
        must be the same (or 1). Pathlength=1 sources will be considered as static.

    observers: array_like or sens_obj or list of sens_obj
        Observers can be array_like positions of shape (N1, N2, ..., 3) or a sensor or
        a 1D list of K sensors with pixel position shape of (N1, N2, ..., 3) in units
        of millimeters.

    sumup: bool, default=False
        If true, the field of all sources is summed up.

    Specific kwargs
    ---------------
    niter: int, default=50
        for Cylinder sources diametral iteration (Simpsons formula).

    Returns
    -------
    H-field: ndarray, shape (L, M, K, N1, N2, ..., 3), unit [kA/m]
        H-field of each source at each path position for each sensor and each sensor pixel
        position in units of kA/m.
        Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
        single sensor or no sensor) is removed.

    Info
    ----
    This function automatically joins all sensor and position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(False, sources, observers, sumup, **specs)

from typing import Sequence
import numpy as np
import magpylib as mag3
from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2


def getB_from_sensor(sources, sensors, sumup=False, **specs):
    """ Compute the B-field for given sources and sensors

    Parameters
    ----------
    sources: src_obj or list
        Source object or a 1D list of L source objects and collections. Pathlength of all sources
        must be the same (or 1). Pathlength=1 sources will be considered as static.

    sensors: sens_obj or list
        Sensor object or a 1D list of K sensor objects. All sensors should have similar pos_pix
        shape of (N1, N2, ..., 3), otherwise all sensors will be merged to K=1. pos_pix unit is
        millimeters.

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
        single sensor) is removed.

    Info
    ----
    This function automatically joins all sensor position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(True, sources, sensors, sumup, **specs)


def getH_from_sensor(sources, sensors, sumup=False, **specs):
    """ Compute the H-field for given sources and sensors

    Parameters
    ----------
    sources: src_obj or list
        Source object or a 1D list of L source objects and collections. Pathlength of all sources
        must be the same (or 1). Pathlength=1 sources will be considered as static.

    sensors: sens_obj or list
        Sensor object or a 1D list of K sensor objects. All sensors should have similar pos_pix
        shape of (N1, N2, ..., 3), otherwise all sensors will be merged to K=1. pos_pix unit is
        millimeters.

    sumup: bool, default=False
        If true, the field of all sources is summed up.

    Specific kwargs
    ---------------
    niter: int, default=50
        for Cylinder sources diametral iteration (Simpsons formula).

    Returns
    -------
    B-field: ndarray, shape (L, M, K, N1, N2, ..., 3), unit [kA/m]
        B-field of each source at each path position for each sensor and each sensor pixel
        position in units of mT.
        Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
        single sensor) is removed.

    Info
    ----
    This function automatically joins all sensor position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    return getBH_level2(True, sources, sensors, sumup, **specs)


def getB(sources, pos_obs, sumup=False, **specs):
    """ Compute the B-field for given sources and sensors

    Parameters
    ----------
    sources: src_obj or list
        Source object or a 1D list of L source objects and collections. Pathlength of all sources
        must be the same (or 1). Pathlength=1 sources will be considered as static.

    pos_obs: array_like
        Observer positions with a shape of (N1, N2, ..., 3) in units of millimeters.

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
        single sensor) is removed.

    Info
    ----
    This function automatically joins all sensor position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    sens = mag3.Sensor(pos_pix=pos_obs) # create a static dummy sensor for level2 input
    return getBH_level2(True, sources, sens, sumup, **specs)


def getH(sources:Sequence, pos_obs:np.ndarray, sumup:bool=False, **specs:dict) -> np.ndarray:
    """ Compute the H-field for given sources and sensors

    Parameters
    ----------
    sources: src_obj or list
        Source object or a 1D list of L source objects and collections. Pathlength of all sources
        must be the same (or 1). Pathlength=1 sources will be considered as static.

    pos_obs: array_like
        Observer positions with a shape of (N1, N2, ..., 3) in units of millimeters.

    sumup: bool, default=False
        If true, the field of all sources is summed up.

    Specific kwargs
    ---------------
    niter: int, default=50
        for Cylinder sources diametral iteration (Simpsons formula).

    Returns
    -------
    B-field: ndarray, shape (L, M, K, N1, N2, ..., 3), unit [kA/m]
        B-field of each source at each path position for each sensor and each sensor pixel
        position in units of mT.
        Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
        single sensor) is removed.

    Info
    ----
    This function automatically joins all sensor position inputs together and groups similar
    sources for optimal vectorization of the computation. For maximal performance call this
    function as little as possible, do not use it in a loop if not absolutely necessary.
    """
    sens = mag3.Sensor(pos_pix=pos_obs)
    return getBH_level2(False, sources, sens, sumup, **specs)

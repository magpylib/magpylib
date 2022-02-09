from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2


# ON INTERFACE
def getB(sources=None, observers=None, sumup=False, squeeze=True, **kwargs):
    """
    Compute B-field in [mT] for given sources and observers.

    - Object-oriented (default):
        ``sources`` are previously defined Magpylib objects or list thereof

    - Direct interface:
        Field implementations can be directly accessed for faster computation. Note that ``sources``
        parameter only accepts a single source and corresponding input parameters must be defined
        via keyword arguments (see Other parameters).

    Parameters
    ----------
    sources: source object, Collection or 1D list thereof
        Sources can be a single source object, a Collection or a 1D list of L source
        objects and/or collections.

        In direct interface mode, must be either 'Cuboid', 'Cylinder',
        'Cylinder_old', 'Sphere', 'Dipole', 'Loop' or 'Line'. and other parameters depending on
        source type must be specified.

    observers: array_like or Sensor or 1D list thereof
        Observers can be array_like positions of shape (N1, N2, ..., 3) where the field
        should be evaluated, can be a Sensor object with pixel shape (N1, N2, ..., 3) or
        a 1D list of K Sensor objects with similar pixel shape. All positions are given
        in units of [mm].

        In direct interface mode, array_like, shape (3,) or (N,3)
        Observer positions in units of [mm].

    sumup: bool, default=False
        If True, the fields of all sources are summed up.

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    Other Parameters
    ----------------
    position: array_like, shape (3,) or (N,3), default=(0,0,0)
        Source positions in units of [mm].

    orientation: scipy Rotation object, default=unit rotation
        Source rotations relative to the initial state (see object docstrings).

    magnetization: array_like, shape (3,) or (N,3)
        Only `source_type in ('Cuboid', 'Cylinder', 'Sphere')`! Magnetization vector (mu0*M) or
        remanence field of homogeneous magnet magnetization in units of [mT].

    moment:  array_like, shape (3,) or (N,3)
        Only `source_type = 'Moment'`! Magnetic dipole moment in units of [mT*mm^3]. For
        homogeneous magnets the relation is moment = magnetization*volume.

    current: array_like, shape (N,)
        Only `source_type in ('Line', 'Loop')`! Current flowing in loop in units of [A].

    dimension: array_like
        Only `source_type in ('Cuboid', 'Cylinder', 'CylinderSegment')`! Magnet dimension
        input in units of [mm]. Dimension format of sources similar as in object oriented
        interface.

    diameter: array_like, shape (N)
        Only `source_type in (Sphere, Loop)`! Diameter of source in units of [mm].

    segment_start: array_like, shape (N,3)
        Only `source_type = 'Line'`! Start positions of line current segments in units of [mm].

    segment_end: array_like, shape (N,3)
        Only `source_type = 'Line'`! End positions of line current segments in units of [mm].

    Returns
    -------
    B-field: ndarray, shape squeeze(L, M, K, N1, N2, ..., 3)
        B-field of each source (L) at each path position (M) for each sensor (K) and each
        sensor pixel position (N1,N2,...) in units of [mT]. Sensor pixel positions are
        equivalent to simple observer positions. Paths of objects that are shorter than
        M will be considered as static beyond their end.

    Notes
    -----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.

    "Static" inputs of shape (x,) will automatically be tiled to shape (N,x) to
    fit with other inputs.

    Examples
    --------
    Compute the B-field of a spherical magnet at a sensor positioned at (1,2,3):

    >>> import magpylib as magpy
    >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
    >>> sensor = magpy.Sensor(position=(1,2,3))
    >>> B = magpy.getB(source, sensor)
    >>> print(B)
    [-0.62497314  0.34089444  0.51134166]

    Compute the B-field of a spherical magnet at five path positions as seen
    by an observer at position (1,2,3):

    >>> import magpylib as magpy
    >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
    >>> source.move([(x,0,0) for x in [1,2,3,4,5]])
    >>> B = magpy.getB(source, (1,2,3))
    >>> print(B)
    [[-0.88894262  0.          0.        ]
     [-0.62497314 -0.34089444 -0.51134166]
     [-0.17483825 -0.41961181 -0.62941771]
     [ 0.09177028 -0.33037301 -0.49555952]
     [ 0.17480239 -0.22080302 -0.33120453]]

    Compute the B-field of two sources at two observer positions, with and without
    sumup:

    >>> import magpylib as magpy
    >>> src1 = magpy.current.Loop(current=15, diameter=2)
    >>> src2 = magpy.misc.Dipole(moment=(100,100,100))
    >>> obs_pos = [(1,1,1), (1,2,3)]
    >>> B = magpy.getB([src1,src2], obs_pos)
    >>> print(B)
    [[[0.93539608 0.93539608 0.40046672]
      [0.05387784 0.10775569 0.0872515 ]]
     [[3.06293831 3.06293831 3.06293831]
      [0.04340403 0.23872216 0.43404028]]]
    >>> B = magpy.getB([src1,src2], obs_pos, sumup=True)
    >>> print(B)
    [[3.99833439 3.99833439 3.46340502]
     [0.09728187 0.34647784 0.52129178]]
    """
    return getBH_level2(sources, observers, sumup=sumup, squeeze=squeeze, field='B', **kwargs)


# ON INTERFACE
def getH(sources=None, observers=None, sumup=False, squeeze=True, **kwargs):
    """
    Compute H-field in [kA/m] for given sources and observers.

    - Object-oriented (default):
        ``sources`` are previously defined Magpylib objects or list thereof

    - Direct interface:
        Field implementations can be directly accessed for faster computation. Note that ``sources``
        parameter only accepts a single source and corresponding input parameters must be defined
        via keyword arguments (see Other parameters).

    Parameters
    ----------
    sources: source object, Collection or 1D list thereof
        Sources can be a single source object, a Collection or a 1D list of L source
        objects and/or collections.

        In direct interface mode, must be either 'Cuboid', 'Cylinder',
        'Cylinder_old', 'Sphere', 'Dipole', 'Loop' or 'Line'. and other parameters depending on
        source type must be specified.

    observers: array_like or Sensor or 1D list thereof
        Observers can be array_like positions of shape (N1, N2, ..., 3) where the field
        should be evaluated, can be a Sensor object with pixel shape (N1, N2, ..., 3) or
        a 1D list of K Sensor objects with similar pixel shape. All positions are given
        in units of [mm].

        In direct interface mode, array_like, shape (3,) or (N,3)
        Observer positions in units of [mm].

    sumup: bool, default=False
        If True, the fields of all sources are summed up.

    squeeze: bool, default=True
        If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    Other Parameters
    ----------------
    position: array_like, shape (3,) or (N,3), default=(0,0,0)
        Source positions in units of [mm].

    orientation: scipy Rotation object, default=unit rotation
        Source rotations relative to the initial state (see object docstrings).

    magnetization: array_like, shape (3,) or (N,3)
        Only `source_type in ('Cuboid', 'Cylinder', 'Sphere')`! Magnetization vector (mu0*M) or
        remanence field of homogeneous magnet magnetization in units of [mT].

    moment:  array_like, shape (3,) or (N,3)
        Only `source_type = 'Moment'`! Magnetic dipole moment in units of [mT*mm^3]. For
        homogeneous magnets the relation is moment = magnetization*volume.

    current: array_like, shape (N,)
        Only `source_type in ('Line', 'Loop')`! Current flowing in loop in units of [A].

    dimension: array_like
        Only `source_type in ('Cuboid', 'Cylinder', 'CylinderSegment')`! Magnet dimension
        input in units of [mm]. Dimension format of sources similar as in object oriented
        interface.

    diameter: array_like, shape (N)
        Only `source_type in (Sphere, Loop)`! Diameter of source in units of [mm].

    segment_start: array_like, shape (N,3)
        Only `source_type = 'Line'`! Start positions of line current segments in units of [mm].

    segment_end: array_like, shape (N,3)
        Only `source_type = 'Line'`! End positions of line current segments in units of [mm].

    Returns
    -------
    H-field: ndarray, shape squeeze(L, M, K, N1, N2, ..., 3)
        H-field of each source (L) at each path position (M) for each sensor (K) and each
        sensor pixel position (N1,N2,...) in units of [kA/m]. Sensor pixel positions are
        equivalent to simple observer positions. Paths of objects that are shorter than
        M will be considered as static beyond their end.

    Notes
    -----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.

    "Static" inputs of shape (x,) will automatically be tiled to shape (N,x) to
    fit with other inputs.

    Examples
    --------
    Compute the H-field of a spherical magnet at a sensor positioned at (1,2,3):

    >>> import magpylib as magpy
    >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
    >>> sensor = magpy.Sensor(position=(1,2,3))
    >>> H = magpy.getH(source, sensor)
    >>> print(H)
    [-0.49733782  0.27127518  0.40691277]

    Compute the H-field of a spherical magnet at five path positions as seen
    by an observer at position (1,2,3):

    >>> import magpylib as magpy
    >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
    >>> source.move([(x,0,0) for x in [1,2,3,4,5]])
    >>> H = magpy.getH(source, (1,2,3))
    >>> print(H)
    [[-0.70739806  0.          0.        ]
     [-0.49733782 -0.27127518 -0.40691277]
     [-0.13913186 -0.33391647 -0.5008747 ]
     [ 0.07302847 -0.26290249 -0.39435373]
     [ 0.13910332 -0.17570946 -0.26356419]]

    Compute the H-field of two sources at two observer positions, with and without
    sumup:

    >>> import magpylib as magpy
    >>> src1 = magpy.current.Loop(current=15, diameter=2)
    >>> src2 = magpy.misc.Dipole(moment=(100,100,100))
    >>> obs_pos = [(1,1,1), (1,2,3)]
    >>> H = magpy.getH([src1,src2], obs_pos)
    >>> print(H)
    [[[0.74436455 0.74436455 0.31868129]
      [0.04287463 0.08574925 0.06943254]]
     [[2.43740886 2.43740886 2.43740886]
      [0.03453983 0.18996906 0.34539828]]]
    >>> H = magpy.getH([src1,src2], obs_pos, sumup=True)
    >>> print(H)
    [[3.18177341 3.18177341 2.75609015]
     [0.07741445 0.27571831 0.41483082]]
    """
    return getBH_level2(sources, observers, sumup=sumup, squeeze=squeeze, field='H', **kwargs)

from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2


def getB(
    sources=None, observers=None, sumup=False, squeeze=True, pixel_agg=None, **kwargs
):
    """Compute B-field in [mT] for given sources and observers.

    Field implementations can be directly accessed (avoiding the object oriented
    Magpylib interface) by providing a string input `sources=source_type`, array_like
    positions as `observers` input, and all other necessary input parameters (see below)
    as kwargs.

    Parameters
    ----------
    sources: source and collection objects or 1D list thereof
        Sources that generate the magnetic field. Can be a single source (or collection)
        or a 1D list of l source and/or collection objects.

        Direct interface: input must be one of (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`,
        `'Sphere'`, `'Dipole'`, `'Loop'` or `'Line'`).

    observers: array_like or (list of) `Sensor` objects
        Can be array_like positions of shape (n1, n2, ..., 3) where the field
        should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
        of such sensor objects (must all have similar pixel shapes). All positions
        are given in units of [mm].

        Direct interface: Input must be array_like with shape (3,) or (n,3) corresponding
        positions to observer positions in units of [mm].

    sumup: bool, default=`False`
        If `True`, the fields of all sources are summed up.

    squeeze: bool, default=`True`
        If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    pixel_agg: str, default=`None`
        Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
        which is applied to observer output values, e.g. mean of all sensor pixel outputs.
        With this option, observers input with different (pixel) shapes is allowed.

    Other Parameters (Direct interface)
    -----------------------------------
    position: array_like, shape (3,) or (n,3), default=`(0,0,0)`
        Source position(s) in the global coordinates in units of [mm].

    orientation: scipy `Rotation` object with length 1 or n, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation.

    magnetization: array_like, shape (3,) or (n,3)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`, `'Sphere'`)!
        Magnetization vector(s) (mu0*M, remanence field) in units of [kA/m] given in
        the local object coordinates (rotates with object).

    moment: array_like, shape (3) or (n,3), unit [mT*mm^3]
        Only source_type == `'Dipole'`!
        Magnetic dipole moment(s) in units of [mT*mm^3] given in the local object coordinates
        (rotates with object). For homogeneous magnets the relation moment=magnetization*volume
        holds.

    current: array_like, shape (n,)
        Only source_type == `'Loop'` or `'Line'`!
        Electrical current in units of [A].

    dimension: array_like, shape (x,) or (n,x)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`)!
        Magnet dimension input in units of [mm] and [deg]. Dimension format x of sources is similar
        as in object oriented interface.

    diameter: array_like, shape (n,)
        Only source_type == `'Sphere'` or `'Loop'`!
        Diameter of source in units of [mm].

    segment_start: array_like, shape (n,3)
        Only source_type == `'Line'`!
        Start positions of line current segments in units of [mm].

    segment_end: array_like, shape (n,3)
        Only source_type == `'Line'`!
        End positions of line current segments in units of [mm].

    Returns
    -------
    B-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3)
        B-field at each path position (m) for each sensor (k) and each sensor pixel
        position (n1, n2, ...) in units of [mT]. Sensor pixel positions are equivalent
        to simple observer positions. Paths of objects that are shorter than m will be
        considered as static beyond their end.

    Direct interface: ndarray, shape (n,3)
        B-field for every parameter set in units of [mT].

    Notes
    -----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.

    Examples
    --------
    In this example we compute the B-field [mT] of a spherical magnet and a current loop
    at the observer position (1,1,1) given in units of [mm]:

    >>> import magpylib as magpy
    >>> src1 = magpy.current.Loop(current=100, diameter=2)
    >>> src2 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1)
    >>> B = magpy.getB([src1, src2], (1,1,1))
    >>> print(B)
    [[6.23597388e+00 6.23597388e+00 2.66977810e+00]
     [8.01875374e-01 8.01875374e-01 1.48029737e-16]]

    We can also use sensor objects as observers input:

    >>> sens1 = magpy.Sensor(position=(1,1,1))
    >>> sens2 = sens1.copy(position=(1,1,-1))
    >>> B = magpy.getB([src1, src2], [sens1, sens2])
    >>> print(B)
    [[[ 6.23597388e+00  6.23597388e+00  2.66977810e+00]
      [-6.23597388e+00 -6.23597388e+00  2.66977810e+00]]
    <BLANKLINE>
     [[ 8.01875374e-01  8.01875374e-01  1.48029737e-16]
      [-8.01875374e-01 -8.01875374e-01  1.48029737e-16]]]

    Through the direct interface we can compute the same fields for the loop as:

    >>> obs = [(1,1,1), (1,1,-1)]
    >>> B = magpy.getB('Loop', obs, current=100, diameter=2)
    >>> print(B)
    [[ 6.23597388  6.23597388  2.6697781 ]
     [-6.23597388 -6.23597388  2.6697781 ]]

    But also for a set of four completely different instances:

    >>> B = magpy.getB(
    ...     'Loop',
    ...     observers=((1,1,1), (1,1,-1), (1,2,3), (2,2,2)),
    ...     current=(11, 22, 33, 44),
    ...     diameter=(1, 2, 3, 4),
    ...     position=((0,0,0), (0,0,1), (0,0,2), (0,0,3)),
    ... )
    >>> print(B)
    [[ 0.17111325  0.17111325  0.01705189]
     [-0.38852048 -0.38852048  0.49400758]
     [ 1.14713551  2.29427102 -0.22065346]
     [-2.48213467 -2.48213467 -0.79683487]]
    """
    return getBH_level2(
        sources,
        observers,
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        field="B",
        **kwargs
    )


def getH(
    sources=None, observers=None, sumup=False, squeeze=True, pixel_agg=None, **kwargs
):
    """Compute H-field in [kA/m] for given sources and observers.

    Field implementations can be directly accessed (avoiding the object oriented
    Magpylib interface) by providing a string input `sources=source_type`, array_like
    positions as `observers` input, and all other necessary input parameters (see below)
    as kwargs.

    Parameters
    ----------
    sources: source and collection objects or 1D list thereof
        Sources that generate the magnetic field. Can be a single source (or collection)
        or a 1D list of l source and/or collection objects.

        Direct interface: input must be one of (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`,
        `'Sphere'`, `'Dipole'`, `'Loop'` or `'Line'`).

    observers: array_like or (list of) `Sensor` objects
        Can be array_like positions of shape (n1, n2, ..., 3) where the field
        should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
        of such sensor objects (must all have similar pixel shapes). All positions
        are given in units of [mm].

        Direct interface: Input must be array_like with shape (3,) or (n,3) corresponding
        positions to observer positions in units of [mm].

    sumup: bool, default=`False`
        If `True`, the fields of all sources are summed up.

    squeeze: bool, default=`True`
        If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
        a single sensor or only a single source) are eliminated.

    pixel_agg: str, default=`None`
        Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
        which is applied to observer output values, e.g. mean of all sensor pixel outputs.
        With this option, observer inputs with different (pixel) shapes are allowed.

    Other Parameters (Direct interface)
    -----------------------------------
    position: array_like, shape (3,) or (n,3), default=`(0,0,0)`
        Source position(s) in the global coordinates in units of [mm].

    orientation: scipy `Rotation` object with length 1 or n, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation.

    magnetization: array_like, shape (3,) or (n,3)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`, `'Sphere'`)!
        Magnetization vector(s) (mu0*M, remanence field) in units of [kA/m] given in
        the local object coordinates (rotates with object).

    moment: array_like, shape (3) or (n,3), unit [mT*mm^3]
        Only source_type == `'Dipole'`!
        Magnetic dipole moment(s) in units of [mT*mm^3] given in the local object coordinates
        (rotates with object). For homogeneous magnets the relation moment=magnetization*volume
        holds.

    current: array_like, shape (n,)
        Only source_type == `'Loop'` or `'Line'`!
        Electrical current in units of [A].

    dimension: array_like, shape (x,) or (n,x)
        Only source_type in (`'Cuboid'`, `'Cylinder'`, `'CylinderSegment'`)!
        Magnet dimension input in units of [mm] and [deg]. Dimension format x of sources is similar
        as in object oriented interface.

    diameter: array_like, shape (n,)
        Only source_type == `'Sphere'` or `'Loop'`!
        Diameter of source in units of [mm].

    segment_start: array_like, shape (n,3)
        Only source_type == `'Line'`!
        Start positions of line current segments in units of [mm].

    segment_end: array_like, shape (n,3)
        Only source_type == `'Line'`!
        End positions of line current segments in units of [mm].

    Returns
    -------
    H-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3)
        H-field at each path position (m) for each sensor (k) and each sensor pixel
        position (n1, n2, ...) in units of [kA/m]. Sensor pixel positions are equivalent
        to simple observer positions. Paths of objects that are shorter than m will be
        considered as static beyond their end.

    Direct interface: ndarray, shape (n,3)
        H-field for every parameter set in units of [kA/m].

    Notes
    -----
    This function automatically joins all sensor and position inputs together and groups
    similar sources for optimal vectorization of the computation. For maximal performance
    call this function as little as possible and avoid using it in loops.

    Examples
    --------
    In this example we compute the H-field [kA/m] of a spherical magnet and a current loop
    at the observer position (1,1,1) given in units of [mm]:

    >>> import magpylib as magpy
    >>> src1 = magpy.current.Loop(current=100, diameter=2)
    >>> src2 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1)
    >>> H = magpy.getH([src1, src2], (1,1,1))
    >>> print(H)
    [[4.96243034e+00 4.96243034e+00 2.12454191e+00]
     [6.38112147e-01 6.38112147e-01 1.17798322e-16]]

    We can also use sensor objects as observers input:

    >>> sens1 = magpy.Sensor(position=(1,1,1))
    >>> sens2 = sens1.copy(position=(1,1,-1))
    >>> H = magpy.getH([src1, src2], [sens1, sens2])
    >>> print(H)
    [[[ 4.96243034e+00  4.96243034e+00  2.12454191e+00]
      [-4.96243034e+00 -4.96243034e+00  2.12454191e+00]]
    <BLANKLINE>
     [[ 6.38112147e-01  6.38112147e-01  1.17798322e-16]
      [-6.38112147e-01 -6.38112147e-01  1.17798322e-16]]]

    Through the direct interface we can compute the same fields for the loop as:

    >>> obs = [(1,1,1), (1,1,-1)]
    >>> H = magpy.getH('Loop', obs, current=100, diameter=2)
    >>> print(H)
    [[ 4.96243034  4.96243034  2.12454191]
     [-4.96243034 -4.96243034  2.12454191]]

    But also for a set of four completely different instances:

    >>> H = magpy.getH(
    ...     'Loop',
    ...     observers=((1,1,1), (1,1,-1), (1,2,3), (2,2,2)),
    ...     current=(11, 22, 33, 44),
    ...     diameter=(1, 2, 3, 4),
    ...     position=((0,0,0), (0,0,1), (0,0,2), (0,0,3)),
    ... )
    >>> print(H)
    [[ 0.1361676   0.1361676   0.01356947]
     [-0.30917477 -0.30917477  0.39311875]
     [ 0.91286143  1.82572286 -0.17559045]
     [-1.97522001 -1.97522001 -0.63410104]]
    """
    return getBH_level2(
        sources,
        observers,
        sumup=sumup,
        squeeze=squeeze,
        pixel_agg=pixel_agg,
        field="H",
        **kwargs
    )

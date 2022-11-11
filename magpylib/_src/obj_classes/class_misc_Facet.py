"""Magnet Facet class code
"""

import numpy as np
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.fields.field_BH_facet import magnet_facet_field_from_obj
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.display.traces_generic import make_Facet


class Facet(BaseMagnet):
    """Triangular surface mesh magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the TriangularMesh facets coordinates
    are the same as in the global coordinate system. The geometric center of the TriangularMesh
    is determined by its vertices and is not necessarily located in the origin.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local object coordinates (rotates with object).

    facets: ndarray, shape (n,4,3)
        Facets in the relative coordinate system of the TriangularMesh object.

    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting facets, the initial position is set to the barycenter.

    barycenter: array_like, shape (3,)
        Read only property that returns the geometric barycenter (=center of mass)
        of the object.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    magnet source: `TriangularMesh` object

    Examples
    --------
    """

    _field_func = staticmethod(magnet_facet_field_from_obj)
    _field_func_kwargs_ndim = {"magnetization": 2, "facets": 3}
    _draw_func = make_Facet

    def __init__(
        self,
        magnetization=None,
        facets=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        self.facets = facets

        # init inheritance
        super().__init__(position, orientation, magnetization, style, **kwargs)

    # property getters and setters
    @property
    def facets(self):
        """Facets objects"""
        return self._facets

    @facets.setter
    def facets(self, val):
        """Set Facets facets (a,b,c), shape (3,), [mm]."""
        self._facets = check_format_input_vector(
            val,
            dims=(2,3),
            shape_m1=3,
            sig_name="Facet.facets",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3,3)",
            reshape=(-1,3,3),
            allow_None=True,
        )

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self._facets)

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @staticmethod
    def _get_barycenter(position, orientation, facets):
        """Returns the barycenter of a tetrahedron."""
        centroid = np.mean(facets, axis=(0,1))
        barycenter = orientation.apply(centroid) + position
        return barycenter

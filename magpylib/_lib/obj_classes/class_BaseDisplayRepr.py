"""BaseGeo class code"""

from magpylib._lib.display import display

# ALL METHODS ON INTERFACE
class BaseDisplayRepr:
    """ Provides the display(self) and self.repr methods for all objects

    Properties
    ----------

    Methods
    -------
    - display(self, **kwargs)
    - repr
    """
    def __init__(self):
        self._object_type = None

    # ------------------------------------------------------------------
    # INTERFACE
    def display(
        self,
        markers=[(0,0,0)],
        axis=None,
        show_direction=False,
        show_path=True,
        size_sensors=1,
        size_direction=1,
        size_dipoles=1,
        zoom = 0.5,
        plotting_backend=None,
        **kwargs):
        """
        Display object graphically.

        Parameters
        ----------
        markers: array_like, shape (N,3), default=[(0,0,0)]
            Display position markers in the global CS. By default a marker is placed
            in the origin.

        axis: pyplot.axis, default=None
            Display graphical output in a given pyplot axis (must be 3D). By default a new
            pyplot figure is created and displayed.

        show_direction: bool, default=False
            Set True to show magnetization and current directions.

        show_path: bool or int, default=True
            Options True, False, positive int. By default object paths are shown. If
            show_path is a positive integer, objects will be displayed at multiple path
            positions along the path, in steps of show_path.

        size_sensor: float, default=1
            Adjust automatic display size of sensors.

        size_direction: float, default=1
            Adjust automatic display size of direction arrows.

        size_dipoles: float, default=1
            Adjust automatic display size of dipoles.
            define plotting backend

<<<<<<< HEAD
        zoom: float, default = 0.5
            Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.

=======
>>>>>>> 532ee543e3d8a0b385ef8cbfee88317255a0a37f
        plotting_backend: default=None
            One of 'matplotlib', 'plolty'. If not set, parameter will default to 
            Config.PLOTTING_BACKEND
            
        Returns
        -------
        None: NoneType

        Examples
        --------

        Display Magpylib objects graphically using Matplotlib:

        >>> import magpylib as magpy
        >>> obj = magpy.magnet.Sphere(magnetization=(0,0,1), diameter=1)
        >>> obj.move([(.2,0,0)]*50, increment=True)
        >>> obj.rotate_from_angax(angle=[10]*50, axis='z', anchor=0, start=0, increment=True)
        >>> obj.display(show_direction=True, show_path=10)
        --> graphic output

        Display figure on your own 3D Matplotlib axis:

        >>> import matplotlib.pyplot as plt
        >>> import magpylib as magpy
        >>> my_axis = plt.axes(projection='3d')
        >>> obj = magpy.magnet.Cuboid(magnetization=(0,0,1), dimension=(1,2,3))
        >>> obj.move([(x,0,0) for x in [0,1,2,3,4,5]])
        >>> obj.display(axis=my_axis)
        >>> plt.show()
        --> graphic output

        """
        #pylint: disable=dangerous-default-value
        display(
            self,
            markers=markers,
            axis=axis,
            show_direction=show_direction,
            show_path=show_path,
            size_direction=size_direction,
            size_sensors=size_sensors,
            size_dipoles=size_dipoles,
            zoom=zoom,
            plotting_backend=plotting_backend,
            **kwargs)

    # ------------------------------------------------------------------
    # INTERFACE
    def __repr__(self) -> str:
        return f'{self._object_type}(id={str(id(self))})'

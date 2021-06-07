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
        self.obj_type = None

    # ------------------------------------------------------------------
    # INTERFACE
    def display(
        self,
        markers=[(0,0,0)],
        axis=None,
        direc=False,
        show_path=True,
        size_sensors=1,
        size_direc=1):
        """
        Display objects and paths graphically using matplotlib 3D.

        Parameters
        ----------
        objects: sources, collections or sensors
            Show a 3D reprensation of given objects in matplotlib.

        markers: array_like, shape (N,3), default=[(0,0,0)]
            Display position markers in the global CS. By default a marker is in the origin.

        axis: pyplot.axis, default=None
            Display graphical output in a given pyplot axis (must be 3D). By default a new
            pyplot figure is created and displayed.

        direc: bool, default=False
            Set True to show magnetization and current directions.

        show_path: bool or int, default=True
            Options True, False, positive int. By default object paths are shown. If
            show_path is a positive integer, objects will be displayed at each path
            position in steps of show_path.

        size_sensor: float, default=1
            Adjust automatic display size of sensors.

        size_direc: float, default=1
            Adjust automatic display size of direction arrows

        Returns
        -------
        None
        """
        #pylint: disable=dangerous-default-value
        display(
            self,
            markers=markers,
            axis=axis,
            direc=direc,
            show_path=show_path,
            size_direc=size_direc,
            size_sensors=size_sensors)

    # ------------------------------------------------------------------
    # INTERFACE
    def __repr__(self) -> str:
        return f'{self.obj_type}(id={str(id(self))})'

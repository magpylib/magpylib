"""Base class for objects that can be targets of force computation."""

from abc import ABC, abstractmethod

from magpylib._src.fields.field_FT import getFT_magnet, getFT_dipole, getFT_current

# Registry of valid force functions
VALID_FORCE_FUNCTIONS = [
    getFT_magnet,
    getFT_dipole,
    getFT_current,
]

class BaseTarget(ABC):
    """Base class for Magpylib objects that can be targets of force computation.

    Parameters
    ----------
    meshing : dict
        Parameters that define the mesh fineness for force computation.
        Should contain mesh-specific parameters like resolution, method, etc.

    Properties
    ----------
    meshing : dict
        Mesh parameters for force computation targets.

    Methods
    -------
    _generate_mesh()
        Abstract method to generate mesh for force computation.
        Must be implemented by subclasses.
    """
    # This must be set by subclasses to one of the VALID_FORCE_FUNCTIONS values
    _force_func = None

    def __init__(self, meshing=None):
        """Initialize BaseTarget with meshing parameters."""
        self._meshing = meshing

        # Validate that subclass has set a valid _force_func
        if self._force_func is None:
            msg = f"Missing force function implementation in subclass of {self}"
            raise NotImplementedError(msg)
        
        # Get the underlying function (handle both functions and bound methods)
        #force_func = self._force_func
        #if hasattr(force_func, '__func__'):  # It's a bound method
        #    force_func = force_func.__func__
        if self._force_func.__func__ not in VALID_FORCE_FUNCTIONS:
            msg = f"Bad force function defined in {self}."
            raise ValueError(msg)

    @property
    def meshing(self):
        """Get mesh parameters for force computation."""
        return self._meshing

    @meshing.setter
    def meshing(self, value):
        """Set mesh parameters for force computation."""
        # meshing input checks happens at class level because
        # each class may have different requirements
        self._meshing = value

    @abstractmethod
    def _generate_mesh(self):
        """Generate mesh of shape (n,3) for force computation.
        
        Returns
        -------
        mesh : np.ndarray, shape (n, 3)
            Mesh points or elements for force computation.
            
        Notes
        -----
        This method must be implemented by subclasses to define how
        the object should be meshed for force calculations.
        """
        pass

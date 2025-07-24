"""Base class for objects that can be targets of force computation."""

from abc import ABC, abstractmethod

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
    _get_mesh()
        Abstract method to generate mesh for force computation.
        Must be implemented by subclasses.
    """
    def __init__(self, meshing=None):
        """Initialize BaseTarget with meshing parameters."""
        self._meshing = meshing

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

# %% IMPORTS
from numpy import array, float64, isnan, ndarray, pi
from magpylib._lib.utility import (initializeMulticorePool, isPosVector,
                                   recoordinateAndGetB)
from magpylib._lib.mathLibPrivate import (Qconj, Qmult, arccosSTABLE,
                                          fastNorm3D, fastSum3D, getRotQuat)
from typing import Tuple
from multiprocessing import Pool, cpu_count
from itertools import product, repeat
import sys

# Type hint definitions
# These aren't type hints, but look good
# in Spyder IDE. Pycharm recognizes it.
Auto = 0  # Maximum cores, for multicore function. if 0 find max.
numpyArray = 0
constant = None
#######################################


class FieldSampler:
    def _getBmultiList(self, listOfArgs, processes=Auto):
        # Used in getBsweep() For lists of positions for B field samples to be
        # calculated in parallel. Return a list of calculated B field samples.
        pool = initializeMulticorePool(processes)
        results = pool.map(self.getB, listOfArgs)
        pool.close()
        pool.join()
        return results

    def _getBDisplacement(self, listOfArgs, processes=Auto):
        # Used in getBparallel() For lists of arguments where first argument is
        # a position for a B field sample Second argument is the magnet's
        # absolute position vector Third argument is a tuple of the magnet's
        # absolute rotation arguments
        pool = initializeMulticorePool(processes)
        results = pool.starmap(recoordinateAndGetB,
                               zip(repeat(self,
                                          times=len(listOfArgs)),
                                   listOfArgs))
        pool.close()
        pool.join()
        return results

    def getBsweep(self, INPUT, multiprocessing=False, processes=Auto):
        """
        Advanced input for advanced people who want advanced results.

        Enter a list of positions to calculate field samples in a parallelized
        environment. Alternatively, enter a list of lists - where each list in
        the list each contain a field sample position vector in the first
        index, an absolute magnet position vector in the 2nd index, and an
        orientation argument tuple where the first index is an angle scalar and
        the second index is an axis (also a tuple). You can also add a third
        index position for the anchor if you really want to.

        The possibilities are only limited by your imagination plus the number
        of CPU cores.


        Parameters
        ----------
        INPUT : [list of vec3] or [list of [Bfield Sample, Magnet Position,
        Magnet Rotation]]

        Example
        -------

        For carousel simulation:

        >>> from multiprocessing import freeze_support
        >>> if __name__ == "__main__":
        >>>     freeze_support()
        >>>     # Input
        >>>     from magpylib.source import magnet
        >>>     mag=[1,2,3]
        >>>     dim=[1,2,3]
        >>>     pos=[0,0,0]
        >>>     listOfArgs = [  [   [1,2,3],        #pos
        ...                         [0,0,1],        #MPos
        ...                         (180,(0,1,0)),],#Morientation
        ...                     [   [1,2,3],
        ...                         [0,1,0],
        ...                         (90,(1,0,0)),],
        ...                     [   [1,2,3],
        ...                         [1,0,0],
        ...                         (255,(0,1,0)),],]
        >>>     # Run
        >>>     pm = magnet.Box(mag,dim,pos)
        >>>     result = pm.getBsweep(listOfArgs)
        >>>     print(result)
                ( [ 0.00453617, -0.07055326,  0.03153698],
                [0.00488989, 0.04731373, 0.02416068],
                [0.0249435,  0.00106315, 0.02894469])

        For parallel field list calculation:

        >>> from multiprocessing import freeze_support
        >>> if __name__ == "__main__":
        >>>     freeze_support()
        >>>     # Input
        >>>     from magpylib.source import magnet
        >>>     mag=[6,7,8]
        >>>     dim=[10,10,10]
        >>>     pos=[2,2,2]
        >>>     listOfPos = [[.5,.5,5],[.5,.5,5],[.5,.5,5]]
        >>>     # Run
        >>>     pm = magnet.Box(mag,dim,pos)
        >>>     result = pm.getBsweep(listOfPos)
        >>>     print(result)
                (   [3.99074612, 4.67238469, 4.22419432],
                    [3.99074612, 4.67238469, 4.22419432],
                    [3.99074612, 4.67238469, 4.22419432],)

        """

        if multiprocessing is True:
            if all(isPosVector(item) for item in INPUT):
                return self._getBmultiList(INPUT, processes=processes)
            else:
                return self._getBDisplacement(INPUT, processes=processes)
        else:
            if all(isPosVector(item) for item in INPUT):
                return list(map(self.getB, INPUT))
            else:
                return list(map(recoordinateAndGetB,
                                repeat(self, times=len(INPUT)), INPUT))

    def getB(self, pos):
        """
        This method returns the magnetic field vector generated by the source
        at the argument position `pos` in units of [mT]

        Parameters
        ----------
        pos : vec3 [mm] Position or list of Positions where magnetic field
            should be determined.


        Returns
        -------
        magnetic field vector : arr3 [mT] Magnetic field at the argument
            position `pos` generated by the source in units of [mT].
        """
        # Return a list of vec3 results
        # This method will be overriden by the classes that inherit it.
        # Throw a warning and return 0s if it somehow isn't.
        # Note: Collection() has its own docstring
        # for getB since it inherits nothing.
        import warnings
        warnings.warn(
            "called getB method is not implemented in this class,"
            "returning [0,0,0]", RuntimeWarning)
        return [0, 0, 0]

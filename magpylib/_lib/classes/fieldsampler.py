# -------------------------------------------------------------------------------
# magpylib -- A Python 3 toolbox for working with magnetic fields.
# Copyright (C) Silicon Austria Labs, https://silicon-austria-labs.com/,
#               Michael Ortner <magpylib@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.
# The acceptance of the conditions of the GNU Affero General Public License are
# compulsory for the usage of the software.
#
# For contact information, reach out over at <magpylib@gmail.com> or our issues
# page at https://www.github.com/magpylib/magpylib/issues.
# -------------------------------------------------------------------------------
# %% IMPORTS
from itertools import repeat
from numpy import array
from magpylib._lib.utility import (initializeMulticorePool, isPosVector,
                                   recoordinateAndGetB)
# Type hint definitions
# These aren't type hints, but look good
# in Spyder IDE. Pycharm recognizes it.
Auto = 0  # Maximum cores, for multicore function. if 0 find max.
numpyArray = 0
constant = None
#######################################


class FieldSampler:
    """Field Sampler Class

    This class keeps track of all the methods for interfacing between Field
    calculation methods and the user.

    All source classes will inherit this, so non-implemented methods for some
    objects that are here should throw a warning if called.

    Returns
    -------
    [FieldSampler]
        [Interfaces for Field calculations]
    """

    def _getBmultiList(self, listOfArgs, processes=Auto):
        # Used in getBsweep() For lists of positions for B field samples to be
        # calculated in parallel. Return a list of calculated B field samples.
        pool = initializeMulticorePool(processes)
        results = pool.map(self.getB, listOfArgs)
        pool.close()
        pool.join()
        return array(results)

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
        return array(results)

    def getBsweep(self, INPUT, multiprocessing=False, processes=Auto):
        """
        This method can be used to determine the field for a given set
        of sample positions, or for different magnet positions and orientations.
        While this can manually be achieved by looping getB, this getBsweep
        implements the possibility of parallelization. 

        The input can have two different formats (ONLY THE FIRST ONE CAN BE
        USED FOR COLLECTIONS!):

        Warning
        -------
           Multiprocessing enabled calculations will drastically reduce performance if done for small sets, i.e. under a few hundred.

        Parameters
        ----------
        INPUT : TYPE [type 1 or type 2 input] 
           INPUT TYPE 1 is a list of N sample positions. In this case the magnetic field of the source is determined for all N sample positions and returned in an Nx3 array. INPUT TYPE 2 is a list of the following format [(samplePos1, sourcePos1, sourceOrient1),…]. Here for each case of sample position and source state the field is evaluated and returned in an Nx3 array. This corresponds to a system where sample and magnet move simultaneously. TYPE 2 DOES NOT WORK FOR COLLECTIONS !

        multiprocessing : bool [bool] Default = False
           Enable/disable parallel multiprocessing; This requires some additional code on Windows, please refer to example below.

        processes : cores [int]
           Define the number of allowed processes when multiprocessing is enabled.

        Example
        -------
        For INPUT of type 1:

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

        For INPUT of type 2:

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


        """

        if multiprocessing is True:
            if all(isPosVector(item) for item in INPUT):
                return self._getBmultiList(INPUT, processes=processes)
            else:
                return self._getBDisplacement(INPUT, processes=processes)
        else:
            if all(isPosVector(item) for item in INPUT):
                return array(list(map(self.getB, INPUT)))
            else:
                return array(list(map(recoordinateAndGetB,
                                repeat(self, times=len(INPUT)), INPUT)))

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

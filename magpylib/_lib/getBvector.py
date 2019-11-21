# -*- coding: utf-8 -*-
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
from magpylib._lib.fields.PM_Box_vector import Bfield_BoxV
from magpylib._lib.fields.PM_Sphere_vector import Bfield_SphereV
from magpylib._lib.mathLibPrivate_vector import QconjV, QrotationV, QmultV, getRotQuatV, angleAxisRotationV
import numpy as np

def getBv(type,MAG,DIM,POSo,POSm,ANG=[],AX=[],ANCH=[]):
    """
    This function applies the vectorized code paradigm native to numpy to provide
    computation performance when multiple magnetic field calculations are 
    performed. Use this function only when performing more than ~10 computations.

    Parameters
    ----------

    type : string
        source type either 'box', 'cylinder', 'sphere', 'line', 'circular', 'dipole'.

    MAG : Nx3 numpy array float [mT]
        vector of N magnetizations.

    DIM : NxY numpy array float [mm]
        vector of N dimensions for each evaluation. The form of this vector depends
        on the source type.

    POSo : Nx3 numpy array float [mm]
        vector of N positions of the observer.
    
    POSm : Nx3 numpy array float [mm]
        vector of N initial source positions. These positions will be adjusted by
        the given rotation parameters.

    ANG=[] : length M list of size N numpy arrays float [deg]
       Angles of M subsequent rotation operations applied to the N-sized POSm and
       the implicit source orientation.
    
    AX=[] : length M list of Nx3 numpy arrays float []
        Axis vectors of M subsequent rotation operations applied to the N-sized
        POSm and the implicit source orientation.
    
    ANCH=[] : length M list of Nx3 numpy arrays float [mm]
        Anchor positions of M subsequent rotations applied ot the N-sized POSm and
        the implicit source orientation.
    """

    N = len(MAG)

    # set field type
    if type == 'box':
        Bfield = Bfield_BoxV
    elif type == 'sphere':
        Bfield = Bfield_SphereV
    else:
        print('Bad type')
        return 0

    Q = np.array([[1,0,0,0]]*N) #initial orientation
    Pm = POSm   #initial position
    #apply rotation operations
    for ANGLE,AXIS,ANCHOR in zip(ANG,AX,ANCH):
        Q = QmultV(getRotQuatV(ANGLE,AXIS),Q)
        Pm = angleAxisRotationV(ANGLE,AXIS,Pm-ANCHOR)+ANCHOR

    #calculate the B-field
    POSrel = POSo-Pm        #relative position
    Qc = QconjV(Q)          #orientierung
    POSrot = QrotationV(Qc,POSrel)  #rotation der pos in das CS der Quelle
    Brot = Bfield(MAG, POSrot, DIM) #feldberechnung
    B = QrotationV(Q,Brot)  #rückrotation des feldes
    
    return B
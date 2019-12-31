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
from magpylib._lib.fields.PM_Cylinder_vector import Bfield_CylinderV
from magpylib._lib.fields.PM_Sphere_vector import Bfield_SphereV
from magpylib._lib.fields.Moment_Dipole_vector import Bfield_DipoleV
from magpylib._lib.fields.Current_CircularLoop_vector import Bfield_CircularCurrentLoopV
from magpylib._lib.fields.Current_Line_vector import Bfield_LineSegmentV
from magpylib._lib.mathLib_vector import QconjV, QrotationV, QmultV, getRotQuatV, angleAxisRotationV_priv
import numpy as np

def getBv_magnet(type,MAG,DIM,POSm,POSo,ANG=[],AX=[],ANCH=[],Nphi0=50):
    """
    Calculate the field of magnets using vectorized performance code.

    Parameters
    ----------

    type : string
        source type either 'box', 'cylinder', 'sphere'.

    MAG : Nx3 numpy array float [mT]
        vector of N magnetizations.

    DIM : NxY numpy array float [mm]
        vector of N dimensions for each evaluation. The form of this vector depends
        on the source type. Y=3/2/1 for box/cylinder/sphere

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
    
    Nphi0=50 : integer gives number of iterations used when calculating diametral
        magnetized cylindrical magnets.
    """

    N = len(POSo)

    Q = np.zeros([N,4])
    Q[:,0] = 1              # init orientation
    
    Pm = POSm               #initial position

    #apply rotation operations
    for ANGLE,AXIS,ANCHOR in zip(ANG,AX,ANCH):
        Q = QmultV(getRotQuatV(ANGLE,AXIS),Q)
        Pm = angleAxisRotationV_priv(ANGLE,AXIS,Pm-ANCHOR)+ANCHOR

    # transform into CS of source
    POSrel = POSo-Pm        #relative position
    Qc = QconjV(Q)          #orientierung
    POSrot = QrotationV(Qc,POSrel)  #rotation der pos in das CS der Quelle
    
    # calculate field
    if type == 'box':
        Brot = Bfield_BoxV(MAG, POSrot, DIM)
    elif type == 'cylinder':
        Brot = Bfield_CylinderV(MAG, POSrot, DIM,Nphi0)
    elif type == 'sphere':
        Brot = Bfield_SphereV(MAG, POSrot, DIM)
    else:
        print('Bad type or WIP')
        return 0
    
    # transform back
    B = QrotationV(Q,Brot)  #rückrotation des feldes
    
    return B

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


def getBv_current(type,CURR,DIM,POSm,POSo,ANG=[],AX=[],ANCH=[]):
    """
    Calculate the field of currents using vectorized performance code.

    Parameters
    ----------

    type : string
        source type either 'circular' or 'line'

    MAG : Nx3 numpy array float [mT]
        vector of N magnetizations.

    DIM : NxY numpy array float [mm]
        vector of N dimensions for each evaluation. The form of this vector depends
        on the source type. Y=1/3x3 for circular/line.

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
    N = len(POSo)

    Q = np.zeros([N,4])
    Q[:,0] = 1              # init orientation
    
    Pm = POSm               #initial position

    #apply rotation operations
    for ANGLE,AXIS,ANCHOR in zip(ANG,AX,ANCH):
        Q = QmultV(getRotQuatV(ANGLE,AXIS),Q)
        Pm = angleAxisRotationV_priv(ANGLE,AXIS,Pm-ANCHOR)+ANCHOR

    # transform into CS of source
    POSrel = POSo-Pm        #relative position
    Qc = QconjV(Q)          #orientierung
    POSrot = QrotationV(Qc,POSrel)  #rotation der pos in das CS der Quelle

    # calculate field
    if type == 'circular':
        Brot = Bfield_CircularCurrentLoopV(CURR, DIM, POSrot)
    elif type == 'line':
        Brot = Bfield_LineSegmentV(POSrot,DIM[:,0],DIM[:,1],CURR)
    else:
        print('Bad type')
        return 0
    
    # transform back
    B = QrotationV(Q,Brot)  #rückrotation des feldes
    
    return B

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def getBv_moment(type,MOM,POSm,POSo,ANG=[],AX=[],ANCH=[]):
    """
    Calculate the field of magnetic moments using vectorized performance code.

    Parameters
    ----------

    type : string
        source type: 'dipole'

    MOM : Nx3 numpy array float [mT]
        vector of N dipole moments.

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

    N = len(POSo)

    Q = np.zeros([N,4])
    Q[:,0] = 1              # init orientation
    
    Pm = POSm               #initial position

    #apply rotation operations
    for ANGLE,AXIS,ANCHOR in zip(ANG,AX,ANCH):
        Q = QmultV(getRotQuatV(ANGLE,AXIS),Q)
        Pm = angleAxisRotationV_priv(ANGLE,AXIS,Pm-ANCHOR)+ANCHOR

    # transform into CS of source
    POSrel = POSo-Pm        #relative position
    Qc = QconjV(Q)          #orientierung
    POSrot = QrotationV(Qc,POSrel)  #rotation der pos in das CS der Quelle

    # calculate field
    if type == 'dipole':
        Brot = Bfield_DipoleV(MOM, POSrot)
    else:
        print('Bad type')
        return 0
    
    # transform back
    B = QrotationV(Q,Brot)  #rückrotation des feldes
    
    return B

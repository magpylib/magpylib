"""_lib.obj_classes"""

__all__ = ['Box', 'Cylinder', 'Sphere', 'Collection', 'Sensor',
    'Dipole', 'Circular']

# create interface to outside of package
from magpylib._lib.obj_classes.class_mag_Box import Box
from magpylib._lib.obj_classes.class_mag_Cylinder import Cylinder
from magpylib._lib.obj_classes.class_Collection import Collection
from magpylib._lib.obj_classes.class_Sensor import Sensor
from magpylib._lib.obj_classes.class_mag_Sphere import Sphere
from magpylib._lib.obj_classes.class_misc_Dipole import Dipole
from magpylib._lib.obj_classes.class_current_Circular import Circular

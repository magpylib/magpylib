"""_lib.obj_classes"""

__all__ = ['Box', 'Cylinder', 'Collection', 'Sensor', 'motion_merge']

# create interface to outside of package
from magpylib._lib.obj_classes.class_mag_Box import Box
from magpylib._lib.obj_classes.class_mag_Cylinder import Cylinder
from magpylib._lib.obj_classes.class_Collection import Collection
from magpylib._lib.obj_classes.class_Sensor import Sensor
from magpylib._lib.obj_classes.class_BaseGeo import motion_merge

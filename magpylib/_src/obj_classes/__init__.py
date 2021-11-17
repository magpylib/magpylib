"""_src.obj_classes"""

__all__ = ['Cuboid', 'Cylinder', 'Sphere', 'Collection', 'Sensor',
    'Dipole', 'Loop', 'Line', 'CylinderSegment','Custom']

# create interface to outside of package
from magpylib._src.obj_classes.class_mag_Cuboid import Cuboid
from magpylib._src.obj_classes.class_mag_Cylinder import Cylinder
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.obj_classes.class_Sensor import Sensor
from magpylib._src.obj_classes.class_mag_Sphere import Sphere
from magpylib._src.obj_classes.class_misc_Dipole import Dipole
from magpylib._src.obj_classes.class_current_Loop import Loop
from magpylib._src.obj_classes.class_current_Line import Line
from magpylib._src.obj_classes.class_mag_CylinderSegment import CylinderSegment
from magpylib._src.obj_classes.class_misc_Custom import Custom

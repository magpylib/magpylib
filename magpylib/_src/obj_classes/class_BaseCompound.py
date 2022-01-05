"""BaseCompound class code"""
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_Collection import BaseCollection

class BaseCompound(BaseGeo, BaseCollection):
    """BaseCompound class inheriting from BaseGeo and BaseCollection. Basically a Collection with
    position and orientation attributes"""
    def __init__(self, *args, position=(0.0, 0.0, 0.0), orientation=None, style=None):
        BaseGeo.__init__(self, position=position, orientation=orientation, style=style)
        BaseCollection.__init__(self, *args)

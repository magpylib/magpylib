"""_src.fields"""

__all__ = ['getB', 'getB_dict', 'getH', 'getH_dict']

# create interface to outside of package
from magpylib._src.fields.field_wrap_BH_level3 import getB, getH
from magpylib._src.fields.field_wrap_BH_level2_dict import getB_dict, getH_dict

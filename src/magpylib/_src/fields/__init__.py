"""_src.fields"""

__all__ = ["getB", "getH", "getJ", "getM"]

# create interface to outside of package
from magpylib._src.fields.field_wrap_BH import getB, getH, getJ, getM

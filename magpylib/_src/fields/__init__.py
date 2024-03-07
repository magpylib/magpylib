"""_src.fields"""

__all__ = ["getB", "getH", "getM", "getJ"]

# create interface to outside of package
from magpylib._src.fields.field_wrap_BH import getB, getH, getM, getJ

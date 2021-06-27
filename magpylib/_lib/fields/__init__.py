"""_lib.fields"""

__all__ = ['getB', 'getBv', 'getH', 'getHv']

# create interface to outside of package
from magpylib._lib.fields.field_wrap_BH_level3 import getB, getH
from magpylib._lib.fields.field_wrap_BH_v import getBv, getHv

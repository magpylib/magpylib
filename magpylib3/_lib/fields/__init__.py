"""_lib.fields"""

__all__ = ['getB', 'getBv', 'getH', 'getHv']

# create interface to outside of package
from magpylib3._lib.fields.field_wrap_getBH import getB, getH
from magpylib3._lib.fields.field_wrap_getBHv import getBv, getHv

"""_lib.math_utility"""

__all__ = ['celv', 'format_src_input', 'check_duplicates', 'same_path_length',
    'rotobj_from_angax', 'check_allowed_keys']

# create interface
from magpylib3._lib.math_utility.special_functions import celv
from magpylib3._lib.math_utility.utility import (format_src_input, check_duplicates,
    same_path_length, rotobj_from_angax, check_allowed_keys)

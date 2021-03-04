"""_lib.math_utility"""

__all__ = ['celv', 'format_src_input', 'check_duplicates', 'good_path_format',
           'rotobj_from_angax', 'check_allowed_keys', 'all_same', 'get_good_path_length']

# create interface
from magpylib3._lib.math_utility.special_functions import celv
from magpylib3._lib.math_utility.utility import (
    format_src_input, check_duplicates, good_path_format, rotobj_from_angax, 
    check_allowed_keys, all_same, get_good_path_length)

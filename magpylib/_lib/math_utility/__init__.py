"""_lib.math_utility"""

__all__ = ['celv', 'format_src_input', 'check_duplicates', 'test_path_format',
           'rotobj_from_angax', 'all_same', 'get_good_path_length']

# create interface
from magpylib._lib.math_utility.special_functions import celv
from magpylib._lib.math_utility.utility import (
    format_src_input, check_duplicates, test_path_format, rotobj_from_angax,
    all_same, get_good_path_length)

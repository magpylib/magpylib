import numpy as np
import magpylib as mag3
from magpylib._lib.exceptions import MagpylibBadUserInput

from magpylib._lib.math_utility.utility import test_path_format

def test_path_format():
    """ test with single source input
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm1.pos = [(1,2,3),(1,2,3)]
    flag = False
    try:
        test_path_format(pm1)
    except MagpylibBadUserInput:
        flag = True
    assert flag, 'test_path_format_fail'

test_test_path_format()

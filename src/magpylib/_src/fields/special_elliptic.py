# from scipy.special import ellipe, ellipk
from __future__ import annotations

from typing import Any

import array_api_extra as xpx
import scipy as sp

type Array = Any


def ellipe(m: Array):
    # return cel(kc, one, one, kc2)
    return xpx.lazy_apply(sp.special.ellipe, m, as_numpy=True)


def ellipk(m: Array):
    return xpx.lazy_apply(sp.special.ellipk, m, as_numpy=True)

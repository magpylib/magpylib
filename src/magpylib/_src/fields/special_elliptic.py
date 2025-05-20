# from scipy.special import ellipe, ellipk
from __future__ import annotations

from typing import Any

import array_api_extra as xpx
import scipy as sp

type Array = Any


def ellipeinc(phi: Array, m: Array):
    return xpx.lazy_apply(sp.special.ellipeinc, phi, m, as_numpy=True)


def ellipkinc(phi: Array, m: Array):
    return xpx.lazy_apply(sp.special.ellipkinc, phi, m, as_numpy=True)

import numpy as np

import magpylib as magpy
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_scalar_potential
from magpylib._src.fields.field_BH_cuboid import BHJM_magnet_cuboid

from scipy.constants import mu_0

def derivative(fun, x, args, eps=1e-6):

    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=0)

    deltax = np.zeros(x.shape)
    deltax[:,0] = eps
    deltay = np.zeros(x.shape)
    deltay[:,1] = eps
    deltaz = np.zeros(x.shape)
    deltaz[:,2] = eps
    dfdx = (fun(x+deltax, *args) - fun(x-deltax, *args)) / 2 / eps
    dfdy = (fun(x+deltay, *args) - fun(x-deltay, *args)) / 2 / eps
    dfdz = (fun(x+deltaz, *args) - fun(x-deltaz, *args)) / 2 / eps

    return (dfdx, dfdy, dfdz)


def test_core_derivative():
    """Comparison to field"""
    observers = np.array(((0,0,0), (0.5,1,1.5), (1.5,3,4.5), (2,4,6)))
    dimensions = np.array(((2,4,6), (2,4,6), (2,4,6), (2,4,6)))
    magnetizations = np.array(((1,2,3), (1,2,3), (1,2,3), (1,2,3)))

    res = derivative(magnet_cuboid_scalar_potential, observers, (dimensions, magnetizations), eps=1e-6)

    H_potential = np.zeros(dimensions.shape)
    H_potential[:,0] = -res[0]
    H_potential[:,1] = -res[1]
    H_potential[:,2] = -res[2]


    H_direct = magpy.getH(
        sources='Cuboid',
        observers=observers,
        dimension=dimensions,
        polarization=magnetizations*mu_0
    )

    np.testing.assert_allclose(H_potential, H_direct)

def test_core_value_check():
    """Comparison to selected results"""
    observers = np.array(((0,0,0), (1,0,0), (0,2,0), (0,0,3), (1,2,0), (0,2,3), (1,0,3), (1,2,3)))
    dimensions = np.tile((2,4,6), (8,1))
    magnetizations = np.tile((1,2,3), (8,1))
    results_precalculated = np.array([-1.06018489e-16, 0.66430034, 1.34565504, 1.98630676, 1.30687569, 1.90791399, 1.72798245, 1.58039597])

    results = magnet_cuboid_scalar_potential(observers, dimensions, magnetizations)

    np.testing.assert_allclose(results_precalculated, results)


def test_BHJM_value_check():
    """Comparison to selected results"""
    observers = np.array(((0,0,0), (1,0,0), (0,2,0), (0,0,3), (1,2,0), (0,2,3), (1,0,3), (1,2,3)))
    dimensions = np.tile((2,4,6), (8,1))
    magnetizations = np.tile((1,2,3), (8,1))

    results_core = magnet_cuboid_scalar_potential(observers, dimensions, magnetizations)

    results_BHJM = BHJM_magnet_cuboid(field="phi", observers=observers, polarization=magnetizations*mu_0, dimension=dimensions)

    np.testing.assert_allclose(results_core, results_BHJM)
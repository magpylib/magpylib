import numpy as np
import magpylib as magpy


def mesh_cuboid(cuboid, nnn):
    """
    Splits Cuboid up into small Cuboid cells

    Parameters
    ----------
    cuboid: magpylib.Cuboid object
        input object to be discretized
    nnn: tuple, shape (3,), dtype=int
        discretization (nx,ny,nz)

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """

    # load cuboid properties
    pos = cuboid.position
    rot = cuboid.orientation
    dim = cuboid.dimension
    mag = cuboid.magnetization

    # secure input type
    nnn = np.array(nnn, dtype=int)

    # new dimension
    new_dim = dim/nnn

    # inside position grid
    xs,ys,zs = [np.linspace(d/2*(1/n-1), d/2*(1-1/n), n) for d,n in zip(dim,nnn)]
    grid = np.array([(x,y,z) for x in xs for y in ys for z in zs])
    grid = rot.apply(grid) + pos

    # create cells as magpylib objects ad return Collection
    cells = [magpy.magnet.Cuboid(mag, new_dim, pp, rot) for pp in grid]

    return magpy.Collection(cells)

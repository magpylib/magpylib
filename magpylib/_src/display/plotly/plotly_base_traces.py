"""plotly base traces utility functions"""

import numpy as np
from magpylib._src.display.plotly.plotly_utility import merge_mesh3d


def make_BaseCuboid(dimension=(1.0, 1.0, 1.0), position=(0.0, 0.0, 0.0)) -> dict:
    """
    Provides the base plotly cuboid mesh3d parameters in a dictionary based on dimension and
    position
    The zero position is in the barycenter of the vertices.
    """
    return dict(
        type="mesh3d",
        i=np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j=np.array([0, 7, 1, 2, 6, 7, 1, 2, 5, 5, 2, 2]),
        k=np.array([3, 4, 2, 3, 5, 6, 5, 5, 0, 1, 7, 6]),
        x=np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * 0.5 * dimension[0] + position[0],
        y=np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * 0.5 * dimension[1] + position[1],
        z=np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * 0.5 * dimension[2] + position[2],
    )


def make_BasePrism(
    base_vertices=3, diameter=1, height=1, position=(0.0, 0.0, 0.0)
) -> dict:
    """
    Provides the base plotly prism mesh3d parameters in a dictionary based on number of vertices of
    the base, the diameter the height and position.
    The zero position is in the barycenter of the vertices.
    """
    N = base_vertices
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c1 = np.array([1 * np.cos(t), 1 * np.sin(t), t * 0 - 1]) * 0.5
    c2 = np.array([1 * np.cos(t), 1 * np.sin(t), t * 0 + 1]) * 0.5
    c3 = np.array([[0, 0], [0, 0], [-1, 1]]) * 0.5
    c = np.concatenate([c1, c2, c3], axis=1)
    c = c.T * np.array([diameter, diameter, height]) + np.array(position)
    i1 = np.arange(N)
    j1 = i1 + 1
    j1[-1] = 0
    k1 = i1 + N

    i2 = i1 + N
    j2 = j1 + N
    j2[-1] = N
    k2 = i1 + 1
    k2[-1] = 0

    i3 = i1
    j3 = j1
    k3 = i1 * 0 + 2 * N

    i4 = i2
    j4 = j2
    k4 = k3 + 1

    # k2&j2 and k3&j3 inverted because of face orientation
    i = np.concatenate([i1, i2, i3, i4])
    j = np.concatenate([j1, k2, k3, j4])
    k = np.concatenate([k1, j2, j3, k4])

    x, y, z = c.T
    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_BaseEllipsoid(
    dimension=(1.0, 1.0, 1.0),
    position=(0.0, 0.0, 0.0),
    Nvert=15,
    min_vert=3,
    max_vert=20,
) -> dict:
    """
    Provides the base plotly ellipsoid mesh3d parameters in a dictionary based on number of vertices
    of the circumference, the 3 dimensions and position.
    The zero position is in the barycenter of the vertices.
    `Nvert` will be forced automatically in the range [`min_vert`, `max_vert`]
    """
    N = min(max(Nvert, min_vert), max_vert)
    phi = np.linspace(0, 2 * np.pi, Nvert, endpoint=False)
    theta = np.linspace(-np.pi / 2, np.pi / 2, Nvert, endpoint=True)
    phi, theta = np.meshgrid(phi, theta)

    x = np.cos(theta) * np.sin(phi) * dimension[0] * 0.5 + position[0]
    y = np.cos(theta) * np.cos(phi) * dimension[1] * 0.5 + position[1]
    z = np.sin(theta) * dimension[2] * 0.5 + position[2]

    x, y, z = x.flatten()[N - 1 :], y.flatten()[N - 1 :], z.flatten()[N - 1 :]

    i1 = [0] * N
    j1 = np.array([N] + list(range(1, N)), dtype=int)
    k1 = np.array(list(range(1, N)) + [N], dtype=int)

    i2 = np.concatenate([k1 + i * N for i in range(N - 2)])
    j2 = np.concatenate([j1 + i * N for i in range(N - 2)])
    k2 = np.concatenate([j1 + (i + 1) * N for i in range(N - 2)])

    i3 = np.concatenate([k1 + i * N for i in range(N - 2)])
    j3 = np.concatenate([j1 + (i + 1) * N for i in range(N - 2)])
    k3 = np.concatenate([k1 + (i + 1) * N for i in range(N - 2)])

    i = np.concatenate([i1, i2, i3])
    j = np.concatenate([j1, j2, j3])
    k = np.concatenate([k1, k2, k3])

    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_BaseCylinderSegment(r1=1, r2=2, h=1, phi1=0, phi2=90, Nvert=30) -> dict:
    """
    Provides the base plotly CylinderSegment mesh3d parameters in a dictionary based on inner
    and outer diameters, height, start angle and end angles in degrees.
    The zero position is in the barycenter of the vertices.
    """
    N = Nvert
    phi = np.linspace(phi1, phi2, N)
    x = np.cos(np.deg2rad(phi))
    y = np.sin(np.deg2rad(phi))
    z = np.zeros(N)
    c1 = np.array([r1 * x, r1 * y, z + h / 2])
    c2 = np.array([r2 * x, r2 * y, z + h / 2])
    c3 = np.array([r1 * x, r1 * y, z - h / 2])
    c4 = np.array([r2 * x, r2 * y, z - h / 2])
    x, y, z = np.concatenate([c1, c2, c3, c4], axis=1)

    i1 = np.arange(N - 1)
    j1 = i1 + N
    k1 = i1 + 1

    i2 = k1
    j2 = j1
    k2 = j1 + 1

    i3 = i1
    j3 = k1
    k3 = j1 + N

    i4 = k3 + 1
    j4 = k3
    k4 = k1

    i5 = np.array([0, N])
    j5 = np.array([2 * N, 0])
    k5 = np.array([3 * N, 3 * N])

    i = np.hstack(
        [i1, i2, i1 + 2 * N, i2 + 2 * N, i3, i4, i3 + N, i4 + N, i5, i5 + N - 1]
    )
    j = np.hstack(
        [j1, j2, k1 + 2 * N, k2 + 2 * N, j3, j4, k3 + N, k4 + N, j5, k5 + N - 1]
    )
    k = np.hstack(
        [k1, k2, j1 + 2 * N, j2 + 2 * N, k3, k4, j3 + N, j4 + N, k5, j5 + N - 1]
    )

    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_BaseCone(
    base_vertices=3, diameter=1, height=1, position=(0.0, 0.0, 0.0)
) -> dict:
    """
    Provides the base plotly Cone mesh3d parameters in a dictionary based on number of vertices of
    the base, the diameter the height and position.
    The zero position is in the barycenter of the vertices.
    """
    N = base_vertices
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c = np.array([np.cos(t), np.sin(t), t * 0 - 1]) * 0.5
    tp = np.array([[0, 0, 0.5]]).T
    c = np.concatenate([c, tp], axis=1)
    c = c.T * np.array([diameter, diameter, height]) + np.array(position)
    x, y, z = c.T

    i = np.arange(N, dtype=int)
    j = i + 1
    j[-1] = 0
    k = np.array([N] * N, dtype=int)
    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_BaseArrow(base_vertices=30, diameter=0.3, height=1, pivot="middle") -> dict:
    """
    Provides the base plotly 3D Arrow mesh3d parameters in a dictionary based on number of vertices
    of the base, the diameter the height and position.
    The zero position is in the barycenter of the vertices.
    """

    h, d, z = height, diameter, 0
    if pivot == "tail":
        z = h / 2
    elif pivot == "tip":
        z = -h / 2
    cone = make_BaseCone(
        base_vertices=base_vertices,
        diameter=d,
        height=d,
        position=(0.0, 0.0, z + h / 2 - d / 2),
    )
    prism = make_BasePrism(
        base_vertices=base_vertices,
        diameter=d / 2,
        height=h - d,
        position=(0.0, 0.0, z + -d / 2),
    )
    arrow = merge_mesh3d(cone, prism)

    return arrow

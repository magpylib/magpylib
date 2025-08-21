import numpy as np
import matplotlib.pyplot as plt

import numpy as np


def subdiv(triangles: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """
    bisection along longest edges
    """
    n_sub = 2**splits                 # subdivisions per tria
    n_tot = np.sum(n_sub)         # total number of trias
    
    # indixes in TRI for broadcasting
    ends = np.cumsum(n_sub)
    starts = np.r_[0, ends[:-1]]
    
    # allocate output
    TRIA = np.empty((n_tot, 3, 3))
    
    # store given triangles in TRI
    TRIA[starts] = triangles

    # loop over n_splits and subdivide triangles. In each step store triangles in TRIA
    #   filling up TRIA step by step. For vectorization power, all input triangles are
    #   subdivided at the same time.

    #s = np.zeros_like(splits) # current number of splits in respective groups
    
    # triangles in TRIA that should be subdivided
    MASK = np.zeros((n_tot), dtype=bool)
    MASK[starts] = True
    
    print(splits)

    for i in range(max(splits)):
        
        # reset completed groups
        mask_split = (i == splits)
        for start in starts[mask_split]:
            MASK[start:start+2**(i+1)] = False

        print(i)
        # select all triangle groups that should be split
        print(MASK)
        triangles = TRIA[MASK]
        
        # create target broadcasting mask
        #MASK[:] = False
        mask_split = (i < splits) # select triangle groups where further splitting is required
        for start in starts[mask_split]:
            MASK[start:start+2**(i+1)] = True
        #s[mask_split] += 1         # update number of splits of this group

        print(MASK)
        print("---")

        # corners
        A = triangles[:,0]
        B = triangles[:,1]
        C = triangles[:,2]

        # Squared lengths of edges
        d2_AB = np.sum((B - A)**2, axis=1)
        d2_BC = np.sum((C - B)**2, axis=1)
        d2_CA = np.sum((A - C)**2, axis=1)

        mask1 = (d2_AB >= d2_BC) * (d2_AB >= d2_CA)
        mask2 = (d2_BC >= d2_CA)
        mask3 = ~(mask1 | mask2)

        new_triangles = np.empty((len(triangles), 2, 3, 3), dtype=float)

        if np.any(mask1):
            new_triangles[mask1, 0, 0] = A[mask1]
            new_triangles[mask1, 0, 1] = (A[mask1] + B[mask1]) / 2.0
            new_triangles[mask1, 0, 2] = C[mask1]
            new_triangles[mask1, 1, 0] = (A[mask1] + B[mask1]) / 2.0
            new_triangles[mask1, 1, 1] = B[mask1]
            new_triangles[mask1, 1, 2] = C[mask1]

        if np.any(mask2):
            new_triangles[mask2, 0, 0] = B[mask2]
            new_triangles[mask2, 0, 1] = (B[mask2] + C[mask2]) / 2.0
            new_triangles[mask2, 0, 2] = A[mask2]
            new_triangles[mask2, 1, 0] = (B[mask2] + C[mask2]) / 2.0
            new_triangles[mask2, 1, 1] = C[mask2]
            new_triangles[mask2, 1, 2] = A[mask2]

        if np.any(mask3):
            new_triangles[mask3, 0, 0] = C[mask3]
            new_triangles[mask3, 0, 1] = (C[mask3] + A[mask3]) / 2.0
            new_triangles[mask3, 0, 2] = B[mask3]
            new_triangles[mask3, 1, 0] = (C[mask3] + A[mask3]) / 2.0
            new_triangles[mask3, 1, 1] = A[mask3]
            new_triangles[mask3, 1, 2] = B[mask3]

        TRIA[MASK] = new_triangles.reshape(-1, 3, 3)

    return TRIA


def target_meshing_triangle(triangles: np.ndarray, n_target: int) -> np.ndarray:
    """
    Refines triangle ABC by bisecting its longest edge.

    Returns two sub-triangles.
    triangles: array shape (n, 3, 3)
    """

    n_tria = len(triangles)
    splits = np.zeros(n_tria, dtype=int)
    surfaces = 0.5 * np.linalg.norm(np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0]), axis=1)

    # longest edge bisection splits triangle surface always in half
    # all triangles should in the end have similar surface
    # so we can easily calculate which triangles we have to split how often
    while n_tria < n_target:
        idx = np.argmax(surfaces)
        surfaces[idx] /= 2.0
        splits[idx] += 1
        n_tria = np.sum(2**splits)

    surfaces = np.repeat(surfaces, 2**splits)
    triangles = subdiv(triangles, splits)
    centroids = np.mean(triangles, axis=1)

    return triangles, centroids, surfaces



    # while n_tria < n_target:

    #     # corners
    #     A = triangles[:,0]
    #     B = triangles[:,1]
    #     C = triangles[:,2]

    #     # Squared lengths of edges
    #     d2_AB = np.sum((B - A)**2, axis=1)
    #     d2_BC = np.sum((C - B)**2, axis=1)
    #     d2_CA = np.sum((A - C)**2, axis=1)

    #     mask1 = (d2_AB >= d2_BC) * (d2_AB >= d2_CA)
    #     mask2 = (d2_BC >= d2_CA)
    #     mask3 = ~(mask1 | mask2)

    #     new_triangles = np.empty((len(triangles), 2, 3, 3), dtype=float)

    #     if np.any(mask1):
    #         new_triangles[mask1, 0, 0] = A[mask1]
    #         new_triangles[mask1, 0, 1] = (A[mask1] + B[mask1]) / 2.0
    #         new_triangles[mask1, 0, 2] = C[mask1]
    #         new_triangles[mask1, 1, 0] = (A[mask1] + B[mask1]) / 2.0
    #         new_triangles[mask1, 1, 1] = B[mask1]
    #         new_triangles[mask1, 1, 2] = C[mask1]

    #     if np.any(mask2):
    #         new_triangles[mask2, 0, 0] = B[mask2]
    #         new_triangles[mask2, 0, 1] = (B[mask2] + C[mask2]) / 2.0
    #         new_triangles[mask2, 0, 2] = A[mask2]
    #         new_triangles[mask2, 1, 0] = (B[mask2] + C[mask2]) / 2.0
    #         new_triangles[mask2, 1, 1] = C[mask2]
    #         new_triangles[mask2, 1, 2] = A[mask2]

    #     if np.any(mask3):
    #         new_triangles[mask3, 0, 0] = C[mask3]
    #         new_triangles[mask3, 0, 1] = (C[mask3] + A[mask3]) / 2.0
    #         new_triangles[mask3, 0, 2] = B[mask3]
    #         new_triangles[mask3, 1, 0] = (C[mask3] + A[mask3]) / 2.0
    #         new_triangles[mask3, 1, 1] = A[mask3]
    #         new_triangles[mask3, 1, 2] = B[mask3]

    #     triangles = new_triangles.reshape(-1, 3, 3)
    #     n_tria = len(triangles)
    
    # centroids = np.mean(triangles, axis=1)
    # #surfaces = 0.5 * np.linalg.norm(np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0]), axis=1)
    # print(surfaces)

    # return triangles, centroids, surfaces

tri_array = np.array([((0,0,0),(1,0,0),(0,1,0)), ((1,0,0),(2,0,0),(3,.2,0)), ((1,1,0), (2,1,0), (3,2,0))])
trias, cent, surf = target_meshing_triangle(tri_array, 12)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

fig, ax = plt.subplots()
#plt.plot(div[:,0], div[:,1], 'ro', ms=5)


for triangle in trias:
    triangle = Polygon(triangle[:,:2], closed=True, color=np.random.rand(3), edgecolor='none')
    ax.add_patch(triangle)

# keep axes scaled correctly
ax.set_aspect('equal')
ax.autoscale_view()
plt.show()


# plt.plot(tri[:,0], tri[:,1], 'ro', ms=8)


# for t in div:
#     plt.plot(t[:,0], t[:,1], 'bo', ms=6)
#     div = subdiv(*t)
#     for t in div:
#         plt.plot(t[:,0], t[:,1], 'yo', ms=4)


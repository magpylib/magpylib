import numpy as np
from scipy.spatial.transform import Rotation as R


def collision_xy_cuboids(obj1, obj2):
    """
    2D collision with hyperplane separation. Assumption: Magpylib Cuboid objects
    stand upright in a plane parallel to the xy-plane.

    all obj1[i] tested against obj2[i]

    Parameters
    ----------
    obj1: list of Cuboid objects
        Objects tested against obj2 for collision
    obj2: list of Cuboid objects
        Objects tested against obj1 for collision

    Returns
    -------
    collision: int or None
        If there is no collision return None, else return i where collision
        happens.
    """

    # compute all Cuboid corners
    all_obj = obj1 + obj2
    n = len(obj1)

    dim = np.array([o.dimension for o in all_obj])
    pos = np.array([o.position for o in all_obj])
    rot = np.array([o.orientation.as_quat() for o in all_obj])
    rot4 = np.repeat(rot, 4, axis=0)
    rot4 = R.from_quat(rot4)
    pos4 = np.repeat(pos, 4, axis=0)
    all_rect = np.array([(d[0]*i/2, d[1]*j*i/2, 0)
        for d in dim for i in [1,-1] for j in [1,-1]])
    all_rect = rot4.apply(all_rect) + pos4
    all_rect = np.reshape(all_rect,(-1,4,3))

    # collide obj1->obj2 AND obj2->obj1
    SQ1 = all_rect[:,:,:2]
    SQ2 = np.r_[all_rect[n:,:,:2], all_rect[:n,:,:2]]

    # select planes
    C = SQ1[:,1]
    edge1 = (SQ1[:,0]-C)
    edge2 = (SQ1[:,2]-C)

    # "normalized" edges - avoid sqrt
    edge1_lam = edge1/np.tile(np.sum(edge1**2, axis=1), (2,1)).T
    edge2_lam = edge2/np.tile(np.sum(edge2**2, axis=1),( 2,1)).T

    # "projections" - avoid sqrt
    SQ2_flip = np.swapaxes(SQ2,0,1)
    proj1 = np.sum((SQ2_flip-C)*edge1_lam,axis=2).T
    proj2 = np.sum((SQ2_flip-C)*edge2_lam,axis=2).T

    # collision test
    no_collision1 = np.all(proj1<0, axis=1) | np.all(proj1>1, axis=1) # all above or below
    no_collision2 = np.all(proj2<0, axis=1) | np.all(proj2>1, axis=1)
    no_collision = no_collision1 | no_collision2                      # on edge1 or edge2
    no_collision = no_collision[:n] | no_collision[n:]                # obj1->obj2 OR obj2->obj1

    return ~no_collision

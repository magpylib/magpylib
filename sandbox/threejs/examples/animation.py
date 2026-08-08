"""Paths, played back from the model rather than from baked frames.

Magpylib animates by re-rendering the whole scene once per path step. For the
scene below that is 588 KB of vertex data; the same paths as position and
quaternion arrays are 1.8 KB, because every frame of a moving object is a rigid
transform of the first. So the backend declines `supports_animation`, takes one
frame, and animates the transforms itself -- which is what a scene graph does
per frame anyway.

The last object is the exception that makes the rule visible: its *dimension*
sweeps rather than its pose, so no matrix can express it and it is reported as
non-rigid instead of being played back wrongly.

Run it, then press space.
"""

from __future__ import annotations

# the backend lives one level up, beside the JS it serves
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# printing is what this script is for
# ruff: noqa: T201
import numpy as np
import threejs_edit  # the import registers the backend

import magpylib as magpy

STEPS = 60


def build():
    """Four kinds of path, three of which are rigid motions."""
    # 1. a straight sweep
    sweeper = magpy.magnet.Cuboid(
        dimension=(0.6, 0.6, 1.4),
        polarization=(0, 0, 1),
        position=np.linspace((-5, 0, 0), (5, 0, 0), STEPS),
        style_label="sweeper",
    )

    # 2. an orbit -- rotation about an anchor moves *and* turns the object
    orbiter = magpy.magnet.Sphere(
        diameter=0.8,
        polarization=(1, 0, 0),
        position=(3, 0, 2),
        style_label="orbiter",
    )
    orbiter.rotate_from_angax(
        np.linspace(0, 360, STEPS), "z", anchor=(0, 0, 2), start=0
    )

    # 3. a spin in place -- orientation varies, position does not
    spinner = magpy.magnet.CylinderSegment(
        dimension=(0.6, 1.4, 0.5, 0, 240),
        polarization=(0, 0, 1),
        position=(0, 4, 0),
        style_label="spinner",
    )
    spinner.rotate_from_angax(np.linspace(0, 720, STEPS), "z", start=0)

    # 4. a shape that changes rather than moves: not a rigid motion, so it
    #    cannot be animated by transform and is reported as such
    grower = magpy.magnet.Cuboid(
        dimension=np.linspace((0.4, 0.4, 0.4), (1.8, 1.8, 1.8), STEPS),
        polarization=(0, 1, 0),
        position=(0, -4, 0),
        style_label="grower",
    )

    probe = magpy.Sensor(
        position=(0, 0, -3),
        pixel=[(x, 0, 0) for x in (-0.4, 0, 0.4)],
        style_label="probe",
    )
    return [sweeper, orbiter, spinner, grower, probe]


def report(objects):
    """What each path costs, and whether a matrix can express it."""
    tracks = threejs_edit.build_tracks({id(o): o for o in objects})
    print(f"{'object':10s} {'steps':>6} {'rigid':>7}  varying")
    print("-" * 44)
    for obj in objects:
        track = tracks.get(str(id(obj)))
        label = obj.style.label
        if track is None:
            print(f"{label:10s} {'-':>6} {'-':>7}  no path")
            continue
        print(
            f"{label:10s} {len(track['position']):6d} {track['rigid']!s:>7}"
            f"  {', '.join(track['varying']) or '-'}"
        )


def main():
    objects = build()
    report(objects)
    print("\nspace plays, the status bar shows the frame")
    magpy.show(*objects, backend="threejs-edit")


if __name__ == "__main__":
    main()

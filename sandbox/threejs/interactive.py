"""Pick-and-drag on top of the prototype backend, to test the editing loop.

The question this answers: can a drag run without a Python round-trip? A
round-trip per frame would cap interaction at whatever `show()` costs, which is
seconds on a large scene -- unusable. The answer is yes, on two conditions,
and both are things the probe already found:

1. Geometry must be **scene-independent**, or moving one object invalidates
   every other object's vertices and the JS scene cannot be kept. That means
   `units_length` pinned and `sizemode="absolute"` (finding 1).
2. The edit must be a **transform**. Moving and rotating are matrix updates the
   GPU already does. Changing a dimension is not -- that needs new geometry,
   so it needs Python, but it is not a per-frame operation.

So the loop is: drag entirely in JS at frame rate, and tell Python **once**, on
drop. The HUD counts round-trips to make the point checkable rather than
asserted.

Run with `python sandbox/threejs/interactive.py`.
"""

from __future__ import annotations

# printing is what this script is for
# ruff: noqa: T201
import webbrowser
from pathlib import Path

from magpylib_threejs import render_page

import magpylib as magpy

HERE = Path(__file__).parent

# `import` statements must sit at the top of a module script, so the addon
# import is prepended to the page rather than appended with the rest.
_IMPORT = """
import { TransformControls } from 'three/addons/controls/TransformControls.js';
"""

_INTERACTION_JS = """
// ---- pick and drag -------------------------------------------------------
// Everything here runs in the browser. Nothing calls back into Python until
// the drag ends, which is the whole point: see the round-trip counter.

const hud = document.createElement('div');
hud.id = 'hud';
hud.innerHTML = `<b>click an object</b><br>
  <span id="sel">nothing selected</span><br>
  <span id="delta"></span><br>
  <span id="calls">python round-trips: 0</span><br>
  <small>W translate &middot; E rotate &middot; Esc deselect</small>`;
document.body.appendChild(hud);
const style = document.createElement('style');
style.textContent = `#hud { position: absolute; bottom: 8px; left: 8px;
  font: 12px/1.5 sans-serif; background: rgba(255,255,255,.9);
  padding: 8px 10px; border-radius: 4px; }`;
document.head.appendChild(style);

const gizmo = new TransformControls(camera, renderer.domElement);
// OrbitControls must yield while the gizmo has the pointer, or dragging a
// handle also spins the camera.
gizmo.addEventListener('dragging-changed', e => { controls.enabled = !e.value; });
scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);

let selected = null, startPosition = null, roundTrips = 0;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

// only meshes are pickable; lines and points are decoration here
const pickable = [...byObjectId.values()];

function select(mesh) {
  selected = mesh;
  if (!mesh) { gizmo.detach(); document.getElementById('sel').textContent =
    'nothing selected'; document.getElementById('delta').textContent = ''; return; }
  gizmo.attach(mesh);
  startPosition = mesh.position.clone();
  document.getElementById('sel').textContent = `selected: ${mesh.name}`;
  document.getElementById('delta').textContent = 'drag a handle';
}

renderer.domElement.addEventListener('pointerdown', event => {
  if (gizmo.dragging) return;
  pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(pickable, false)[0];
  select(hit ? hit.object : null);
});

addEventListener('keydown', e => {
  if (e.key === 'w') gizmo.setMode('translate');
  if (e.key === 'e') gizmo.setMode('rotate');
  if (e.key === 'Escape') select(null);
});

// during the drag: JS only, no message to Python
gizmo.addEventListener('objectChange', () => {
  const d = selected.position.clone().sub(startPosition);
  document.getElementById('delta').textContent =
    `delta: (${d.x.toFixed(3)}, ${d.y.toFixed(3)}, ${d.z.toFixed(3)}) ${UNIT}`;
});

// on drop: one message, carrying object_id and the delta. A real host would
// post this to the engine, which applies obj.position += delta and recomputes
// whatever depends on it.
gizmo.addEventListener('mouseUp', () => {
  if (!selected) return;
  const d = selected.position.clone().sub(startPosition);
  if (d.lengthSq() === 0) return;
  roundTrips += 1;
  document.getElementById('calls').textContent =
    `python round-trips: ${roundTrips}`;
  console.log('would send to python:',
    { object_id: selected.userData.objectId, delta: d.toArray() });
  startPosition = selected.position.clone();
});
"""


def main():
    # The host owns the objects, so it can resolve object_id back to the
    # object and read the transform Magpylib does not put in the payload.
    # `object_id` is documented as valid for exactly this: "an interactive
    # viewer holding the same objects". Nothing has to change in Magpylib.
    registry: dict[int, object] = {}

    def show(scene):
        anchors = {oid: obj.position for oid, obj in registry.items()}
        return render_page(scene, extra_js=_INTERACTION_JS, anchors=anchors)

    magpy.register_backend(
        "threejs-edit",
        show,
        supports_colorgradient=True,
        merge_traces=False,  # one mesh per object, so each can be picked
        handles_traces=frozenset({"mesh3d", "scatter3d"}),
        accepts_options=frozenset(),
    )

    # Pin both scene-dependent scalings, or dragging one object would change
    # the geometry of the others and the JS scene could not be kept.
    magpy.defaults.display.style.sensor.sizemode = "absolute"
    magpy.defaults.display.style.dipole.sizemode = "absolute"

    objects = [
        magpy.magnet.Cuboid(dimension=(1, 2, 3), polarization=(0, 0, 1)),
        magpy.magnet.Cylinder(
            dimension=(1, 1), polarization=(0, 0, 1), position=(3, 0, 0)
        ),
        magpy.magnet.Sphere(diameter=1.5, polarization=(1, 0, 0), position=(-3, 0, 0)),
        magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 3, 0)),
        magpy.Sensor(position=(0, -3, 0), pixel=[(x, 0, 0) for x in (-0.3, 0, 0.3)]),
    ]

    # units_length="m" matches magpylib's own unit, so positions need no
    # conversion. Any other unit and the host must scale the anchors to match.
    registry.update({id(obj): obj for obj in objects})

    html = magpy.show(
        *objects, backend="threejs-edit", units_length="m", return_fig=True
    )
    # the addon import has to precede the module body
    html = html.replace(
        "import { OrbitControls }",
        _IMPORT.strip() + "\nimport { OrbitControls }",
    ).replace("const DATA =", "const UNIT = 'm';\nconst DATA =")

    page = HERE / "interactive.html"
    page.write_text(html)
    print(f"wrote {page}")
    webbrowser.open(f"file://{page}")


if __name__ == "__main__":
    main()

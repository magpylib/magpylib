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
import json
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
// ---- pick, drag, and an edit history -------------------------------------
// Everything here runs in the browser. Nothing calls back into Python until an
// edit is committed, which is the whole point: see the round-trip counter.
//
// Every edit -- gizmo drop, typed value, undo, redo -- goes through applyEdit
// and emits the same {object_id, field, value} message. Undo is therefore not
// a separate mechanism: it is an edit whose value happens to be the previous
// one. The browser holds the history only so it can name that value; the model
// is the magpylib object, and a real host would replay these against it.

const hud = document.createElement('div');
hud.id = 'hud';
hud.innerHTML = `<b>click an object</b><br>
  <span id="sel">nothing selected</span><br>
  <span id="delta"></span><br>
  <span id="calls">python round-trips: 0</span><br>
  <span id="snap">snap: off</span> &middot; <span id="hist">history: 0</span><br>
  <small>W move &middot; E rotate &middot; R resize &middot; S snap<br>
  &#8984;/Ctrl+Z undo &middot; &#8679;+Z redo &middot; Backspace reset</small>`;
document.body.appendChild(hud);

const inspector = document.createElement('div');
inspector.id = 'inspector';
inspector.innerHTML = `<b>exact values</b><div id="fields">select an object</div>`;
document.body.appendChild(inspector);

const style = document.createElement('style');
style.textContent = `#hud { position: absolute; bottom: 8px; left: 8px;
  font: 12px/1.5 sans-serif; background: rgba(255,255,255,.9);
  padding: 8px 10px; border-radius: 4px; }
#inspector { position: absolute; top: 8px; right: 8px;
  font: 12px/1.8 sans-serif; background: rgba(255,255,255,.9);
  padding: 8px 10px; border-radius: 4px; }
#inspector label { display: block; }
#inspector input { width: 5.5em; margin-left: .4em; font: inherit; }`;
document.head.appendChild(style);

const gizmo = new TransformControls(camera, renderer.domElement);
// OrbitControls must yield while the gizmo has the pointer, or dragging a
// handle also spins the camera.
gizmo.addEventListener('dragging-changed', e => { controls.enabled = !e.value; });
scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);

let selected = null, roundTrips = 0;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const pickable = [...byObjectId.values()];   // lines and points are decoration

// ---- state, so an edit knows what it is replacing -------------------------
const state = new Map(), initial = new Map();
for (const [oid, mesh] of byObjectId) {
  const shape = SHAPES[String(oid)];
  const s = { position: mesh.position.toArray(),
              quaternion: mesh.quaternion.toArray() };
  if (shape) s[shape.attr] = shape.value;
  const pol = POLARIZATION[String(oid)];
  if (pol) s[pol.attr] = pol.value;
  state.set(oid, s);
  initial.set(oid, JSON.parse(JSON.stringify(s)));
}

const history = [], redoStack = [];

function applyToMesh(oid, field, value) {
  const mesh = byObjectId.get(oid), shape = SHAPES[String(oid)];
  if (field === 'position') { mesh.position.fromArray(value); return; }
  if (field === 'quaternion') { mesh.quaternion.fromArray(value); return; }
  // Polarization is the one edit with no preview here. It changes `intensity`
  // -- a per-vertex attribute -- and magpylib computes that from the vector in
  // *world* space, i.e. the object's orientation applied to the stored local
  // vector. Recomputing it in the browser means reimplementing that frame
  // convention, and getting it subtly wrong for every rotated object. Ask
  // magpylib instead: one object re-renders in 0.3 ms.
  if (field === 'polarization') return;
  // a shape value is expressed as a scale of the base mesh
  const base = Array.isArray(shape.value) ? shape.value : [shape.value];
  const v = Array.isArray(value) ? value : [value];
  const s = v.map((n, j) => n / base[j]);
  if (shape.constraint === 'uniform') mesh.scale.setScalar(s[0]);
  else if (shape.constraint === 'xy') mesh.scale.set(s[0], s[0], s[1]);
  else mesh.scale.set(s[0], s[1], s[2]);
}

function send(oid, field, value) {
  roundTrips += 1;
  document.getElementById('calls').textContent =
    `python round-trips: ${roundTrips}`;
  console.log('would send to python:', { object_id: oid, [field]: value });
}

// The single door every change goes through. `record` is false when replaying,
// so undo does not itself become undoable.
function applyEdit(oid, field, value, record = true) {
  const before = state.get(oid)[field];
  if (JSON.stringify(before) === JSON.stringify(value)) return;
  applyToMesh(oid, field, value);
  if (record) { history.push({ oid, field, before, after: value });
                redoStack.length = 0; }
  state.get(oid)[field] = value;
  send(oid, field, value);
  document.getElementById('hist').textContent = `history: ${history.length}`;
  if (selected && selected.userData.objectId === oid) buildInspector(selected);
}

function undo() {
  const e = history.pop();
  if (!e) return;
  redoStack.push(e);
  applyEdit(e.oid, e.field, e.before, false);
  document.getElementById('hist').textContent = `history: ${history.length}`;
}

function redo() {
  const e = redoStack.pop();
  if (!e) return;
  history.push(e);
  applyEdit(e.oid, e.field, e.after, false);
  document.getElementById('hist').textContent = `history: ${history.length}`;
}

// Reset is not a special operation either: it is every field set back to the
// value it was first rendered with.
function reset() {
  for (const [oid, fields] of initial) {
    for (const [field, value] of Object.entries(fields)) {
      applyEdit(oid, field, value, false);
    }
  }
  history.length = 0; redoStack.length = 0;
  document.getElementById('hist').textContent = 'history: 0';
}

// ---- inspector -----------------------------------------------------------
function buildInspector(mesh) {
  const fields = document.getElementById('fields');
  if (!mesh) { fields.textContent = 'select an object'; return; }
  const oid = mesh.userData.objectId, shape = SHAPES[String(oid)];
  const current = state.get(oid);

  const rows = ['x', 'y', 'z'].map((axis, i) =>
    `<label>position ${axis}<input data-field="position" data-i="${i}"
       type="number" step="0.1" value="${current.position[i].toFixed(3)}"></label>`);

  if (shape) {
    const v = Array.isArray(current[shape.attr]) ? current[shape.attr]
                                                 : [current[shape.attr]];
    v.forEach((n, i) => rows.push(
      `<label>${shape.attr}${v.length > 1 ? ' ' + i : ''}<input
         data-field="${shape.attr}" data-i="${i}" type="number" step="0.1"
         min="0.001" value="${n.toFixed(3)}"></label>`));
  }
  const pol = POLARIZATION[String(oid)];
  if (pol) {
    current[pol.attr].forEach((n, i) => rows.push(
      `<label>${pol.attr} ${'xyz'[i]}<input data-field="${pol.attr}" data-i="${i}"
         type="number" step="0.1" value="${n.toFixed(3)}"></label>`));
    rows.push('<small>direction needs magpylib to redraw;<br>' +
              'amplitude changes nothing visible</small>');
  }
  fields.innerHTML = rows.join('');

  fields.querySelectorAll('input').forEach(input => {
    input.addEventListener('change', () => {
      const i = Number(input.dataset.i), n = Number(input.value);
      const field = input.dataset.field;
      let value = state.get(oid)[field];
      if (Array.isArray(value)) { value = value.slice(); value[i] = n; }
      else { value = n; }
      if (shape && field === shape.attr && shape.constraint === 'uniform'
          && Array.isArray(value)) value.fill(n);
      applyEdit(oid, field, value);
    });
  });
}

function select(mesh) {
  selected = mesh;
  buildInspector(mesh);
  if (!mesh) { gizmo.detach();
    document.getElementById('sel').textContent = 'nothing selected';
    document.getElementById('delta').textContent = ''; return; }
  gizmo.attach(mesh);
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

// ---- keyboard ------------------------------------------------------------
let snapping = false;
function setSnapping(on) {
  snapping = on;
  gizmo.setTranslationSnap(on ? 0.1 : null);
  gizmo.setRotationSnap(on ? THREE.MathUtils.degToRad(15) : null);
  gizmo.setScaleSnap(on ? 0.1 : null);
  document.getElementById('snap').textContent =
    `snap: ${on ? '0.1 / 15deg' : 'off'}`;
}

addEventListener('keydown', e => {
  // otherwise typing a value in the inspector also fires the shortcuts
  if (e.target.tagName === 'INPUT') return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
    e.shiftKey ? redo() : undo();
    return;
  }
  if (e.key === 'Backspace') reset();
  if (e.key === 's') setSnapping(!snapping);
  if (e.key === 'w') gizmo.setMode('translate');
  if (e.key === 'e') gizmo.setMode('rotate');
  if (e.key === 'r') {
    const shape = selected && SHAPES[String(selected.userData.objectId)];
    if (shape) gizmo.setMode('scale');
    else document.getElementById('delta').textContent =
      'this object has no scale-covariant shape parameter';
  }
  if (e.key === 'Escape') select(null);
});

// ---- during the drag: JS only, nothing sent ------------------------------
gizmo.addEventListener('objectChange', () => {
  const shape = SHAPES[String(selected.userData.objectId)];
  if (gizmo.mode === 'scale' && shape) {
    if (shape.constraint === 'uniform') {
      selected.scale.setScalar(Math.max(selected.scale.x, selected.scale.y,
                                        selected.scale.z));
    } else if (shape.constraint === 'xy') {
      const s = Math.abs(selected.scale.x - 1) > Math.abs(selected.scale.y - 1)
        ? selected.scale.x : selected.scale.y;
      selected.scale.set(s, s, selected.scale.z);
    }
  }
  const p = selected.position;
  document.getElementById('delta').textContent =
    `at (${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}) ${UNIT}`;
});

// ---- on drop: one message, through the same door as everything else ------
gizmo.addEventListener('mouseUp', () => {
  if (!selected) return;
  const oid = selected.userData.objectId, shape = SHAPES[String(oid)];
  if (gizmo.mode === 'translate') {
    applyEdit(oid, 'position', selected.position.toArray());
  } else if (gizmo.mode === 'rotate') {
    applyEdit(oid, 'quaternion', selected.quaternion.toArray());
  } else if (gizmo.mode === 'scale' && shape) {
    const base = Array.isArray(shape.value) ? shape.value : [shape.value];
    const s = selected.scale;
    const next = shape.constraint === 'uniform' ? base[0] * s.x
      : shape.constraint === 'xy' ? [base[0] * s.x, base[1] * s.z]
      : [base[0] * s.x, base[1] * s.y, base[2] * s.z];
    applyEdit(oid, shape.attr, next);
  }
});
"""


#: Classes whose mesh is *exactly* the base mesh scaled, so a resize can be
#: previewed with `mesh.scale` and magpylib told only the final value. Every
#: entry below was verified by rendering twice and comparing vertices.
#:
#: `constraint` records which scale axes are independent, which follows from
#: how many numbers the parameter has: a Cylinder's `dimension` is
#: (diameter, height), so x and y are locked together and z is free.
#:
#: Excluded, and why:
#:  * `CylinderSegment` -- its angles do not scale with its radii.
#:  * `Sensor` -- a composite. Its pixels sit at real coordinates while the
#:    cross is styled, so `style.size` scales only part of the mesh; scaling
#:    the whole thing would displace the pixels.
#:  * meshes (`Tetrahedron`, `TriangularMesh`) -- vertices are the parameter,
#:    so there is no single scale to drag.
SCALE_COVARIANT = {
    "Cuboid": ("dimension", "free"),
    "Sphere": ("diameter", "uniform"),
    "Cylinder": ("dimension", "xy"),
    # A Dipole has no physical size at all -- its arrow is styled geometry, and
    # `style.size` is one scalar, so the resize is uniform by construction and
    # there is no aspect ratio to preserve. Only holds with sizemode="absolute";
    # under "scaled" the value is a multiplier on the scene-derived autosize.
    "Dipole": ("style.size", "uniform"),
}


def _resolve(obj, path):
    """Read a dotted attribute path, falling back to the library default.

    An unset style property reads as `None` on the object and only takes its
    value from `magpy.defaults.display.style.<family>` when the figure is
    drawn. `merged()` does not help: it resolves set-vs-inherited *within* the
    object's own tree, so `Dipole().style.merged().size` is still `None` while
    the effective size is 1. A host reading style has to consult the default
    itself.
    """
    value = obj
    for part in path.split("."):
        value = getattr(value, part)
    if value is None and path.startswith("style."):
        node = magpy.defaults.display.style
        for part in (type(obj).__name__.lower(), *path.split(".")[1:]):
            node = getattr(node, part)
        value = node
    return value


def shape_of(obj):
    """The scale-covariant shape parameter of `obj`, or None."""
    entry = SCALE_COVARIANT.get(type(obj).__name__)
    if entry is None:
        return None
    attr, constraint = entry
    value = _resolve(obj, attr)
    return {
        "kind": type(obj).__name__,
        "attr": attr,
        "value": value.tolist() if hasattr(value, "tolist") else float(value),
        "constraint": constraint,
    }


def polarization_of(obj):
    """The polarization vector of `obj`, or None if it has none.

    Reported in the object's **local** frame, which is how magpylib stores it.
    The rendered `intensity` is the vertex projected on the *world* vector, so
    the two differ by the object's orientation -- reading the attribute as if
    it were world-space is wrong for anything rotated (measured: 0.44 off for a
    50 degree rotation). That is one reason this edit is not previewed in the
    browser; see `_MAGNETIZATION_NOTE`.
    """
    pol = getattr(obj, "polarization", None)
    if pol is None:
        return None
    return {
        "attr": "polarization",
        "value": [float(component) for component in pol],
    }


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
    shapes = {
        str(oid): s for oid, obj in registry.items() if (s := shape_of(obj)) is not None
    }
    polarizations = {
        str(oid): p
        for oid, obj in registry.items()
        if (p := polarization_of(obj)) is not None
    }
    html = html.replace(
        # the addon import has to precede the module body
        "import { OrbitControls }",
        _IMPORT.strip() + "\nimport { OrbitControls }",
    ).replace(
        "const DATA =",
        f"const UNIT = 'm';\nconst SHAPES = {json.dumps(shapes)};\n"
        f"const POLARIZATION = {json.dumps(polarizations)};\nconst DATA =",
    )

    page = HERE / "interactive.html"
    page.write_text(html)
    print(f"wrote {page}")
    webbrowser.open(f"file://{page}")


if __name__ == "__main__":
    main()

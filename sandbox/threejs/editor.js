// Editor layer for the prototype three.js backend: picking, gizmos, an
// inspector and an edit history. Loaded verbatim by interactive.py and
// appended to the page the backend renders, so it runs in the same module
// scope and can use `scene`, `camera`, `controls` and `byObjectId`.
//
// Kept as a real .js file rather than a Python string: it is the larger
// half of the program, and it wants syntax highlighting and `node --check`
// more than it wants to be quoted.

// ---- pick, drag, and an edit history -------------------------------------
// Everything here runs in the browser. Nothing calls back into Python until an
// edit is committed, which is the whole point: see the round-trip counter.
//
// Every edit -- gizmo drop, typed value, undo, redo -- goes through applyEdit
// and emits the same {object_id, field, value} message. Undo is therefore not
// a separate mechanism: it is an edit whose value happens to be the previous
// one. The browser holds the history only so it can name that value; the model
// is the magpylib object, and a real host would replay these against it.

const hud = document.createElement("div");
hud.id = "hud";
hud.innerHTML = `<b>click an object</b><br>
  <span id="sel">nothing selected</span><br>
  <span id="delta"></span><br>
  <span id="calls">python round-trips: 0</span><br>
  <span id="snap">snap: off</span> &middot; <span id="hist">history: 0</span><br>
  <small>W move &middot; E rotate &middot; R resize &middot; S snap<br>
  P polarization &middot; &#8984;/Ctrl+Z undo &middot; &#8679;+Z redo<br>
  Backspace reset</small>`;
document.body.appendChild(hud);

const inspector = document.createElement("div");
inspector.id = "inspector";
inspector.innerHTML = `<b>exact values</b><div id="fields">select an object</div>`;
document.body.appendChild(inspector);

const style = document.createElement("style");
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
gizmo.addEventListener("dragging-changed", (e) => {
  controls.enabled = !e.value;
});
scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);

let selected = null,
  roundTrips = 0;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const pickable = [...byObjectId.values()]; // lines and points are decoration

// ---- state, so an edit knows what it is replacing -------------------------
const state = new Map(),
  initial = new Map();
for (const [oid, mesh] of byObjectId) {
  const shape = SHAPES[String(oid)];
  const s = {
    position: mesh.position.toArray(),
    quaternion: mesh.quaternion.toArray(),
  };
  if (shape) s[shape.attr] = shape.value;
  const pol = POLARIZATION[String(oid)];
  if (pol) s[pol.attr] = pol.value;
  state.set(oid, s);
  initial.set(oid, JSON.parse(JSON.stringify(s)));
}

const history = [],
  redoStack = [];

// ---- polarization gizmo ---------------------------------------------------
// The arrow shows the *world* direction, which is the object's orientation
// applied to the stored local vector -- magpylib keeps polarization in the
// body frame. Rotating the arrow therefore sets a world direction that has to
// be converted back through the inverse of the mesh's quaternion before it can
// be sent. Getting that wrong is invisible until an object is rotated.
const polGroup = new THREE.Group();
const polArrow = new THREE.ArrowHelper(
  new THREE.Vector3(0, 0, 1),
  new THREE.Vector3(),
  1,
  0xd62728,
  0.3,
  0.18,
);
polGroup.add(polArrow);
polGroup.visible = false;
scene.add(polGroup);
const Z = new THREE.Vector3(0, 0, 1);

function worldPolarization(oid) {
  const local = new THREE.Vector3().fromArray(
    state.get(oid)[POLARIZATION[String(oid)].attr],
  );
  return local.applyQuaternion(byObjectId.get(oid).quaternion);
}

/** Park the arrow on `mesh`, pointing along its world polarization. */
function placePolarization(mesh) {
  const oid = mesh && mesh.userData.objectId;
  if (!mesh || !POLARIZATION[String(oid)]) {
    polGroup.visible = false;
    return;
  }
  const dir = worldPolarization(oid);
  if (dir.lengthSq() === 0) {
    polGroup.visible = false;
    return;
  }
  mesh.geometry.computeBoundingSphere();
  const span =
    mesh.geometry.boundingSphere.radius *
    Math.max(mesh.scale.x, mesh.scale.y, mesh.scale.z);
  polArrow.setLength(span * 1.8, span * 0.5, span * 0.3);
  polGroup.position.copy(mesh.position);
  polGroup.quaternion.setFromUnitVectors(Z, dir.clone().normalize());
  polGroup.visible = true;
}

function applyToMesh(oid, field, value) {
  const mesh = byObjectId.get(oid),
    shape = SHAPES[String(oid)];
  if (field === "position") {
    mesh.position.fromArray(value);
    return;
  }
  if (field === "quaternion") {
    mesh.quaternion.fromArray(value);
    return;
  }
  // Polarization is the one edit with no preview here. It changes `intensity`
  // -- a per-vertex attribute -- and magpylib computes that from the vector in
  // *world* space, i.e. the object's orientation applied to the stored local
  // vector. Recomputing it in the browser means reimplementing that frame
  // convention, and getting it subtly wrong for every rotated object. Ask
  // magpylib instead: one object re-renders in 0.3 ms.
  if (field === "polarization") {
    placePolarization(mesh);
    return;
  }
  // a shape value is expressed as a scale of the base mesh
  const base = Array.isArray(shape.value) ? shape.value : [shape.value];
  const v = Array.isArray(value) ? value : [value];
  const s = v.map((n, j) => n / base[j]);
  if (shape.constraint === "uniform") mesh.scale.setScalar(s[0]);
  else if (shape.constraint === "xy") mesh.scale.set(s[0], s[0], s[1]);
  else mesh.scale.set(s[0], s[1], s[2]);
}

function send(oid, field, value) {
  roundTrips += 1;
  document.getElementById("calls").textContent =
    `python round-trips: ${roundTrips}`;
  console.log("would send to python:", { object_id: oid, [field]: value });
}

// The single door every change goes through. `record` is false when replaying,
// so undo does not itself become undoable.
function applyEdit(oid, field, value, record = true) {
  const before = state.get(oid)[field];
  if (JSON.stringify(before) === JSON.stringify(value)) return;
  applyToMesh(oid, field, value);
  if (record) {
    history.push({ oid, field, before, after: value });
    redoStack.length = 0;
  }
  state.get(oid)[field] = value;
  send(oid, field, value);
  document.getElementById("hist").textContent = `history: ${history.length}`;
  if (selected && selected.userData.objectId === oid) buildInspector(selected);
}

function undo() {
  const e = history.pop();
  if (!e) return;
  redoStack.push(e);
  applyEdit(e.oid, e.field, e.before, false);
  document.getElementById("hist").textContent = `history: ${history.length}`;
}

function redo() {
  const e = redoStack.pop();
  if (!e) return;
  history.push(e);
  applyEdit(e.oid, e.field, e.after, false);
  document.getElementById("hist").textContent = `history: ${history.length}`;
}

// Reset is not a special operation either: it is every field set back to the
// value it was first rendered with.
function reset() {
  for (const [oid, fields] of initial) {
    for (const [field, value] of Object.entries(fields)) {
      applyEdit(oid, field, value, false);
    }
  }
  history.length = 0;
  redoStack.length = 0;
  document.getElementById("hist").textContent = "history: 0";
}

// ---- inspector -----------------------------------------------------------
function buildInspector(mesh) {
  const fields = document.getElementById("fields");
  if (!mesh) {
    fields.textContent = "select an object";
    return;
  }
  const oid = mesh.userData.objectId,
    shape = SHAPES[String(oid)];
  const current = state.get(oid);

  const rows = ["x", "y", "z"].map(
    (axis, i) =>
      `<label>position ${axis}<input data-field="position" data-i="${i}"
       type="number" step="0.1" value="${current.position[i].toFixed(3)}"></label>`,
  );

  if (shape) {
    const v = Array.isArray(current[shape.attr])
      ? current[shape.attr]
      : [current[shape.attr]];
    v.forEach((n, i) =>
      rows.push(
        `<label>${shape.attr}${v.length > 1 ? " " + i : ""}<input
         data-field="${shape.attr}" data-i="${i}" type="number" step="0.1"
         min="0.001" value="${n.toFixed(3)}"></label>`,
      ),
    );
  }
  const pol = POLARIZATION[String(oid)];
  if (pol) {
    current[pol.attr].forEach((n, i) =>
      rows.push(
        `<label>${pol.attr} ${"xyz"[i]}<input data-field="${pol.attr}" data-i="${i}"
         type="number" step="0.1" value="${n.toFixed(3)}"></label>`,
      ),
    );
    rows.push(
      "<small>direction needs magpylib to redraw;<br>" +
        "amplitude changes nothing visible</small>",
    );
  }
  fields.innerHTML = rows.join("");

  fields.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", () => {
      const i = Number(input.dataset.i),
        n = Number(input.value);
      const field = input.dataset.field;
      let value = state.get(oid)[field];
      if (Array.isArray(value)) {
        value = value.slice();
        value[i] = n;
      } else {
        value = n;
      }
      if (
        shape &&
        field === shape.attr &&
        shape.constraint === "uniform" &&
        Array.isArray(value)
      )
        value.fill(n);
      applyEdit(oid, field, value);
    });
  });
}

function select(mesh) {
  selected = mesh;
  buildInspector(mesh);
  if (!mesh) {
    gizmo.detach();
    polGroup.visible = false;
    document.getElementById("sel").textContent = "nothing selected";
    document.getElementById("delta").textContent = "";
    return;
  }
  gizmo.attach(mesh);
  placePolarization(mesh);
  document.getElementById("sel").textContent = `selected: ${mesh.name}`;
  document.getElementById("delta").textContent = "drag a handle";
}

renderer.domElement.addEventListener("pointerdown", (event) => {
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
  document.getElementById("snap").textContent =
    `snap: ${on ? "0.1 / 15deg" : "off"}`;
}

addEventListener("keydown", (e) => {
  // otherwise typing a value in the inspector also fires the shortcuts
  if (e.target.tagName === "INPUT") return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
    e.shiftKey ? redo() : undo();
    return;
  }
  if (e.key === "Backspace") reset();
  if (e.key === "s") setSnapping(!snapping);
  if (e.key === "w") gizmo.setMode("translate");
  if (e.key === "e") gizmo.setMode("rotate");
  if (e.key === "r") {
    const shape = selected && SHAPES[String(selected.userData.objectId)];
    if (shape) gizmo.setMode("scale");
    else
      document.getElementById("delta").textContent =
        "this object has no scale-covariant shape parameter";
  }
  if (e.key === "p") {
    if (selected && POLARIZATION[String(selected.userData.objectId)]) {
      gizmo.attach(polGroup);
      gizmo.setMode("rotate");
      document.getElementById("delta").textContent =
        "rotating polarization -- colours redraw once magpylib answers";
    } else {
      document.getElementById("delta").textContent =
        "this object has no polarization";
    }
  }
  if (e.key === "Escape") select(null);
});

// ---- during the drag: JS only, nothing sent ------------------------------
gizmo.addEventListener("objectChange", () => {
  const shape = SHAPES[String(selected.userData.objectId)];
  if (gizmo.mode === "scale" && shape) {
    if (shape.constraint === "uniform") {
      selected.scale.setScalar(
        Math.max(selected.scale.x, selected.scale.y, selected.scale.z),
      );
    } else if (shape.constraint === "xy") {
      const s =
        Math.abs(selected.scale.x - 1) > Math.abs(selected.scale.y - 1)
          ? selected.scale.x
          : selected.scale.y;
      selected.scale.set(s, s, selected.scale.z);
    }
  }
  if (gizmo.object !== polGroup) placePolarization(selected);
  const p = selected.position;
  document.getElementById("delta").textContent =
    `at (${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}) ${UNIT}`;
});

// ---- on drop: one message, through the same door as everything else ------
gizmo.addEventListener("mouseUp", () => {
  if (!selected) return;
  const oid = selected.userData.objectId,
    shape = SHAPES[String(oid)];
  if (gizmo.object === polGroup) {
    const pol = POLARIZATION[String(oid)];
    // amplitude is not what the arrow expresses, so it is carried through
    const magnitude = new THREE.Vector3()
      .fromArray(state.get(oid)[pol.attr])
      .length();
    // the arrow points in world space; magpylib stores the body frame
    const local = Z.clone()
      .applyQuaternion(polGroup.quaternion)
      .applyQuaternion(selected.quaternion.clone().invert())
      .multiplyScalar(magnitude);
    applyEdit(
      oid,
      pol.attr,
      local.toArray().map((n) => Number(n.toFixed(6))),
    );
    return;
  }
  if (gizmo.mode === "translate") {
    applyEdit(oid, "position", selected.position.toArray());
  } else if (gizmo.mode === "rotate") {
    applyEdit(oid, "quaternion", selected.quaternion.toArray());
  } else if (gizmo.mode === "scale" && shape) {
    const base = Array.isArray(shape.value) ? shape.value : [shape.value];
    const s = selected.scale;
    const next =
      shape.constraint === "uniform"
        ? base[0] * s.x
        : shape.constraint === "xy"
          ? [base[0] * s.x, base[1] * s.z]
          : [base[0] * s.x, base[1] * s.y, base[2] * s.z];
    applyEdit(oid, shape.attr, next);
  }
});

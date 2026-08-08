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

// ---- chrome ---------------------------------------------------------------
// Four panels sharing one visual language: status top-left, the viewer's own
// legend beneath it, controls bottom-left, inspector right.

const style = document.createElement("style");
style.textContent = `
:root {
  --panel: rgba(255,255,255,.93);
  --edge: rgba(15,23,42,.10);
  --ink: #1f2328;
  --muted: #6b7280;
  --shadow: 0 1px 2px rgba(15,23,42,.05), 0 6px 20px rgba(15,23,42,.08);
}
.mp-panel {
  position: absolute; z-index: 5;
  background: var(--panel); color: var(--ink);
  border: 1px solid var(--edge); border-radius: 9px;
  box-shadow: var(--shadow); backdrop-filter: blur(8px);
  padding: 9px 11px;
  font: 12px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.mp-title {
  font-size: 10px; font-weight: 600; letter-spacing: .07em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
#legend { top: 12px; left: 12px; }
#hud    { bottom: 42px; left: 12px; }
#inspector { top: 12px; right: 12px; min-width: 172px; }
#code {
  display: none; left: 50%; top: 50%; transform: translate(-50%, -50%);
  max-width: min(680px, 82vw); max-height: 70vh; overflow: auto;
}
#code pre {
  margin: 0; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre; color: #1f2328;
}
#code .hint { float: right; text-transform: none; letter-spacing: 0; }

/* a status bar rather than another floating panel: connection state is about
   the session, not about anything in the scene */
#statusbar {
  position: absolute; left: 0; right: 0; bottom: 0; z-index: 6;
  display: flex; align-items: center; gap: 9px;
  height: 26px; padding: 0 12px; box-sizing: border-box;
  background: rgba(248,249,251,.94); border-top: 1px solid var(--edge);
  backdrop-filter: blur(8px);
  font: 11px/1 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--ink);
}
#statusbar i {
  width: 8px; height: 8px; border-radius: 50%; flex: none; background: #c3c8d0;
  box-shadow: 0 0 0 3px rgba(0,0,0,.04);
}
#statusbar .sep { flex: 1; }
#statusbar .meta { color: var(--muted); }
#statusbar[data-state="live"] i { background: #22a06b; box-shadow: 0 0 0 3px rgba(34,160,107,.16); }
#statusbar[data-state="busy"] i { background: #e8912d; box-shadow: 0 0 0 3px rgba(232,145,45,.18); }
#statusbar[data-state="dead"] i { background: #d64545; box-shadow: 0 0 0 3px rgba(214,69,69,.18); }
#statusbar[data-state="dead"] { background: #fdecec; border-top-color: rgba(214,69,69,.35); }
/* aiming the polarization edits the magnet's field, not its placement, so the
   viewport gets a border and the mode is named in the bar */
body.aiming::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 4;
  border: 2px solid rgba(214,69,69,.45); box-sizing: border-box;
}
body.aiming #mode { color: #d64545; font-weight: 600; }

kbd {
  font: 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #f4f5f7; border: 1px solid var(--edge); border-bottom-width: 2px;
  border-radius: 4px; padding: 2px 5px; color: #3d4451;
}
#controls > summary {
  cursor: pointer; list-style: none; user-select: none;
  margin: 0 0 6px; display: flex; align-items: center; gap: 5px;
}
#controls > summary::-webkit-details-marker { display: none; }
#controls > summary::before {
  content: ""; width: 0; height: 0; flex: none;
  border: 4px solid transparent; border-left-color: var(--ink);
  transition: transform .14s ease;
}
#controls > summary:hover { color: var(--ink); }
#controls > summary:hover::before { border-left-color: #2e91e5; }
#controls[open] > summary::before { transform: rotate(90deg) translateX(-1px); }
#controls:not([open]) > summary { margin-bottom: 0; }
#keys { display: grid; grid-template-columns: auto 1fr; gap: 3px 8px; align-items: center; }
#keys span { color: var(--muted); }

#readout { margin-bottom: 7px; }
#sel { font-weight: 600; }
#delta { color: var(--muted); }
#stats { color: var(--muted); font-size: 11px; margin: 5px 0 8px; }

#fields .row {
  display: grid; grid-template-columns: 1fr auto; align-items: center;
  gap: 10px; margin-bottom: 3px;
}
#fields .row span { color: var(--muted); }
#fields input {
  width: 6.2em; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
  padding: 2px 6px; color: var(--ink);
  border: 1px solid var(--edge); border-radius: 5px; background: #fff;
}
#fields input:focus { outline: 2px solid rgba(46,145,229,.35); outline-offset: 0; }
#fields .note { color: var(--muted); font-size: 11px; display: block; margin-top: 6px; }
#fields .group { margin-top: 8px; }
`;
document.head.appendChild(style);

const status = document.createElement("div");
status.id = "statusbar";
status.innerHTML =
  `<i></i><span id="statustext">connecting</span>` +
  `<span class="meta">&middot;</span><span id="mode">mode: translate</span>` +
  `<span class="meta">&middot;</span><span id="delta"></span>` +
  `<span class="sep"></span>` +
  `<span id="calls">0 round-trips</span>` +
  `<span class="meta">&middot;</span><span id="hist">0 edits</span>` +
  `<span class="meta">&middot;</span><span id="snap">snap: off</span>` +
  `<span class="meta">&middot;</span><span id="axis">axes: all</span>` +
  `<span class="meta">&middot;</span><span id="space">space: world</span>` +
  `<span class="meta">&middot;</span><span id="pivot">pivot: origin</span>` +
  `<span class="meta">&middot;</span><span id="proj">perspective</span>` +
  `<span class="meta">&middot;</span><span id="shown">all shown</span>` +
  `<span class="meta">&middot; ${INFO.backend} &middot; magpylib ` +
  `${INFO.magpylib} &middot; python ${INFO.python}</span>`;
document.body.appendChild(status);

function setStatus(state, text) {
  status.dataset.state = state;
  document.getElementById("statustext").textContent = text;
}

// An editor is only useful while python is listening, so the page says whether
// it is -- and notices when it stops, rather than silently accepting edits
// that go nowhere.
if (!INFO.alive) {
  setStatus("static", "no backend \u2014 saved page");
} else {
  let missed = 0;
  const beat = async () => {
    try {
      const response = await fetch(INFO.alive, { cache: "no-store" });
      if (!response.ok) throw new Error(response.status);
      missed = 0;
      setStatus("live", "backend connected");
    } catch {
      missed += 1;
      if (missed >= 2)
        setStatus("dead", "backend gone \u2014 edits go nowhere");
    }
  };
  beat();
  setInterval(beat, 2000);
}

const legendBox = document.getElementById("legend");
if (legendBox) legendBox.className = "mp-panel";

const hud = document.createElement("div");
hud.id = "hud";
hud.className = "mp-panel";
hud.innerHTML = `
  <details id="controls">
  <summary class="mp-title">controls</summary>
  <div id="keys">
    <kbd>W</kbd><span>move</span>
    <kbd>E</kbd><span>rotate</span>
    <kbd>R</kbd><span>resize</span>
    <kbd>P</kbd><span>aim polarization</span>
    <kbd>S</kbd><span>snap to grid</span>
    <kbd>X</kbd><span>constrain axis (<kbd>A</kbd> all)</span>
    <kbd>L</kbd><span>local / world space</span>
    <kbd>F</kbd><span>frame selected</span>
    <kbd>\u21b9</kbd><span>next object</span>
    <kbd>C</kbd><span>export magpylib code</span>
    <kbd>\u2318Z</kbd><span>undo / <kbd>\u21e7</kbd> redo</span>
    <kbd>\u232b</kbd><span>reset</span>
    <kbd>H</kbd><span>hide (<kbd>\u21e7</kbd> isolate)</span>
    <kbd>1</kbd><span>front &middot; <kbd>3</kbd> right &middot; <kbd>7</kbd> top</span>
    <kbd>5</kbd><span>ortho / perspective</span>
    <kbd>,</kbd><span>pivot origin / centre</span>
    <kbd>\u21e7</kbd><span>click to multi-select</span>
  </div>
  </details>`;
document.body.appendChild(hud);

const inspector = document.createElement("div");
inspector.id = "inspector";
inspector.className = "mp-panel";
inspector.innerHTML =
  `<div class="mp-title" id="sel">values</div>` +
  `<div id="fields">select an object</div>`;
document.body.appendChild(inspector);

const code = document.createElement("div");
code.id = "code";
code.className = "mp-panel";
code.innerHTML =
  `<div class="mp-title">magpylib code <span class="hint"></span></div>` +
  `<pre></pre>`;
document.body.appendChild(code);

const gizmo = new TransformControls(camera, renderer.domElement);
// OrbitControls must yield while the gizmo has the pointer, or dragging a
// handle also spins the camera.
gizmo.addEventListener("dragging-changed", (e) => {
  controls.enabled = !e.value;
});
scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);

let selected = null,
  roundTrips = 0;

// Multi-select drives a proxy rather than a mesh. `Object3D.attach` re-parents
// while preserving world transform, so moving the proxy moves the group and
// each member's own transform stays readable when it is detached again.
const perspective = camera; // the viewer's own camera, kept to swap back to
const selection = [];
const proxy = new THREE.Object3D();
scene.add(proxy);
let pivotAtOrigin = true; // vs the bounding-box centre of the selection

function selectionCentre() {
  const box = new THREE.Box3();
  for (const mesh of selection) box.expandByObject(mesh);
  return pivotAtOrigin && selection.length === 1
    ? selection[0].position.clone()
    : box.getCenter(new THREE.Vector3());
}

function bindProxy() {
  for (const mesh of [...proxy.children]) scene.attach(mesh);
  if (!selection.length) {
    gizmo.detach();
    return;
  }
  proxy.position.copy(selectionCentre());
  proxy.quaternion.identity();
  proxy.scale.set(1, 1, 1);
  for (const mesh of selection) proxy.attach(mesh);
  gizmo.attach(proxy);
}

function releaseProxy() {
  for (const mesh of [...proxy.children]) scene.attach(mesh);
}
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

// ---- aiming the polarization ----------------------------------------------
// An empty handle the rotate gizmo can drive. There is no arrow: the object's
// own N/S colouring already shows the direction and re-projects live as the
// handle turns, so a second indicator only competed with the first.
//
// The handle points along the *world* direction, which is the object's
// orientation applied to the stored vector -- magpylib keeps polarization in
// the body frame. Committing therefore means converting back through the
// inverse of the mesh's world quaternion. Getting that wrong stays invisible
// until an object is rotated.
const polGroup = new THREE.Group();
polGroup.visible = false;
scene.add(polGroup);
const Z = new THREE.Vector3(0, 0, 1);

function worldPolarization(oid) {
  const local = new THREE.Vector3().fromArray(
    state.get(oid)[POLARIZATION[String(oid)].attr],
  );
  // world, not local: while a drag is in progress the mesh is parented to the
  // proxy, so its own quaternion is relative to that rather than to the scene
  return local.applyQuaternion(
    byObjectId.get(oid).getWorldQuaternion(new THREE.Quaternion()),
  );
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
  polGroup.position.copy(mesh.getWorldPosition(new THREE.Vector3()));
  polGroup.quaternion.setFromUnitVectors(Z, dir.clone().normalize());
  polGroup.visible = true;
}

/** Re-project the colorscale onto `mesh` for a new body-frame direction.
 *
 * A *preview*. magpylib derives `intensity` the same way -- the vertex
 * projected on the magnetization axis, normalised -- and this reproduces it
 * exactly for the convex primitives (checked to 2e-16). It is still a
 * duplicate of magpylib's convention, in the same class as previewing a
 * resize with mesh.scale: the authoritative value is whatever magpylib
 * returns when the edit is committed.
 */
function updateGradient(mesh, localDir) {
  const uv = mesh.geometry.attributes.uv;
  if (!uv) return; // flat or facecolor mesh: nothing to re-project
  const pos = mesh.geometry.attributes.position;
  const d = localDir.clone().normalize();
  const proj = new Float64Array(pos.count);
  let min = Infinity,
    max = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    const p = pos.getX(i) * d.x + pos.getY(i) * d.y + pos.getZ(i) * d.z;
    proj[i] = p;
    if (p < min) min = p;
    if (p > max) max = p;
  }
  const span = max - min || 1;
  for (let i = 0; i < pos.count; i++) uv.setX(i, (proj[i] - min) / span);
  uv.needsUpdate = true;
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
    updateGradient(mesh, new THREE.Vector3().fromArray(value));
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

async function send(oid, field, value) {
  roundTrips += 1;
  document.getElementById("calls").textContent = `${roundTrips} round-trips`;
  if (!INFO.root) return; // a saved page has nobody to tell
  setStatus("busy", "sending");
  try {
    await fetch(INFO.root, {
      method: "POST",
      body: JSON.stringify({ object_id: oid, [field]: value }),
    });
    setStatus("live", "backend connected");
  } catch {
    setStatus("dead", "backend gone \u2014 edits go nowhere");
  }
}

// ---- export ---------------------------------------------------------------
// The point of the round-trip: python holds the edited objects, so it can
// print them back as a script. Reading it from the server rather than
// rebuilding it here is what makes it trustworthy -- it is the objects, not
// the browser's idea of them.
async function exportCode() {
  if (!INFO.root) {
    setStatus("static", "no backend \u2014 cannot export");
    return;
  }
  const code = await (
    await fetch(INFO.root + "export", { cache: "no-store" })
  ).text();
  const panel = document.getElementById("code");
  panel.querySelector("pre").textContent = code;
  panel.style.display = "block";
  try {
    await navigator.clipboard.writeText(code);
    panel.querySelector(".hint").textContent = "copied to clipboard";
  } catch {
    panel.querySelector(".hint").textContent = "select and copy";
  }
}

// The single door every change goes through. `record` is false when replaying,
// so undo does not itself become undoable.
function applyEdit(oid, field, value, record = true) {
  const before = state.get(oid)[field];
  if (JSON.stringify(before) === JSON.stringify(value)) return;
  // State first: applyToMesh re-derives the view from it -- the polarization
  // arrow is placed by reading it back -- so updating afterwards would redraw
  // from the value being replaced, and the arrow would snap to where it was.
  state.get(oid)[field] = value;
  applyToMesh(oid, field, value);
  if (record) {
    history.push({ oid, field, before, after: value });
    redoStack.length = 0;
  }
  send(oid, field, value);
  document.getElementById("hist").textContent = `${history.length} edits`;
  if (selected && selected.userData.objectId === oid) buildInspector(selected);
}

function undo() {
  const e = history.pop();
  if (!e) return;
  redoStack.push(e);
  applyEdit(e.oid, e.field, e.before, false);
  document.getElementById("hist").textContent = `${history.length} edits`;
}

function redo() {
  const e = redoStack.pop();
  if (!e) return;
  history.push(e);
  applyEdit(e.oid, e.field, e.after, false);
  document.getElementById("hist").textContent = `${history.length} edits`;
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
  document.getElementById("hist").textContent = "0 edits";
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

  const row = (label, field, i, value, step = 0.1) =>
    `<div class="row"><span>${label}</span><input data-field="${field}"` +
    ` data-i="${i}" type="number" step="${step}" value="${value.toFixed(3)}"></div>`;

  const rows = ["x", "y", "z"].map((axis, i) =>
    row(`position ${axis}`, "position", i, current.position[i]),
  );

  if (shape) {
    const v = Array.isArray(current[shape.attr])
      ? current[shape.attr]
      : [current[shape.attr]];
    v.forEach((n, i) =>
      rows.push(
        row(`${shape.attr}${v.length > 1 ? " " + i : ""}`, shape.attr, i, n),
      ),
    );
  }
  const pol = POLARIZATION[String(oid)];
  if (pol) {
    rows.push('<div class="group"></div>');
    current[pol.attr].forEach((n, i) =>
      rows.push(row(`polarization ${"xyz"[i]}`, pol.attr, i, n)),
    );
    // Amplitude is not a fourth component: it is the length of the vector, and
    // the gizmo cannot express it because nothing in the render depends on it.
    // Typing it here rescales the vector and leaves the direction alone.
    rows.push(
      row(
        "|polarization|",
        "polarization-magnitude",
        0,
        Math.hypot(...current[pol.attr]),
        0.05,
      ),
    );
    rows.push(
      '<small class="note">direction redraws once magpylib answers;' +
        "<br>amplitude has no visible effect at all</small>",
    );
  }
  fields.innerHTML = rows.join("");

  fields.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", () => {
      const i = Number(input.dataset.i),
        n = Number(input.value);
      const field = input.dataset.field;
      if (field === "polarization-magnitude") {
        const vector = state.get(oid)[pol.attr];
        const length = Math.hypot(...vector) || 1;
        applyEdit(
          oid,
          pol.attr,
          vector.map((c) => (c / length) * n),
        );
        return;
      }
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

function select(mesh, additive = false) {
  if (!additive) selection.length = 0;
  if (mesh) {
    const at = selection.indexOf(mesh);
    at === -1 ? selection.push(mesh) : selection.splice(at, 1);
  }
  selected = selection.length === 1 ? selection[0] : null;
  // picking something else leaves aiming mode, so the gizmo never keeps
  // meaning "polarization" for an object that was not being aimed
  if (document.body.classList.contains("aiming")) setGizmoMode("translate");
  bindProxy();
  buildInspector(selected);
  placePolarization(selected);

  const label =
    selection.length === 0
      ? "values"
      : selection.length === 1
        ? selected.name
        : `${selection.length} objects`;
  document.getElementById("sel").textContent = label;
  document.getElementById("delta").textContent = "";
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (gizmo.dragging) return;
  pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(
    pickable.filter((m) => m.visible),
    false,
  )[0];
  select(hit ? hit.object : null, event.shiftKey);
});

// ---- keyboard ------------------------------------------------------------
let snapping = false;
function setSnapping(on) {
  snapping = on;
  gizmo.setTranslationSnap(on ? 0.1 : null);
  gizmo.setRotationSnap(on ? THREE.MathUtils.degToRad(15) : null);
  gizmo.setScaleSnap(on ? 0.1 : null);
  document.getElementById("snap").textContent =
    `snap: ${on ? `0.1 ${UNIT} / 15\u00b0` : "off"}`;
}

/** Move the camera so `mesh` fills the view, keeping the current direction. */
function frame(mesh) {
  const target = mesh ? mesh.position : centre;
  mesh?.geometry.computeBoundingSphere();
  const radius = mesh
    ? mesh.geometry.boundingSphere.radius *
      Math.max(mesh.scale.x, mesh.scale.y, mesh.scale.z)
    : span / 2;
  const offset = camera.position.clone().sub(controls.target).normalize();
  camera.position.copy(target).addScaledVector(offset, radius * 3.2);
  controls.target.copy(target);
  controls.update();
}

/** Select the next pickable object, so a scene can be walked without aiming. */
function cycle(step) {
  if (!pickable.length) return;
  const at = pickable.indexOf(selected);
  select(pickable[(at + step + pickable.length) % pickable.length]);
}

// Constraining a drag to one axis is the difference between nudging along x
// and drifting in three axes at once.
function constrain(axis) {
  gizmo.showX = axis === null || axis === "x";
  gizmo.showY = axis === null || axis === "y";
  gizmo.showZ = axis === null || axis === "z";
  document.getElementById("axis").textContent =
    axis === null ? "axes: all" : `axis: ${axis}`;
}

let localSpace = false;

// The same rotate gizmo means two different things depending on what it is
// attached to, so say which, and make it a different size: aiming the
// polarization gets a smaller ring than turning the object.
function setGizmoMode(kind) {
  if (kind === "polarization") {
    gizmo.attach(polGroup);
    gizmo.setMode("rotate");
    gizmo.size = 0.62;
  } else {
    if (gizmo.object === polGroup) bindProxy();
    gizmo.setMode(kind);
    gizmo.size = 1;
  }
  const label = kind === "polarization" ? "aim polarization" : kind;
  document.getElementById("mode").textContent = `mode: ${label}`;
  document.body.classList.toggle("aiming", kind === "polarization");
}
const hidden = new Set();

function setHidden(mesh, on) {
  mesh.visible = !on;
  on ? hidden.add(mesh) : hidden.delete(mesh);
  document.getElementById("shown").textContent = hidden.size
    ? `${hidden.size} hidden`
    : "all shown";
}

/** Show only the selection, or everything again if nothing is hidden. */
function isolate() {
  if (hidden.size) {
    for (const mesh of [...hidden]) setHidden(mesh, false);
    return;
  }
  for (const mesh of pickable) {
    if (!selection.includes(mesh)) setHidden(mesh, true);
  }
}

// ---- views ----------------------------------------------------------------
const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, span * 100);
let usingOrtho = false;

function fitOrtho() {
  const aspect = window.innerWidth / window.innerHeight;
  const half = controls.target.distanceTo(camera.position) * 0.4;
  ortho.left = -half * aspect;
  ortho.right = half * aspect;
  ortho.top = half;
  ortho.bottom = -half;
  ortho.updateProjectionMatrix();
}

function toggleOrtho() {
  usingOrtho = !usingOrtho;
  const from = usingOrtho ? camera : ortho;
  const next = usingOrtho ? ortho : perspective;
  next.position.copy(from.position);
  next.up.set(0, 0, 1);
  next.lookAt(controls.target);
  if (usingOrtho) fitOrtho();
  setCamera(next);
  document.getElementById("proj").textContent = usingOrtho
    ? "orthographic"
    : "perspective";
}

/** Look down `axis` at whatever is currently framed. */
function axisView(axis) {
  const distance = camera.position.distanceTo(controls.target) || span * 2;
  camera.position
    .copy(controls.target)
    .add(axis.clone().multiplyScalar(distance));
  camera.up.set(0, 0, 1);
  controls.update();
  if (usingOrtho) fitOrtho();
}

addEventListener("keydown", (e) => {
  // otherwise typing a value in the inspector also fires the shortcuts
  if (e.target.tagName === "INPUT") return;
  if (e.key === "f") frame(selected);
  if (e.key === "Home") frame(null);
  if (e.key === "Tab") {
    e.preventDefault();
    cycle(e.shiftKey ? -1 : 1);
  }
  if (e.key === "x" || e.key === "y" || e.key === "z") constrain(e.key);
  if (e.key === "a") constrain(null);
  if (e.key === "l") {
    // magpylib stores polarization in the body frame, so a local-space gizmo
    // is the one that matches what is being edited
    localSpace = !localSpace;
    gizmo.setSpace(localSpace ? "local" : "world");
    document.getElementById("space").textContent =
      `space: ${localSpace ? "local" : "world"}`;
  }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
    e.shiftKey ? redo() : undo();
    return;
  }
  if (e.key === "Backspace") reset();
  if (e.key === "s") setSnapping(!snapping);
  if (e.key === "w") setGizmoMode("translate");
  if (e.key === "e") setGizmoMode("rotate");
  if (e.key === "r") {
    const shape = selected && SHAPES[String(selected.userData.objectId)];
    if (shape) setGizmoMode("scale");
    else
      document.getElementById("delta").textContent =
        "this object has no scale-covariant shape parameter";
  }
  if (e.key === "p") {
    if (selected && POLARIZATION[String(selected.userData.objectId)]) {
      setGizmoMode("polarization");
    } else {
      document.getElementById("delta").textContent =
        "this object has no polarization";
    }
  }
  if (e.key === "h")
    e.shiftKey ? isolate() : selection.forEach((m) => setHidden(m, true));
  if (e.key === "1") axisView(new THREE.Vector3(0, -1, 0)); // front
  if (e.key === "3") axisView(new THREE.Vector3(1, 0, 0)); // right
  if (e.key === "7") axisView(new THREE.Vector3(0, 0, 1)); // top
  if (e.key === "5") toggleOrtho();
  if (e.key === ",") {
    pivotAtOrigin = !pivotAtOrigin;
    document.getElementById("pivot").textContent =
      `pivot: ${pivotAtOrigin ? "origin" : "centre"}`;
    bindProxy();
  }
  if (e.key === "c") exportCode();
  if (e.key === "Escape") {
    document.getElementById("code").style.display = "none";
    select(null);
  }
});

// ---- during the drag: JS only, nothing sent ------------------------------
gizmo.addEventListener("objectChange", () => {
  if (gizmo.object === polGroup && selected) {
    // follow the arrow live; the committed value still comes from mouseUp
    const local = Z.clone()
      .applyQuaternion(polGroup.quaternion)
      .applyQuaternion(selected.quaternion.clone().invert());
    updateGradient(selected, local);
    return;
  }
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
  // Polarization has its own gizmo on polGroup, and never touches the proxy.
  if (gizmo.object === polGroup) {
    if (!selected) return;
    const oid = selected.userData.objectId;
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

  if (!selection.length) return;
  // detach first, so each mesh reports its own transform rather than one
  // relative to the proxy that was just dragged
  releaseProxy();

  if (selection.length > 1) {
    for (const mesh of selection) {
      const oid = mesh.userData.objectId;
      applyEdit(oid, "position", mesh.position.toArray());
      applyEdit(oid, "quaternion", mesh.quaternion.toArray());
    }
  } else {
    const oid = selected.userData.objectId,
      shape = SHAPES[String(oid)];
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
  }
  bindProxy();
});

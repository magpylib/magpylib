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
style.textContent = EDITOR_CSS; // editor.css, handed over by the backend
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
  `<span class="meta">&middot;</span><span id="space">gizmo: world axes</span>` +
  `<span class="meta">&middot;</span><span id="pivot">pivot: origin</span>` +
  `<span class="meta">&middot;</span><span id="proj">perspective</span>` +
  `<span class="meta">&middot;</span><span id="shown">all shown</span>` +
  `<span class="meta">&middot;</span><span id="frameno"></span>` +
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

// ---- tree view -------------------------------------------------------------
// Replaces the viewer's flat legend. The nesting comes from the host walking
// obj.parent, because the payload gives every trace under a Collection the
// same legendgroup and so cannot express more than one level.
const legendBox = document.getElementById("legend");
legendBox.className = "mp-panel";
legendBox.innerHTML = `<div class="mp-title">scene</div><div id="tree"></div>`;

/** Every leaf under `node`, i.e. the objects that actually have a mesh. */
function leavesOf(node) {
  if (!node.children.length) return byObjectId.has(node.id) ? [node.id] : [];
  return node.children.flatMap(leavesOf);
}

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

function flatNodes(nodes = TREE, out = []) {
  for (const node of nodes) {
    out.push(node);
    flatNodes(node.children, out);
  }
  return out;
}

function meshesOf(node) {
  return leavesOf(node)
    .map((id) => byObjectId.get(id))
    .filter(Boolean);
}

function renderTree(nodes, into) {
  for (const node of nodes) {
    const row = document.createElement("div");
    row.className = node.children.length ? "node group" : "node";
    row.dataset.id = node.id;
    row.innerHTML =
      `<span class="caret${node.children.length ? "" : " leaf"}"></span>` +
      `<span class="eye" title="show / hide"></span>` +
      (node.color
        ? `<i class="swatch" style="background:${node.color}"></i>`
        : "") +
      `<span class="label">${node.label}</span>`;
    into.appendChild(row);

    const branch = document.createElement("div");
    branch.className = "branch";
    into.appendChild(branch);
    renderTree(node.children, branch);

    row.querySelector(".caret").addEventListener("click", (event) => {
      event.stopPropagation();
      if (!node.children.length) return;
      row.classList.toggle("closed");
      branch.style.display = row.classList.contains("closed") ? "none" : "";
    });

    // the eye cascades: hiding a collection hides everything beneath it
    row.querySelector(".eye").addEventListener("click", (event) => {
      event.stopPropagation();
      const ids = leavesOf(node);
      const anyShown = ids.some((id) => byObjectId.get(id).visible);
      for (const id of ids) setHidden(byObjectId.get(id), anyShown);
      syncTree();
    });

    // the label selects, so the tree is a way in as well as a readout
    row.querySelector(".label").addEventListener("click", (event) => {
      selectNode(node, selectionMode(event));
    });
  }
}

/** Mirror selection and visibility from the scene onto the tree. */
function syncTree() {
  for (const row of legendBox.querySelectorAll(".node")) {
    const ids = leavesOf(findNode(Number(row.dataset.id)));
    const meshes = ids.map((id) => byObjectId.get(id)).filter(Boolean);
    row.classList.toggle(
      "selected",
      meshes.length > 0 && meshes.every((m) => selection.includes(m)),
    );
    row.classList.toggle(
      "off",
      meshes.length > 0 && meshes.every((m) => !m.visible),
    );
  }
}

function findNode(id, nodes = TREE) {
  for (const node of nodes) {
    if (node.id === id) return node;
    const hit = findNode(id, node.children);
    if (hit) return hit;
  }
  return { id, children: [] };
}

// A shift-click extends whatever text selection already exists, wherever it
// was anchored, and user-select does not prevent that -- so refuse the
// mousedown and clear any selection the browser has already made.
legendBox.addEventListener("mousedown", (event) => {
  event.preventDefault();
  const chosen = window.getSelection();
  if (chosen && !chosen.isCollapsed) chosen.removeAllRanges();
});

renderTree(TREE, document.getElementById("tree"));

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
    <kbd>L</kbd><span>gizmo along object / world axes</span>
    <kbd>F</kbd><span>frame selected</span>
    <kbd>\u21b9</kbd><span>next object</span>
    <kbd>C</kbd><span>export magpylib code</span>
    <kbd>space</kbd><span>play / pause paths</span>
    <kbd>\u2318Z</kbd><span>undo / <kbd>\u21e7</kbd> redo</span>
    <kbd>\u232b</kbd><span>reset</span>
    <kbd>H</kbd><span>hide (<kbd>\u21e7</kbd> isolate)</span>
    <kbd>1</kbd><span>front &middot; <kbd>3</kbd> right &middot; <kbd>7</kbd> top</span>
    <kbd>5</kbd><span>ortho / perspective</span>
    <kbd>,</kbd><span>pivot origin / centre</span>
    <kbd>\u2318</kbd><span>click: toggle one</span>
    <kbd>\u21e7</kbd><span>click: select range</span>
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
gizmo.size = 0.6;
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

// ---- selection and its proxy ----------------------------------------------
// Multi-select drives a proxy rather than a mesh, so one gizmo can move a
// group; `selectNode` further down decides what ends up in the selection.

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

/** Park the handle on `mesh`, pointing along its world polarization. */
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

// The single door every change goes through. `record` is false when replaying,
// so undo does not itself become undoable.
function applyEdit(oid, field, value, record = true) {
  const before = state.get(oid)[field];
  if (JSON.stringify(before) === JSON.stringify(value)) return;
  // State first: applyToMesh re-derives the view from it -- the polarization
  // handle is placed by reading it back -- so updating afterwards would redraw
  // from the value being replaced, and the colouring would snap back.
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

// Selection follows the file-manager split rather than treating the two
// modifiers alike: cmd/ctrl toggles one thing, shift takes the range from the
// last plain click. The range runs in tree order, which is the only order the
// scene actually has.
let anchorId = null;

function selectNode(node, mode = "replace") {
  if (!node) {
    selection.length = 0;
    anchorId = null;
  } else if (mode === "toggle") {
    for (const mesh of meshesOf(node)) {
      const at = selection.indexOf(mesh);
      at === -1 ? selection.push(mesh) : selection.splice(at, 1);
    }
    anchorId = node.id;
  } else if (mode === "range" && anchorId !== null) {
    const flat = flatNodes();
    const from = flat.findIndex((n) => n.id === anchorId);
    const to = flat.findIndex((n) => n.id === node.id);
    selection.length = 0;
    const [lo, hi] = from <= to ? [from, to] : [to, from];
    for (let i = lo; i <= hi; i++) {
      for (const mesh of meshesOf(flat[i])) {
        if (!selection.includes(mesh)) selection.push(mesh);
      }
    }
  } else {
    selection.length = 0;
    selection.push(...meshesOf(node));
    anchorId = node.id;
  }

  selected = selection.length === 1 ? selection[0] : null;
  if (document.body.classList.contains("aiming")) setGizmoMode("translate");
  bindProxy();
  buildInspector(selected);
  placePolarization(selected);
  document.getElementById("sel").textContent =
    selection.length === 0
      ? "values"
      : selection.length === 1
        ? selected.name
        : `${selection.length} objects`;
  document.getElementById("delta").textContent = "";
  syncTree();
}

/** How the modifiers on `event` should change the selection. */
function selectionMode(event) {
  if (event.shiftKey) return "range";
  if (event.metaKey || event.ctrlKey) return "toggle";
  return "replace";
}

function select(mesh, mode = "replace") {
  selectNode(mesh ? findNode(mesh.userData.objectId) : null, mode);
}

renderer.domElement.addEventListener("contextmenu", (event) =>
  event.preventDefault(),
);

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (gizmo.dragging) return;
  pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(
    pickable.filter((m) => m.visible),
    false,
  )[0];
  // Shift is the 3D-editor convention (Blender, Maya, Figma); cmd/ctrl is the
  // file-manager one. Both mean "add to the selection" to somebody, so accept
  // either rather than making people guess.
  select(hit ? hit.object : null, selectionMode(event));
});

// ---- playback -------------------------------------------------------------
// magpylib animates by re-baking every frame's vertices -- 1826 KB for 99
// frames of a sphere, measured, for what is a rigid transform of frame 0 to
// 2e-16. This backend declines supports_animation and animates from the
// model instead: the same path as position and quaternion arrays is 5.4 KB,
// and a scene graph applies a matrix per object per frame anyway.
//
// A path that changes shape rather than pose cannot be a matrix, so those are
// marked and left alone rather than played back wrongly.

const trackLength = Math.max(
  0,
  ...Object.values(TRACKS).map((t) => t.position.length),
);
let frameIndex = trackLength ? trackLength - 1 : 0; // rendered at the last step
let playing = null;

function showFrame(index) {
  frameIndex = ((index % trackLength) + trackLength) % trackLength;
  for (const [oid, track] of Object.entries(TRACKS)) {
    const mesh = byObjectId.get(Number(oid));
    if (!mesh || !track.rigid) continue;
    const at = Math.min(frameIndex, track.position.length - 1);
    mesh.position.fromArray(track.position[at]);
    mesh.quaternion.fromArray(track.quaternion[at]);
  }
  const scrub = document.getElementById("scrub");
  if (scrub) scrub.value = frameIndex;
  document.getElementById("frameno").textContent =
    `frame ${frameIndex + 1}/${trackLength}`;
  placePolarization(selected);
}

// A transport in the status bar, so playback is visible rather than a key you
// have to know. Built here rather than with the rest of the chrome because it
// only exists when something actually has a path.
if (trackLength) {
  const transport = document.createElement("span");
  transport.id = "transport";
  transport.innerHTML =
    `<button id="playbtn" title="space">&#9654;</button>` +
    `<input id="scrub" type="range" min="0" max="${trackLength - 1}" ` +
    `value="${frameIndex}">`;
  status.insertBefore(
    transport,
    document.getElementById("shown").previousSibling,
  );
  document.getElementById("playbtn").addEventListener("click", togglePlay);
  document.getElementById("scrub").addEventListener("input", (event) => {
    if (playing) togglePlay();
    showFrame(Number(event.target.value));
  });
  showFrame(frameIndex);
}

function togglePlay() {
  if (!trackLength) return;
  if (playing) {
    clearInterval(playing);
    playing = null;
  } else {
    playing = setInterval(() => showFrame(frameIndex + 1), 1000 / 20);
  }
  const button = document.getElementById("playbtn");
  if (button) button.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
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
//: Gizmo sizes. The polarization ring stays proportionally smaller than the
//: object handles, since that difference is one of the mode signals.
const GIZMO_SIZE = { object: 0.6, polarization: 0.38 };

function setGizmoMode(kind) {
  if (kind === "polarization") {
    gizmo.attach(polGroup);
    gizmo.setMode("rotate");
    gizmo.size = GIZMO_SIZE.polarization;
  } else {
    if (gizmo.object === polGroup) bindProxy();
    gizmo.setMode(kind);
    gizmo.size = GIZMO_SIZE.object;
  }
  const label = kind === "polarization" ? "aim polarization" : kind;
  document.getElementById("mode").textContent = `mode: ${label}`;
  document.body.classList.toggle("aiming", kind === "polarization");
}
const hidden = new Set();

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
      `gizmo: ${localSpace ? "object axes" : "world axes"}`;
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
  if (e.key === " ") {
    e.preventDefault();
    togglePlay();
  }
  if (e.key === "Escape") {
    document.getElementById("code").style.display = "none";
    select(null);
  }
});

// ---- during the drag: JS only, nothing sent ------------------------------
gizmo.addEventListener("objectChange", () => {
  if (gizmo.object === polGroup && selected) {
    // follow the handle live; the committed value still comes from mouseUp
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
    // the handle expresses direction only, so amplitude is carried through
    const magnitude = new THREE.Vector3()
      .fromArray(state.get(oid)[pol.attr])
      .length();
    // the handle points in world space; magpylib stores the body frame
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
      // A rotation about anything other than the object's own origin moves it
      // as well as turning it -- which is the case whenever the pivot is the
      // selection centre. Sending only the quaternion would drop that, and the
      // view and python would quietly disagree.
      applyEdit(oid, "quaternion", selected.quaternion.toArray());
      applyEdit(oid, "position", selected.position.toArray());
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

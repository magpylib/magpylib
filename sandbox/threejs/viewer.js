// The page body for the prototype three.js backend: builds the scene from
// the payload magpylib hands over. Loaded verbatim by magpylib_threejs.py.
//
// A real .js file rather than a Python string, so it gets highlighting and
// `node --check`, and -- the reason it was moved -- so its braces are not
// doubled for str.format. That doubling hid a real bug once already.
//
// The payload placeholder below is filled in before the page is served.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
// plain WebGL lines are always 1px wide; Line2 renders them as camera-facing
// quads, so line_width is honoured. worldUnits:false makes the width a pixel
// count -- the same screen-space mechanism zoom-invariant sizing would need.
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

const DATA = __DATA__;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// frame the camera on the ranges Magpylib resolved for the panel
const [rx, ry, rz] = DATA.ranges;
const centre = new THREE.Vector3(
  (rx[0] + rx[1]) / 2,
  (ry[0] + ry[1]) / 2,
  (rz[0] + rz[1]) / 2,
);
const span = Math.max(rx[1] - rx[0], ry[1] - ry[0], rz[1] - rz[0]) || 1;

let camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  span / 1000,
  span * 100,
);
camera.up.set(0, 0, 1); // Magpylib is z-up
camera.position
  .copy(centre)
  .add(new THREE.Vector3(span * 1.4, -span * 1.6, span * 1.1));

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(centre);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(1, -1, 1).multiplyScalar(span);
scene.add(key);

const byObjectId = new Map();
for (const item of DATA.meshes) {
  const geometry = new THREE.BufferGeometry();
  const options = {
    transparent: item.opacity < 1,
    opacity: item.opacity,
    side: THREE.DoubleSide,
  };

  if (item.facecolor) {
    // per-triangle colours need one vertex per triangle corner, so the
    // geometry is expanded rather than indexed
    const pos = [],
      col = [],
      c = new THREE.Color();
    for (let t = 0; t < item.facecolor.length; t++) {
      c.set(item.facecolor[t]); // parses '#rrggbb' and 'black'
      for (let corner = 0; corner < 3; corner++) {
        const v = item.index[t * 3 + corner];
        pos.push(
          item.position[v * 3],
          item.position[v * 3 + 1],
          item.position[v * 3 + 2],
        );
        col.push(c.r, c.g, c.b);
      }
    }
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    options.vertexColors = true;
  } else {
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(item.position, 3),
    );
    geometry.setIndex(item.index);
    if (item.lut) {
      // interpolate intensity across the face and look the colour up per
      // fragment, so a piecewise colorscale keeps its plateaus
      geometry.setAttribute("uv", new THREE.Float32BufferAttribute(item.uv, 2));
      const texture = new THREE.DataTexture(
        new Uint8Array(item.lut),
        item.lut.length / 4,
        1,
        THREE.RGBAFormat,
      );
      texture.minFilter = texture.magFilter = THREE.LinearFilter;
      texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.needsUpdate = true;
      options.map = texture;
    } else {
      options.color = new THREE.Color(item.color || "#2e91e5");
    }
  }
  geometry.computeVertexNormals();

  // Magpylib bakes position and orientation into the vertices and sends no
  // transform, so every mesh would sit at the origin with an identity matrix
  // -- and a gizmo attached to it would appear at the world origin rather
  // than on the object. Re-centre the geometry and move the offset onto the
  // mesh, which is visually identical but gives each object a real transform.
  // The anchor is the bounding-box centre, not Magpylib's origin for the
  // object; for a Cuboid they coincide, for a Sensor they do not.
  // The host may supply the object's real origin, looked up from object_id;
  // the bounding-box centre is only the fallback.
  const anchor = new THREE.Vector3();
  const given = DATA.anchors[String(item.object_id)];
  if (given) {
    anchor.fromArray(given);
  } else {
    geometry.computeBoundingBox();
    geometry.boundingBox.getCenter(anchor);
  }
  geometry.translate(-anchor.x, -anchor.y, -anchor.z);

  const material = new THREE.MeshLambertMaterial(options);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(anchor);
  mesh.name = item.name;
  mesh.userData.objectId = item.object_id;
  // kept so an editor can recolour without re-reading the payload
  mesh.userData.lut = item.lut;
  scene.add(mesh);
  byObjectId.set(item.object_id, mesh);
}
const lineMaterials = [];
for (const item of DATA.scatters) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(item.position, 3),
  );

  if (item.lines) {
    const lineGeometry = new LineGeometry();
    lineGeometry.setPositions(item.position);
    // item.line_dash still has no primitive; LineMaterial can dash, but only
    // with a uniform pattern, not magpylib's named styles.
    const material = new LineMaterial({
      color: new THREE.Color(item.line_color),
      linewidth: item.line_width, // pixels, because worldUnits is false
      worldUnits: false,
      transparent: item.opacity < 1,
      opacity: item.opacity,
    });
    material.resolution.set(window.innerWidth, window.innerHeight);
    lineMaterials.push(material);

    const line = new Line2(lineGeometry, material);
    line.computeLineDistances();
    line.name = item.name;
    line.userData.objectId = item.object_id;
    scene.add(line);
  }
  if (item.markers) {
    // item.marker_symbol has no primitive either: PointsMaterial draws
    // squares, and symbols would need a texture atlas.
    // sizeAttenuation:false makes size a pixel count rather than a world
    // length, so markers keep their size under zoom -- as in Plotly. Point
    // size is in device pixels, hence the devicePixelRatio.
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        color: new THREE.Color(item.marker_color),
        size: item.marker_size * window.devicePixelRatio,
        sizeAttenuation: false,
        transparent: item.opacity < 1,
        opacity: item.opacity,
      }),
    );
    points.name = item.name;
    points.userData.objectId = item.object_id;
    scene.add(points);
  }
}

window.magpyObjects = byObjectId; // so a host page can address one object

// A bounding box over Panel.ranges, like the box the built-in backends draw.
// An AxesHelper would be misleading here: its red/green/blue axes are exactly
// what a Sensor looks like, so the frame reads as an object in the scene.
{
  const box = new THREE.Box3(
    new THREE.Vector3(rx[0], ry[0], rz[0]),
    new THREE.Vector3(rx[1], ry[1], rz[1]),
  );
  scene.add(new THREE.Box3Helper(box, new THREE.Color(0xcccccc)));
}

const legend = document.getElementById("legend");
const entries = new Map();
for (const m of DATA.meshes) {
  if (!entries.has(m.name)) entries.set(m.name, m.legend_color);
}
for (const s of DATA.scatters) {
  if (!entries.has(s.name)) entries.set(s.name, s.line_color);
}
legend.innerHTML = [...entries]
  .map(([name, css]) => `<div><i style="background: ${css}"></i>${name}</div>`)
  .join("");

addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  // pixel-width lines need the viewport size to convert into clip space
  for (const material of lineMaterials) {
    material.resolution.set(window.innerWidth, window.innerHeight);
  }
});

// Anything a specialised backend wants to add: picking, gizmos, HUD. Injected
// after formatting so it may contain braces freely.
__EXTRA_JS__;

// `camera` is a binding rather than a constant, and the loop reads it each
// frame, so an editor layer can swap in an orthographic one.
function setCamera(next) {
  camera = next;
  controls.object = next;
  controls.update();
}
renderer.setAnimationLoop(() => renderer.render(scene, camera));

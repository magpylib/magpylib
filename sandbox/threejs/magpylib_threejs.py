"""A minimal three.js display backend for Magpylib -- prototype, not shipped.

Deliberately narrow: `mesh3d` traces only (which covers magnets), no
animation, no subplots. The point is to exercise
`magpylib.graphics.backend` from the outside, the way a real third-party
backend would, and find out where the contract is thin. Findings are
written up in README.md next to this file.

`show()` returns a self-contained HTML page as a string.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import magpylib as magpy

THREE_VERSION = "0.160.0"

#: Magpylib's `line_width` and `marker_size` are nominal: every backend scales
#: them into its own library's units, and nothing in the contract says what a
#: width of 2 should look like -- `backend_plotly` carries its own
#: `SIZE_FACTORS_TO_PLOTLY`, and the others calibrate elsewhere. These are
#: Plotly's numbers, which transfer directly because `Line2` and
#: `PointsMaterial(sizeAttenuation=False)` also measure in pixels.
SIZE_FACTORS = {"line_width": 2.2, "marker_size": 0.7}


def _hex_to_rgb(color):
    """'#rrggbb' -> (r, g, b) floats in 0..1."""
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


LUT_SIZE = 256


def _colorscale_lut(colorscale):
    """Flatten a Magpylib colorscale into a `LUT_SIZE` RGB lookup table.

    The colorscale arrives as ``((stop, '#rrggbb'), ...)``, the Plotly spelling
    Magpylib's trace dialect inherited, and it is *piecewise*: the default
    tricolor scheme holds green to 0.16, grey from 0.26 to 0.74, then red.

    Sampling it per vertex does not work. A Cuboid has eight vertices whose
    intensities are all exactly 0 or 1, so nothing lands on the grey plateau
    and the GPU blends corner colours into a straight green-to-red ramp. What
    has to be interpolated across the face is the *intensity*, with the
    colorscale applied per fragment -- hence a lookup texture, indexed by an
    intensity-valued UV. That is what Plotly's shader does too.
    """
    stops = np.array([s for s, _ in colorscale], dtype=float)
    colors = np.array([_hex_to_rgb(c) for _, c in colorscale], dtype=float)
    samples = np.linspace(0.0, 1.0, LUT_SIZE)
    lut = np.stack(
        [np.interp(samples, stops, colors[:, channel]) for channel in range(3)]
        # RGBA: three.js dropped RGBFormat in r137, so RGBAFormat it is
        + [np.ones_like(samples)],
        axis=1,
    )
    return np.round(lut * 255).astype(int).ravel().tolist()


def _mesh_to_payload(trace):
    """Convert one `mesh3d` trace into what the JS side needs."""
    position = np.stack(
        [np.asarray(trace[axis], dtype=float) for axis in "xyz"], axis=1
    )
    index = np.stack(
        [np.asarray(trace[axis], dtype=int) for axis in ("i", "j", "k")], axis=1
    )

    intensity = trace.get("intensity")
    colorscale = trace.get("colorscale")
    graded = intensity is not None and colorscale is not None
    # per-triangle colours, e.g. a Sensor: body in the object's colour, pixels
    # black, arrow heads red/green/blue. Mixes CSS names with hex, so the
    # strings go over as-is for THREE.Color to parse.
    facecolor = trace.get("facecolor")
    facecolor = None if facecolor is None else [str(c) for c in facecolor]

    uv = None
    if graded:
        intensity = np.clip(np.asarray(intensity, dtype=float), 0, 1)
        uv = np.stack([intensity, np.full(len(position), 0.5)], axis=1)

    return {
        "name": trace.get("name") or "",
        "object_id": trace.get("object_id"),
        "opacity": float(trace.get("opacity", 1) or 1),
        "position": position.ravel().tolist(),
        "index": index.ravel().tolist(),
        "color": trace.get("color"),
        "uv": None if uv is None else uv.ravel().tolist(),
        "lut": _colorscale_lut(colorscale) if graded else None,
        "facecolor": facecolor,
        # legend swatch: the flat colour, else whichever face colour dominates
        "legend_color": (
            trace.get("color")
            or (Counter(facecolor).most_common(1)[0][0] if facecolor else "#2e91e5")
        ),
    }


def _scatter_to_payload(trace):
    """Convert one `scatter3d` trace into what the JS side needs.

    `mode` is a combination, not an enum -- "markers+lines" and
    "markers+text+lines" both occur -- so it is split into tokens and each is
    handled on its own.
    """
    position = np.stack(
        [np.asarray(trace[axis], dtype=float) for axis in "xyz"], axis=1
    )
    modes = set(str(trace.get("mode") or "lines").split("+"))
    return {
        "name": trace.get("name") or "",
        "object_id": trace.get("object_id"),
        "opacity": float(trace.get("opacity", 1) or 1),
        "position": position.ravel().tolist(),
        "lines": "lines" in modes,
        "markers": "markers" in modes,
        # see README: WebGL ignores line width, and neither dash patterns nor
        # marker symbols have a primitive here. Carried so the gap is visible
        # in the payload rather than silently dropped in Python.
        "line_color": trace.get("line_color") or "#2e91e5",
        "line_width": float(trace.get("line_width") or 1) * SIZE_FACTORS["line_width"],
        "line_dash": trace.get("line_dash") or "solid",
        "marker_color": trace.get("marker_color") or "#2e91e5",
        "marker_size": (
            float(trace.get("marker_size") or 3) * SIZE_FACTORS["marker_size"]
        ),
        "marker_symbol": trace.get("marker_symbol") or "o",
    }


_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #ffffff; }}
  #legend {{
    position: absolute; top: 8px; left: 8px; font: 12px sans-serif;
    background: rgba(255,255,255,.85); padding: 6px 8px; border-radius: 4px;
  }}
  #legend div {{ display: flex; align-items: center; gap: 6px; }}
  #legend i {{ width: 10px; height: 10px; border-radius: 2px; }}
</style>
<div id="legend"></div>
<script type="importmap">
{{"imports": {{
  "three": "https://unpkg.com/three@{version}/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@{version}/examples/jsm/"
}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
// plain WebGL lines are always 1px wide; Line2 renders them as camera-facing
// quads, so line_width is honoured. worldUnits:false makes the width a pixel
// count -- the same screen-space mechanism zoom-invariant sizing would need.
import {{ Line2 }} from 'three/addons/lines/Line2.js';
import {{ LineGeometry }} from 'three/addons/lines/LineGeometry.js';
import {{ LineMaterial }} from 'three/addons/lines/LineMaterial.js';

const DATA = {data};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// frame the camera on the ranges Magpylib resolved for the panel
const [rx, ry, rz] = DATA.ranges;
const centre = new THREE.Vector3(
  (rx[0] + rx[1]) / 2, (ry[0] + ry[1]) / 2, (rz[0] + rz[1]) / 2);
const span = Math.max(rx[1] - rx[0], ry[1] - ry[0], rz[1] - rz[0]) || 1;

const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, span / 1000, span * 100);
camera.up.set(0, 0, 1);                       // Magpylib is z-up
camera.position.copy(centre).add(
  new THREE.Vector3(span * 1.4, -span * 1.6, span * 1.1));

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(centre);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(1, -1, 1).multiplyScalar(span);
scene.add(key);

const byObjectId = new Map();
for (const item of DATA.meshes) {{
  const geometry = new THREE.BufferGeometry();
  const options = {{
    transparent: item.opacity < 1,
    opacity: item.opacity,
    side: THREE.DoubleSide,
  }};

  if (item.facecolor) {{
    // per-triangle colours need one vertex per triangle corner, so the
    // geometry is expanded rather than indexed
    const pos = [], col = [], c = new THREE.Color();
    for (let t = 0; t < item.facecolor.length; t++) {{
      c.set(item.facecolor[t]);                 // parses '#rrggbb' and 'black'
      for (let corner = 0; corner < 3; corner++) {{
        const v = item.index[t * 3 + corner];
        pos.push(item.position[v * 3], item.position[v * 3 + 1],
                 item.position[v * 3 + 2]);
        col.push(c.r, c.g, c.b);
      }}
    }}
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    options.vertexColors = true;
  }} else {{
    geometry.setAttribute('position',
      new THREE.Float32BufferAttribute(item.position, 3));
    geometry.setIndex(item.index);
    if (item.lut) {{
      // interpolate intensity across the face and look the colour up per
      // fragment, so a piecewise colorscale keeps its plateaus
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute(item.uv, 2));
      const texture = new THREE.DataTexture(
        new Uint8Array(item.lut), item.lut.length / 4, 1, THREE.RGBAFormat);
      texture.minFilter = texture.magFilter = THREE.LinearFilter;
      texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.needsUpdate = true;
      options.map = texture;
    }} else {{
      options.color = new THREE.Color(item.color || '#2e91e5');
    }}
  }}
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
  if (given) {{
    anchor.fromArray(given);
  }} else {{
    geometry.computeBoundingBox();
    geometry.boundingBox.getCenter(anchor);
  }}
  geometry.translate(-anchor.x, -anchor.y, -anchor.z);

  const material = new THREE.MeshLambertMaterial(options);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(anchor);
  mesh.name = item.name;
  mesh.userData.objectId = item.object_id;
  scene.add(mesh);
  byObjectId.set(item.object_id, mesh);
}}
const lineMaterials = [];
for (const item of DATA.scatters) {{
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position',
    new THREE.Float32BufferAttribute(item.position, 3));

  if (item.lines) {{
    const lineGeometry = new LineGeometry();
    lineGeometry.setPositions(item.position);
    // item.line_dash still has no primitive; LineMaterial can dash, but only
    // with a uniform pattern, not magpylib's named styles.
    const material = new LineMaterial({{
      color: new THREE.Color(item.line_color),
      linewidth: item.line_width,     // pixels, because worldUnits is false
      worldUnits: false,
      transparent: item.opacity < 1,
      opacity: item.opacity,
    }});
    material.resolution.set(window.innerWidth, window.innerHeight);
    lineMaterials.push(material);

    const line = new Line2(lineGeometry, material);
    line.computeLineDistances();
    line.name = item.name;
    line.userData.objectId = item.object_id;
    scene.add(line);
  }}
  if (item.markers) {{
    // item.marker_symbol has no primitive either: PointsMaterial draws
    // squares, and symbols would need a texture atlas.
    // sizeAttenuation:false makes size a pixel count rather than a world
    // length, so markers keep their size under zoom -- as in Plotly. Point
    // size is in device pixels, hence the devicePixelRatio.
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({{
      color: new THREE.Color(item.marker_color),
      size: item.marker_size * window.devicePixelRatio,
      sizeAttenuation: false,
      transparent: item.opacity < 1,
      opacity: item.opacity,
    }}));
    points.name = item.name;
    points.userData.objectId = item.object_id;
    scene.add(points);
  }}
}}

window.magpyObjects = byObjectId;   // so a host page can address one object

// A bounding box over Panel.ranges, like the box the built-in backends draw.
// An AxesHelper would be misleading here: its red/green/blue axes are exactly
// what a Sensor looks like, so the frame reads as an object in the scene.
{{
  const box = new THREE.Box3(
    new THREE.Vector3(rx[0], ry[0], rz[0]),
    new THREE.Vector3(rx[1], ry[1], rz[1]));
  scene.add(new THREE.Box3Helper(box, new THREE.Color(0xcccccc)));
}}

const legend = document.getElementById('legend');
const entries = new Map();
for (const m of DATA.meshes) {{
  if (!entries.has(m.name)) entries.set(m.name, m.legend_color);
}}
for (const s of DATA.scatters) {{
  if (!entries.has(s.name)) entries.set(s.name, s.line_color);
}}
legend.innerHTML = [...entries].map(([name, css]) =>
  `<div><i style="background: ${{css}}"></i>${{name}}</div>`).join('');

addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  // pixel-width lines need the viewport size to convert into clip space
  for (const material of lineMaterials) {{
    material.resolution.set(window.innerWidth, window.innerHeight);
  }}
}});

// Anything a specialised backend wants to add: picking, gizmos, HUD. Injected
// after formatting so it may contain braces freely.
__EXTRA_JS__

renderer.setAnimationLoop(() => renderer.render(scene, camera));
</script>
"""


def render_page(scene, extra_js="", anchors=None):
    """Build the HTML page for `scene`, optionally with extra JS appended.

    `anchors` maps ``object_id`` to the object's true origin, in the same
    length unit as the geometry. Magpylib bakes position and orientation into
    the vertices and sends no transform, but `object_id` is documented as valid
    for "an interactive viewer holding the same objects" -- so a host that owns
    them looks the transform up itself and passes it here. Without it the mesh
    falls back to its bounding-box centre, which is wrong for any object whose
    origin is not its centroid.
    """
    panel = scene.panel(1, 1)
    traces = [trace for frame in scene.frames for trace in frame.traces]
    data = {
        "meshes": [_mesh_to_payload(t) for t in traces if t["type"] == "mesh3d"],
        "scatters": [
            _scatter_to_payload(t) for t in traces if t["type"] == "scatter3d"
        ],
        "ranges": panel.ranges.tolist(),
        "labels": panel.labels,
        "anchors": {str(k): list(v) for k, v in (anchors or {}).items()},
    }
    return (
        _TEMPLATE.format(version=THREE_VERSION, data=json.dumps(data))
        .replace("__TITLE__", scene.title or "Magpylib")
        .replace("__EXTRA_JS__", extra_js)
    )


def show_threejs(scene):
    """Render a Magpylib `Scene` to a self-contained three.js page."""
    html = render_page(scene)

    if scene.canvas is not None:  # see README: canvas has no meaning here
        msg = "the threejs backend cannot draw onto an existing canvas"
        raise NotImplementedError(msg)

    # `return_fig` is advisory: magpylib passes the backend's return value
    # straight back to the caller either way, so honouring it is on us.
    if scene.return_fig:
        return html
    return _display(html)


def _display(html):
    """Show the page: inline in a notebook, otherwise in a browser tab."""
    try:
        from IPython.display import HTML, display  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if get_ipython_shell() is not None:
            display(HTML(html))
            return

    import tempfile  # noqa: PLC0415
    import webbrowser  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(html)
    webbrowser.open(f"file://{handle.name}")


def get_ipython_shell():
    """The active IPython shell, or None outside one."""
    try:
        from IPython import get_ipython  # noqa: PLC0415
    except ImportError:
        return None
    return get_ipython()


def register():
    """Register the backend under the name 'threejs'."""
    return magpy.register_backend(
        "threejs",
        show_threejs,
        # three.js does vertex colours natively, so take the gradient unsliced
        supports_colorgradient=True,
        # every object keeps its own mesh, so it can be addressed and moved
        merge_traces=False,
        # prototype scope
        supports_animation=False,
        supports_subplots=False,
        supports_animation_output=False,
        supports_native_traces=False,
        handles_traces=frozenset({"mesh3d", "scatter3d"}),
        accepts_options=frozenset(),
    )

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

import numpy as np

import magpylib as magpy

THREE_VERSION = "0.160.0"


def _hex_to_rgb(color):
    """'#rrggbb' -> (r, g, b) floats in 0..1."""
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _sample_colorscale(colorscale, values):
    """Map `values` (0..1) through a Magpylib colorscale to RGB triples.

    The colorscale arrives as ``((stop, '#rrggbb'), ...)``, the Plotly
    spelling Magpylib's trace dialect inherited. Interpolating here rather
    than in the shader keeps the JS side dumb.
    """
    stops = np.array([s for s, _ in colorscale], dtype=float)
    colors = np.array([_hex_to_rgb(c) for _, c in colorscale], dtype=float)
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return np.stack(
        [np.interp(values, stops, colors[:, channel]) for channel in range(3)],
        axis=1,
    )


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
    if intensity is not None and colorscale is not None:
        colors = _sample_colorscale(colorscale, intensity)
    else:
        flat = _hex_to_rgb(trace.get("color") or "#2e91e5")
        colors = np.tile(flat, (len(position), 1))

    return {
        "name": trace.get("name") or "",
        "object_id": trace.get("object_id"),
        "opacity": float(trace.get("opacity", 1) or 1),
        "position": position.ravel().tolist(),
        "index": index.ravel().tolist(),
        "color": colors.ravel().tolist(),
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
        "line_width": float(trace.get("line_width") or 1),
        "line_dash": trace.get("line_dash") or "solid",
        "marker_color": trace.get("marker_color") or "#2e91e5",
        "marker_size": float(trace.get("marker_size") or 3),
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
  geometry.setAttribute('position',
    new THREE.Float32BufferAttribute(item.position, 3));
  geometry.setAttribute('color',
    new THREE.Float32BufferAttribute(item.color, 3));
  geometry.setIndex(item.index);
  geometry.computeVertexNormals();

  const material = new THREE.MeshLambertMaterial({{
    vertexColors: true,
    transparent: item.opacity < 1,
    opacity: item.opacity,
    side: THREE.DoubleSide,
  }});
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = item.name;
  mesh.userData.objectId = item.object_id;
  scene.add(mesh);
  byObjectId.set(item.object_id, mesh);
}}
for (const item of DATA.scatters) {{
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position',
    new THREE.Float32BufferAttribute(item.position, 3));

  if (item.lines) {{
    // NOTE: WebGL ignores linewidth on LineBasicMaterial -- every line is one
    // pixel regardless of item.line_width. Real widths need Line2 +
    // LineMaterial from the addons. Likewise item.line_dash has no primitive.
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({{
      color: new THREE.Color(item.line_color),
      transparent: item.opacity < 1,
      opacity: item.opacity,
    }}));
    line.name = item.name;
    line.userData.objectId = item.object_id;
    scene.add(line);
  }}
  if (item.markers) {{
    // item.marker_symbol has no primitive either: PointsMaterial draws
    // squares, and symbols would need a texture atlas.
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({{
      color: new THREE.Color(item.marker_color),
      size: item.marker_size * span / 200,
      transparent: item.opacity < 1,
      opacity: item.opacity,
    }}));
    points.name = item.name;
    points.userData.objectId = item.object_id;
    scene.add(points);
  }}
}}

window.magpyObjects = byObjectId;   // so a host page can address one object

scene.add(new THREE.AxesHelper(span / 2));

const legend = document.getElementById('legend');
const entries = new Map();
for (const m of DATA.meshes) {{
  const [r, g, b] = m.color.slice(0, 3).map(v => Math.round(v * 255));
  entries.set(m.name, `rgb(${{r}},${{g}},${{b}})`);
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
}});

renderer.setAnimationLoop(() => renderer.render(scene, camera));
</script>
"""


def show_threejs(scene):
    """Render a Magpylib `Scene` to a self-contained three.js page."""
    panel = scene.panel(1, 1)
    traces = [trace for frame in scene.frames for trace in frame.traces]
    data = {
        "meshes": [_mesh_to_payload(t) for t in traces if t["type"] == "mesh3d"],
        "scatters": [
            _scatter_to_payload(t) for t in traces if t["type"] == "scatter3d"
        ],
        "ranges": panel.ranges.tolist(),
        "labels": panel.labels,
    }
    html = _TEMPLATE.format(
        version=THREE_VERSION,
        data=json.dumps(data),
    ).replace("__TITLE__", scene.title or "Magpylib")

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

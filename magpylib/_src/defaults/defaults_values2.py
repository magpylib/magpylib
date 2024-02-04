"""Package level config defaults"""
import param

from magpylib._src.defaults.defaults_utility import ALLOWED_LINESTYLES
from magpylib._src.defaults.defaults_utility import ALLOWED_SYMBOLS
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS

ALLOWED_SIZEMODES = ("scaled", "absolute")

DEFAULTS = {
    "display.autosizefactor": {
        "$type": "Number",
        "default": 10,
        "bounds": (0, None),
        "inclusive_bounds": (False, True),
        "softbounds": (5, 15),
        "doc": """
        Defines at which scale objects like sensors and dipoles are displayed.
        -> object_size = canvas_size / AUTOSIZE_FACTOR""",
    },
    "display.animation.fps": {
        "$type": "Integer",
        "default": 20,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": """Target number of frames to be displayed per second.""",
    },
    "display.animation.maxfps": {
        "$type": "Integer",
        "default": 30,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": """Maximum number of frames to be displayed per second before downsampling kicks in.""",
    },
    "display.animation.maxframes": {
        "$type": "Integer",
        "default": 200,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": """Maximum total number of frames to be displayed before downsampling kicks in.""",
    },
    "display.animation.time": {
        "$type": "Number",
        "default": 5,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": """Default animation time.""",
    },
    "display.animation.slider": {
        "$type": "Boolean",
        "default": True,
        "doc": """If True, an interactive slider will be displayed and stay in sync with the animation, Ïwill be hidden otherwise.""",
    },
    "display.animation.output ": {
        "$type": "String",
        "doc": """Animation output type""",
        "regex": r"^(mp4|gif|(.*\.(mp4|gif))?)$",  # either `mp4` or `gif` or ending with `.mp4` or `.gif`"
    },
    "display.backend": {
        "$type": "Selector",
        "default": "auto",
        "objects": ["auto", *SUPPORTED_PLOTTING_BACKENDS],
        "doc": """
    Plotting backend to be used by default, if not explicitly set in the `display`
    function (e.g. 'matplotlib', 'plotly').
    Supported backends are defined in magpylib.SUPPORTED_PLOTTING_BACKENDS""",
    },
    "display.colorsequence": {
        "$type": "List",
        # "item_type": "Color",
        "default": [
            "#2E91E5",
            "#E15F99",
            "#1CA71C",
            "#FB0D0D",
            "#DA16FF",
            "#B68100",
            "#750D86",
            "#EB663B",
            "#511CFB",
            "#00A08B",
            "#FB00D1",
            "#FC0080",
            "#B2828D",
            "#6C7C32",
            "#778AAE",
            "#862A16",
            "#A777F1",
            "#620042",
            "#1616A7",
            "#DA60CA",
            "#6C4516",
            "#0D2A63",
            "#AF0038",
            "#222A2A",
        ],
        "doc": """
        An iterable of color values used to cycle trough for every object displayed.
        A color may be specified by
            - a hex string (e.g. '#ff0000')
            - an rgb/rgba string (e.g. 'rgb(255,0,0)')
            - an hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - an hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - a named CSS color""",
    },
    "display.style.base.path.line.width": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Path line width.",
    },
    "display.style.base.path.line.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Path line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.base.path.line.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Path line color.",
    },
    "display.style.base.path.marker.size": {
        "$type": None,
        "default": 3,
        "doc": "Path marker size.",
    },
    "display.style.base.path.marker.symbol": {
        "$type": "Selector",
        "default": "o",
        "objects": ALLOWED_SYMBOLS,
        "doc": f"Path marker symbol. Can be one of: {ALLOWED_SYMBOLS}.",
    },
    "display.style.base.path.marker.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Path marker color.",
    },
    "display.style.base.path.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide path.",
    },
    "display.style.base.path.frames.indices ": {
        "$type": "List",
        "default": [],
        "item_type": int,
        "doc": """Array_like shape (n,) of integers: describes certain path indices.""",
    },
    "display.style.base.path.frames.step ": {
        "$type": "Integer",
        "default": 1,
        "bounds": (1, None),
        "softbounds": (0, 10),
        "doc": """Displays the object(s) at every i'th path position.""",
    },
    "display.style.base.path.frames.mode ": {
        "$type": "Selector",
        "default": "indices",
        "objects": ["indices", "step"],
        "doc": """
        The object path frames mode.
        - step: integer i: displays the object(s) at every i'th path position.
        - indices: array_like shape (n,) of integers: describes certain path indices.""",
    },
    "display.style.base.path.numbering": {
        "$type": "Boolean",
        "default": False,
        "doc": "Show/hide numbering on path positions. Only applies if show=True.",
    },
    "display.style.base.description.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide object description in legend (shown in parentheses).",
    },
    "display.style.base.description.text": {
        "$type": "String",
        "default": "",
        "doc": "Object description text.",
    },
    "display.style.base.legend.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide legend.",
    },
    "display.style.base.legend.text": {
        "$type": "String",
        "default": "",
        "doc": "Custom legend text. Overrides complete legend.",
    },
    "display.style.base.opacity": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 1),
        "inclusive_bounds": (True, True),
        "softbounds": (0, 1),
        "doc": "Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.",
    },
    "display.style.base.model3d.showdefault": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide default 3D-model.",
    },
    "display.style.base.model3d.data": {
        "$type": "List",
        "default": [],
        "doc": """
        A trace or list of traces where each is an instance of `Trace3d` or dictionary of
        equivalent key/value pairs. Defines properties for an additional user-defined model3d
        object which is positioned relatively to the main object to be displayed and moved
        automatically with it. This feature also allows the user to replace the original 3d
        representation of the object.""",
    },
    "display.style.base.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Object explicit color",
    },
    "display.style.magnet.magnetization.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide magnetization indication (arrow and/or color).",
    },
    "display.style.magnet.magnetization.arrow.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide magnetization arrow.",
    },
    "display.style.magnet.magnetization.arrow.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Magnetization arrow size.",
    },
    "display.style.magnet.magnetization.arrow.sizemode": {
        "$type": "Selector",
        "default": "scaled",
        "objects": ALLOWED_SIZEMODES,
        "doc": f"The way the object size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`",
    },
    "display.style.magnet.magnetization.arrow.offset": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 1),
        "inclusive_bounds": (True, True),
        "softbounds": (0, 1),
        "doc": """
        Defines the arrow offset. `offset=0` results in the arrow head to be coincident with
        the start of the line, and `offset=1` with the end.""",
    },
    "display.style.magnet.magnetization.arrow.width": {
        "$type": None,
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Line width",
    },
    "display.style.magnet.magnetization.arrow.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Arrow line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.magnet.magnetization.arrow.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Magnetization explicit arrow color. Takes the object color by default.",
    },
    "display.style.magnet.magnetization.color.north": {
        "$type": "Color",
        "default": "#E71111",
        "doc": "The color of the magnetic north pole.",
    },
    "display.style.magnet.magnetization.color.middle": {
        "$type": "Color",
        "default": "#DDDDDD",
        "doc": "The color between the magnetic poles.",
    },
    "display.style.magnet.magnetization.color.south": {
        "$type": "Color",
        "default": "#00B050",
        "doc": "The color of the magnetic south pole.",
    },
    "display.style.magnet.magnetization.color.transition": {
        "$type": "Number",
        "default": 0.2,
        "bounds": (0, 1),
        "inclusive_bounds": (True, True),
        "softbounds": (0, 1),
        "doc": """Sets the transition smoothness between poles colors. Must be between 0 and 1.
        - `transition=0`: discrete transition
        - `transition=1`: smoothest transition
        """,
    },
    "display.style.magnet.magnetization.color.mode": {
        "$type": "Selector",
        "default": "tricolor",
        "objects": ("tricolor", "bicolor", "tricycle"),
        "doc": """
        Sets the coloring mode for the magnetization.
        - `'bicolor'`: only north and south poles are shown, middle color is hidden.
        - `'tricolor'`: both pole colors and middle color are shown.
        - `'tricycle'`: both pole colors are shown and middle color is replaced by a color cycling
        through the color sequence.""",
    },
    "display.style.magnet.magnetization.mode": {
        "$type": "Selector",
        "default": "auto",
        "objects": ("auto", "arrow", "color", "arrow+color", "color+arrow"),
        "doc": """
        One of {"auto", "arrow", "color", "arrow+color"}, default="auto"
        Magnetization can be displayed via arrows, color or both. By default `mode='auto'` means
        that the chosen backend determines which mode is applied by its capability. If the backend
        can display both and `auto` is chosen, the priority is given to `color`.""",
    },
    # TODO next ->
    "display.style.current.arrow.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "",
    },
    "display.style.current.arrow.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.current.arrow.sizemode": {
        "$type": None,
        "default": "scaled",
        "doc": "",
    },
    "display.style.current.arrow.offset": {
        "$type": None,
        "default": 0.5,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.current.arrow.width": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.current.arrow.style": {"$type": None, "default": "solid", "doc": ""},
    "display.style.current.arrow.color": {"$type": None, "default": None, "doc": ""},
    "display.style.current.line.show": {"$type": "Boolean", "default": True, "doc": ""},
    "display.style.current.line.width": {
        "$type": None,
        "default": 2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.current.line.style": {"$type": None, "default": "solid", "doc": ""},
    "display.style.current.line.color": {"$type": None, "default": None, "doc": ""},
    "display.style.sensor.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.sensor.sizemode": {"$type": None, "default": "scaled", "doc": ""},
    "display.style.sensor.pixel.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.sensor.pixel.sizemode": {
        "$type": None,
        "default": "scaled",
        "doc": "",
    },
    "display.style.sensor.pixel.color": {"$type": None, "default": None, "doc": ""},
    "display.style.sensor.pixel.symbol": {"$type": None, "default": "o", "doc": ""},
    "display.style.sensor.arrows.x.color": {"$type": None, "default": "red", "doc": ""},
    "display.style.sensor.arrows.y.color": {
        "$type": None,
        "default": "green",
        "doc": "",
    },
    "display.style.sensor.arrows.z.color": {
        "$type": None,
        "default": "blue",
        "doc": "",
    },
    "display.style.dipole.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.dipole.sizemode": {"$type": None, "default": "scaled", "doc": ""},
    "display.style.dipole.pivot": {"$type": None, "default": "middle", "doc": ""},
    "display.style.triangle.magnetization.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "",
    },
    "display.style.triangle.magnetization.arrow.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "",
    },
    "display.style.triangle.magnetization.arrow.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangle.magnetization.arrow.sizemode": {
        "$type": None,
        "default": "scaled",
        "doc": "",
    },
    "display.style.triangle.magnetization.arrow.offset": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangle.magnetization.arrow.width": {
        "$type": None,
        "default": 2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangle.magnetization.arrow.style": {
        "$type": None,
        "default": "solid",
        "doc": "",
    },
    "display.style.triangle.magnetization.arrow.color": {
        "$type": None,
        "default": None,
        "doc": "",
    },
    "display.style.triangle.magnetization.color.north": {
        "$type": None,
        "default": "#E71111",
        "doc": "",
    },
    "display.style.triangle.magnetization.color.middle": {
        "$type": None,
        "default": "#DDDDDD",
        "doc": "",
    },
    "display.style.triangle.magnetization.color.south": {
        "$type": None,
        "default": "#00B050",
        "doc": "",
    },
    "display.style.triangle.magnetization.color.transition": {
        "$type": None,
        "default": 0.2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangle.magnetization.color.mode": {
        "$type": None,
        "default": "tricolor",
        "doc": "",
    },
    "display.style.triangle.magnetization.mode": {
        "$type": None,
        "default": "auto",
        "doc": "",
    },
    "display.style.triangle.orientation.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "",
    },
    "display.style.triangle.orientation.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangle.orientation.color": {
        "$type": None,
        "default": "grey",
        "doc": "",
    },
    "display.style.triangle.orientation.offset": {
        "$type": None,
        "default": 0.9,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangle.orientation.symbol": {
        "$type": None,
        "default": "arrow3d",
        "doc": "",
    },
    "display.style.triangularmesh.orientation.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "",
    },
    "display.style.triangularmesh.orientation.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.orientation.color": {
        "$type": None,
        "default": "grey",
        "doc": "",
    },
    "display.style.triangularmesh.orientation.offset": {
        "$type": None,
        "default": 0.9,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.orientation.symbol": {
        "$type": None,
        "default": "arrow3d",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.grid.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "",
    },
    "display.style.triangularmesh.mesh.grid.line.width": {
        "$type": None,
        "default": 2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.grid.line.style": {
        "$type": None,
        "default": "solid",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.grid.line.color": {
        "$type": None,
        "default": "black",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.grid.marker.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.grid.marker.symbol": {
        "$type": None,
        "default": "o",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.grid.marker.color": {
        "$type": None,
        "default": "black",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.open.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "",
    },
    "display.style.triangularmesh.mesh.open.line.width": {
        "$type": None,
        "default": 2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.open.line.style": {
        "$type": None,
        "default": "solid",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.open.line.color": {
        "$type": None,
        "default": "cyan",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.open.marker.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.open.marker.symbol": {
        "$type": None,
        "default": "o",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.open.marker.color": {
        "$type": None,
        "default": "black",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.line.width": {
        "$type": None,
        "default": 2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.line.style": {
        "$type": None,
        "default": "solid",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.line.color": {
        "$type": None,
        "default": "black",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.marker.size": {
        "$type": None,
        "default": 5,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.marker.symbol": {
        "$type": None,
        "default": "o",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.marker.color": {
        "$type": None,
        "default": "black",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.disconnected.colorsequence": {
        "$type": None,
        "default": ("red", "blue", "green", "cyan", "magenta", "yellow"),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.selfintersecting.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "",
    },
    "display.style.triangularmesh.mesh.selfintersecting.line.width": {
        "$type": None,
        "default": 2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.selfintersecting.line.style": {
        "$type": None,
        "default": "solid",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.selfintersecting.line.color": {
        "$type": None,
        "default": "magenta",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.selfintersecting.marker.size": {
        "$type": None,
        "default": 1,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.triangularmesh.mesh.selfintersecting.marker.symbol": {
        "$type": None,
        "default": "o",
        "doc": "",
    },
    "display.style.triangularmesh.mesh.selfintersecting.marker.color": {
        "$type": None,
        "default": "black",
        "doc": "",
    },
    "display.style.markers.marker.size": {
        "$type": None,
        "default": 2,
        "bounds": (),
        "inclusive_bounds": (),
        "softbounds": (),
        "doc": "",
    },
    "display.style.markers.marker.color": {"$type": None, "default": "grey", "doc": ""},
    "display.style.markers.marker.symbol": {"$type": None, "default": "x", "doc": ""},
}


def convert_to_param(dict_, parent=None):
    parent = "" if not parent else parent[0].upper() + parent[1:]
    params = {}
    for key, val in dict_.items():
        if not isinstance(val, dict):
            raise TypeError(f"{val} must be dict.")
        typ = val.get("$type", None)
        if typ:
            params[key] = getattr(param, typ)(
                **{k: v for k, v in val.items() if k != "$type"}
            )
        else:
            name = parent + key[0].upper() + key[1:]
            val = convert_to_param(val, parent=name)
            params[key] = param.ClassSelector(class_=val, default=val())
    class_ = type(parent, (param.Parameterized,), params)
    return class_

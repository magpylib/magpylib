SUPPORTED_PLOTTING_BACKENDS = ("matplotlib", "plotly", "pyvista")

ALLOWED_PLOTTING_BACKENDS = ("auto", *SUPPORTED_PLOTTING_BACKENDS)
ALLOWED_SIZEMODES = ("scaled", "absolute")
ALLOWED_ORIENTATION_SYMBOLS = ("cone", "arrow3d")
ALLOWED_PIVOTS = ("tail", "middle", "tip")
ALLOWED_SYMBOLS = (".", "+", "D", "d", "s", "x", "o")
ALLOWED_LINESTYLES = (
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    "loosely dotted",
    "loosely dashdotted",
    "-",
    "--",
    "-.",
    ".",
    ":",
    (0, (1, 1)),
)

DEFAULTS = {
    "display.autosizefactor": {
        "$type": "Number",
        "default": 10,
        "bounds": (0, None),
        "inclusive_bounds": (False, True),
        "softbounds": (5, 15),
        "doc": """
        Defines at which scale objects like sensors and dipoles are displayed.
        -> object_size = canvas_size / autosizefactor""",
    },
    "display.animation.fps": {
        "$type": "Integer",
        "default": 20,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": "Target number of frames to be displayed per second.",
    },
    "display.animation.maxfps": {
        "$type": "Integer",
        "default": 30,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": "Maximum number of frames to be displayed per second before downsampling kicks in.",
    },
    "display.animation.maxframes": {
        "$type": "Integer",
        "default": 200,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": "Maximum total number of frames to be displayed before downsampling kicks in.",
    },
    "display.animation.time": {
        "$type": "Number",
        "default": 5,
        "bounds": (0, None),
        "inclusive_bounds": (False, None),
        "doc": "Default animation time.",
    },
    "display.animation.slider": {
        "$type": "Boolean",
        "default": True,
        "doc": """If True, an interactive slider will be displayed and stay in sync with the
        animation, will be hidden otherwise.""",
    },
    "display.animation.output": {
        "$type": "String",
        "default": "",
        "doc": "Animation output type (either `mp4` or `gif` or ending with `.mp4` or `.gif`)",
        "regex": r"^(mp4|gif|(.*\.(mp4|gif))?)$",
    },
    "display.backend": {
        "$type": "Selector",
        "default": "auto",
        "objects": ALLOWED_PLOTTING_BACKENDS,
        "doc": """
    Plotting backend to be used by default, if not explicitly set in the `display`
    function (e.g. 'matplotlib', 'plotly').
    Supported backends are defined in magpylib.SUPPORTED_PLOTTING_BACKENDS""",
    },
    "display.colorsequence": {
        "$type": "List",
        "item_type": "Color",
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
    "display.style.base.label": {
        "$type": "String",
        "default": "",
        "doc": "Object label.",
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
    "display.style.base.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Object explicit color",
    },
    "display.style.base.opacity": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 1),
        "inclusive_bounds": (True, True),
        "softbounds": (0, 1),
        "doc": """
        Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.""",
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
        "doc": "Explicit Path line color. Takes object color by default.",
    },
    "display.style.base.path.marker.size": {
        "$type": "Number",
        "default": 3,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
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
    "display.style.base.path.frames.indices": {
        "$type": "List",
        "default": [-1],
        "item_type": int,
        "doc": "Array_like shape (n,) of integers: describes certain path indices.",
    },
    "display.style.base.path.frames.step": {
        "$type": "Integer",
        "default": 1,
        "bounds": (1, None),
        "softbounds": (0, 10),
        "doc": "Displays the object(s) at every i'th path position.",
    },
    "display.style.base.path.frames.mode": {
        "$type": "Selector",
        "default": "indices",
        "objects": ("indices", "step"),
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
    "display.style.base.model3d.showdefault": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide default 3D-model.",
    },
    "display.style.base.model3d.data": {
        "$type": "List",
        "default": [],
        "item_type": "Trace3d",
        "doc": """
        A trace or list of traces where each is an instance of `Trace3d` or dictionary of
        equivalent key/value pairs. Defines properties for an additional user-defined model3d
        object which is positioned relatively to the main object to be displayed and moved
        automatically with it. This feature also allows the user to replace the original 3d
        representation of the object.""",
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
        Magnetization arrow offset. `offset=0` results in the arrow head to be
        coincident with the start of the line, and `offset=1` with the end.""",
    },
    "display.style.magnet.magnetization.arrow.width": {
        "$type": "Number",
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Magnetization arrow line width",
    },
    "display.style.magnet.magnetization.arrow.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Magnetization arrow line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.magnet.magnetization.arrow.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Explicit magnetization arrow color. Takes the object color by default.",
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
    "display.style.current.arrow.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide current arrow.",
    },
    "display.style.current.arrow.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Current arrow size.",
    },
    "display.style.current.arrow.sizemode": {
        "$type": "Selector",
        "default": "scaled",
        "objects": ALLOWED_SIZEMODES,
        "doc": f"The way the current arrow size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`",
    },
    "display.style.current.arrow.offset": {
        "$type": "Number",
        "default": 0.5,
        "bounds": (0, 1),
        "inclusive_bounds": (True, True),
        "softbounds": (0, 1),
        "doc": """
        Current arrow offset. `offset=0` results in the arrow head to be coincident
        with the start of the line, and `offset=1` with the end.""",
    },
    "display.style.current.arrow.width": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Current arrow line width",
    },
    "display.style.current.arrow.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Current arrow line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.current.arrow.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Explicit current arrow color. Takes the object color by default.",
    },
    "display.style.current.line.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "",
    },
    "display.style.current.line.width": {
        "$type": "Number",
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Current line width.",
    },
    "display.style.current.line.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Current line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.current.line.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Explicit current line color. Takes object color by default.",
    },
    "display.style.sensor.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Sensor size.",
    },
    "display.style.sensor.sizemode": {
        "$type": "Selector",
        "default": "scaled",
        "objects": ALLOWED_SIZEMODES,
        "doc": f"The way the sensor size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`",
    },
    "display.style.sensor.pixel.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Sensor pixel size.",
    },
    "display.style.sensor.pixel.sizemode": {
        "$type": "Selector",
        "default": "scaled",
        "objects": ALLOWED_SIZEMODES,
        "doc": f"The way the sensor pixel size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`",
    },
    "display.style.sensor.pixel.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Sensor pixel color.",
    },
    "display.style.sensor.pixel.symbol": {
        "$type": "Selector",
        "default": "o",
        "objects": ALLOWED_SYMBOLS,
        "doc": f"Pixel symbol. Can be one of: {ALLOWED_SYMBOLS}.",
    },
    "display.style.sensor.arrows.x.color": {
        "$type": "Color",
        "default": "#ff0000",
        "doc": "Sensor x-arrow color.",
    },
    "display.style.sensor.arrows.x.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide sensor x-arrow.",
    },
    "display.style.sensor.arrows.y.color": {
        "$type": "Color",
        "default": "#008000",
        "doc": "Sensor y-arrow color.",
    },
    "display.style.sensor.arrows.y.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide sensor y-arrow.",
    },
    "display.style.sensor.arrows.z.color": {
        "$type": "Color",
        "default": "#0000FF",
        "doc": "Sensor z-arrow color.",
    },
    "display.style.sensor.arrows.z.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide sensor z-arrow.",
    },
    "display.style.dipole.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Dipole size.",
    },
    "display.style.dipole.sizemode": {
        "$type": "Selector",
        "default": "scaled",
        "objects": ALLOWED_SIZEMODES,
        "doc": f"The way the dipole size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`",
    },
    "display.style.dipole.pivot": {
        "$type": "Selector",
        "default": "middle",
        "objects": ALLOWED_PIVOTS,
        "doc": f"""
        The part of the arrow that is anchored to the X, Y grid. The arrow rotates about
        this point. Can be one of `{ALLOWED_PIVOTS}`""",
    },
    "display.style.triangle.magnetization.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide magnetization indication (arrow and/or color).",
    },
    "display.style.triangle.magnetization.arrow.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide magnetization arrow.",
    },
    "display.style.triangle.magnetization.arrow.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Magnetization arrow size.",
    },
    "display.style.triangle.magnetization.arrow.sizemode": {
        "$type": "Selector",
        "default": "scaled",
        "objects": ALLOWED_SIZEMODES,
        "doc": f"The way the object size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`",
    },
    "display.style.triangle.magnetization.arrow.offset": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 1),
        "inclusive_bounds": (True, True),
        "softbounds": (0, 1),
        "doc": """
        Magnetization arrow offset. `offset=0` results in the arrow head to be
        coincident with the start of the line, and `offset=1` with the end.""",
    },
    "display.style.triangle.magnetization.arrow.width": {
        "$type": "Number",
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Magnetization arrow line width",
    },
    "display.style.triangle.magnetization.arrow.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Triangle magnetization arrow line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.triangle.magnetization.arrow.color": {
        "$type": "Color",
        "default": None,
        "allow_None": True,
        "doc": "Explicit triangle magnetization arrow color. Takes the object color by default.",
    },
    "display.style.triangle.magnetization.color.north": {
        "$type": "Color",
        "default": "#E71111",
        "doc": "The color of the magnetic north pole.",
    },
    "display.style.triangle.magnetization.color.middle": {
        "$type": "Color",
        "default": "#DDDDDD",
        "doc": "The color between the magnetic poles.",
    },
    "display.style.triangle.magnetization.color.south": {
        "$type": "Color",
        "default": "#00B050",
        "doc": "The color of the magnetic south pole.",
    },
    "display.style.triangle.magnetization.color.transition": {
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
    "display.style.triangle.magnetization.color.mode": {
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
    "display.style.triangle.magnetization.mode": {
        "$type": "Selector",
        "default": "auto",
        "objects": ("auto", "arrow", "color", "arrow+color", "color+arrow"),
        "doc": """
        One of {"auto", "arrow", "color", "arrow+color"}, default="auto"
        Magnetization can be displayed via arrows, color or both. By default `mode='auto'` means
        that the chosen backend determines which mode is applied by its capability. If the backend
        can display both and `auto` is chosen, the priority is given to `color`.""",
    },
    "display.style.triangle.orientation.show": {
        "$type": "Boolean",
        "default": True,
        "doc": "Show/hide orientation symbol.",
    },
    "display.style.triangle.orientation.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Size of the orientation symbol",
    },
    "display.style.triangle.orientation.color": {
        "$type": "Color",
        "default": "grey",
        "doc": "Explicit orientation symbol color. Takes the objet color by default.",
    },
    "display.style.triangle.orientation.offset": {
        "$type": "Number",
        "default": 0.9,
        "bounds": (-2, 2),
        "inclusive_bounds": (True, True),
        "softbounds": (-0.9, 0.9),
        "doc": """
        Orientation symbol offset, normal to the triangle surface. `offset=0` results
        in the cone/arrow head to be coincident to the triangle surface and `offset=1` with the
        base""",
    },
    "display.style.triangle.orientation.symbol": {
        "$type": "Selector",
        "default": "arrow3d",
        "objects": ALLOWED_ORIENTATION_SYMBOLS,
        "doc": f"""
        Orientation symbol for the triangular faces. Can be one of:
        {ALLOWED_ORIENTATION_SYMBOLS}""",
    },
    "display.style.triangularmesh.orientation.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "Show/hide orientation symbol.",
    },
    "display.style.triangularmesh.orientation.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, None),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Size of the orientation symbol",
    },
    "display.style.triangularmesh.orientation.color": {
        "$type": "Color",
        "default": "grey",
        "doc": "Explicit orientation symbol color. Takes the objet color by default.",
    },
    "display.style.triangularmesh.orientation.offset": {
        "$type": "Number",
        "default": 0.9,
        "bounds": (-2, 2),
        "inclusive_bounds": (True, True),
        "softbounds": (-0.9, 0.9),
        "doc": """
        Orientation symbol offset, normal to the triangle surface. `offset=0` results
        in the cone/arrow head to be coincident to the triangle surface and `offset=1` with the
        base""",
    },
    "display.style.triangularmesh.orientation.symbol": {
        "$type": "Selector",
        "default": "arrow3d",
        "objects": ALLOWED_ORIENTATION_SYMBOLS,
        "doc": f"""
        Orientation symbol for the triangular faces. Can be one of:
        {ALLOWED_ORIENTATION_SYMBOLS}""",
    },
    "display.style.triangularmesh.mesh.grid.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "Show/hide mesh grid",
    },
    "display.style.triangularmesh.mesh.grid.line.width": {
        "$type": "Number",
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh grid line width.",
    },
    "display.style.triangularmesh.mesh.grid.line.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Mesh grid line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.triangularmesh.mesh.grid.line.color": {
        "$type": "Color",
        "default": "#000000",
        "doc": "Explicit current line color. Takes object color by default.",
    },
    "display.style.triangularmesh.mesh.grid.marker.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh grid marker size.",
    },
    "display.style.triangularmesh.mesh.grid.marker.symbol": {
        "$type": "Selector",
        "default": "o",
        "objects": ALLOWED_SYMBOLS,
        "doc": f"Mesh grid marker symbol. Can be one of: {ALLOWED_SYMBOLS}.",
    },
    "display.style.triangularmesh.mesh.grid.marker.color": {
        "$type": "Color",
        "default": "#000000",
        "doc": "Mesh grid marker color.",
    },
    "display.style.triangularmesh.mesh.open.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "Show/hide mesh open",
    },
    "display.style.triangularmesh.mesh.open.line.width": {
        "$type": "Number",
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh open line width.",
    },
    "display.style.triangularmesh.mesh.open.line.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Mesh open line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.triangularmesh.mesh.open.line.color": {
        "$type": "Color",
        "default": "#00FFFF",
        "doc": "Explicit current line color. Takes object color by default.",
    },
    "display.style.triangularmesh.mesh.open.marker.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh open marker size.",
    },
    "display.style.triangularmesh.mesh.open.marker.symbol": {
        "$type": "Selector",
        "default": "o",
        "objects": ALLOWED_SYMBOLS,
        "doc": f"Mesh open marker symbol. Can be one of: {ALLOWED_SYMBOLS}.",
    },
    "display.style.triangularmesh.mesh.open.marker.color": {
        "$type": "Color",
        "default": "#000000",
        "doc": "Mesh open marker color.",
    },
    "display.style.triangularmesh.mesh.disconnected.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "Show/hide mesh disconnected",
    },
    "display.style.triangularmesh.mesh.disconnected.line.width": {
        "$type": "Number",
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh disconnected line width.",
    },
    "display.style.triangularmesh.mesh.disconnected.line.style": {
        "$type": "Selector",
        "default": "dashed",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Mesh disconnected line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.triangularmesh.mesh.disconnected.line.color": {
        "$type": "Color",
        "default": "#FF00FF",
        "doc": "Explicit current line color. Takes object color by default.",
    },
    "display.style.triangularmesh.mesh.disconnected.marker.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh disconnected marker size.",
    },
    "display.style.triangularmesh.mesh.disconnected.marker.symbol": {
        "$type": "Selector",
        "default": "o",
        "objects": ALLOWED_SYMBOLS,
        "doc": f"Mesh disconnected marker symbol. Can be one of: {ALLOWED_SYMBOLS}.",
    },
    "display.style.triangularmesh.mesh.disconnected.marker.color": {
        "$type": "Color",
        "default": "#000000",
        "doc": "Mesh disconnected marker color.",
    },
    "display.style.triangularmesh.mesh.disconnected.colorsequence": {
        "$type": "List",
        "item_type": "Color",
        "default": ["#FF0000", "#0000FF", "#008000", "#00FFFF", "#FF00FF", "#FFFF00"],
        "doc": """
        An iterable of color values used to cycle trough for every disconnected part to be
        displayed. A color may be specified by
            - a hex string (e.g. '#ff0000')
            - an rgb/rgba string (e.g. 'rgb(255,0,0)')
            - an hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - an hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - a named CSS color""",
    },
    "display.style.triangularmesh.mesh.selfintersecting.show": {
        "$type": "Boolean",
        "default": False,
        "doc": "Show/hide mesh selfintersecting",
    },
    "display.style.triangularmesh.mesh.selfintersecting.line.width": {
        "$type": "Number",
        "default": 2,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh selfintersecting line width.",
    },
    "display.style.triangularmesh.mesh.selfintersecting.line.style": {
        "$type": "Selector",
        "default": "solid",
        "objects": ALLOWED_LINESTYLES,
        "doc": f"Mesh selfintersecting line style. Can be one of: {ALLOWED_LINESTYLES}.",
    },
    "display.style.triangularmesh.mesh.selfintersecting.line.color": {
        "$type": "Color",
        "default": "#000000",
        "doc": "Explicit current line color. Takes object color by default.",
    },
    "display.style.triangularmesh.mesh.selfintersecting.marker.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Mesh selfintersecting marker size.",
    },
    "display.style.triangularmesh.mesh.selfintersecting.marker.symbol": {
        "$type": "Selector",
        "default": "o",
        "objects": ALLOWED_SYMBOLS,
        "doc": f"Mesh selfintersecting marker symbol. Can be one of: {ALLOWED_SYMBOLS}.",
    },
    "display.style.triangularmesh.mesh.selfintersecting.marker.color": {
        "$type": "Color",
        "default": "#000000",
        "doc": "Mesh selfintersecting marker color.",
    },
    "display.style.markers.marker.size": {
        "$type": "Number",
        "default": 1,
        "bounds": (0, 20),
        "inclusive_bounds": (True, True),
        "softbounds": (1, 5),
        "doc": "Markers marker size.",
    },
    "display.style.markers.marker.color": {
        "$type": "Color",
        "default": "#808080",
        "doc": "Markers marker color.",
    },
    "display.style.markers.marker.symbol": {
        "$type": "Selector",
        "default": "o",
        "objects": ALLOWED_SYMBOLS,
        "doc": f"Markers marker symbol. Can be one of: {ALLOWED_SYMBOLS}.",
    },
}

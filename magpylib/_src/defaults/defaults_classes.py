import param

from magpylib._src.defaults.defaults_utility import color_validator
from magpylib._src.defaults.defaults_utility import magic_to_dict
from magpylib._src.defaults.defaults_utility import MagicParameterized
from magpylib._src.defaults.defaults_values import DEFAULTS
from magpylib._src.defaults.defaults_values import SUPPORTED_PLOTTING_BACKENDS

# pylint: disable=missing-class-docstring


class Trace3d(MagicParameterized):
    _allowed_backends = (*SUPPORTED_PLOTTING_BACKENDS, "generic")

    def __setattr__(self, name, value):
        validation_func = getattr(self, f"_validate_{name}", None)
        if validation_func is not None:
            value = validation_func(value)
        return super().__setattr__(name, value)

    backend = param.Selector(
        default="generic",
        objects=_allowed_backends,
        doc=f"""
        Plotting backend corresponding to the trace. Can be one of
        {_allowed_backends}""",
    )

    constructor = param.String(
        doc="""
        Model constructor function or method to be called to build a 3D-model object
        (e.g. 'plot_trisurf', 'Mesh3d). Must be in accordance with the given plotting backend."""
    )

    args = param.Parameter(
        default=(),
        doc="""
        Tuple or callable returning a tuple containing positional arguments for building a
        3D-model object.""",
    )

    kwargs = param.Parameter(
        default={},
        doc="""
        Dictionary or callable returning a dictionary containing the keys/values pairs for
        building a 3D-model object.""",
    )

    coordsargs = param.Dict(
        default=None,
        doc="""
        Tells Magpylib the name of the coordinate arrays to be moved or rotated.
        by default:
            - plotly backend: `{"x": "x", "y": "y", "z": "z"}`
            - matplotlib backend: `{"x": "args[0]", "y": "args[1]", "z": "args[2]}"`""",
    )

    show = param.Boolean(
        default=True,
        doc="""Show/hide model3d object based on provided trace.""",
    )

    scale = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, False),
        softbounds=(0.1, 5),
        doc="""
        Scaling factor by which the trace vertices coordinates should be multiplied by.
        Be aware that if the object is not centered at the global CS origin, its position will also
        be affected by scaling.""",
    )

    updatefunc = param.Callable(
        doc="""
        Callable object with no arguments. Should return a dictionary with keys from the
        trace parameters. If provided, the function is called at `show` time and updates the
        trace parameters with the output dictionary. This allows to update a trace dynamically
        depending on class attributes, and postpone the trace construction to when the object is
        displayed."""
    )

    def _validate_coordsargs(self, value):
        assert isinstance(value, dict) and all(key in value for key in "xyz"), (
            f"the `coordsargs` property of {type(self).__name__} must be "
            f"a dictionary with `'x', 'y', 'z'` keys"
            f" but received {repr(value)} instead"
        )
        return value

    def _validate_updatefunc(self, val):
        """Validate updatefunc."""
        if val is None:

            def val():
                return {}

        msg = ""
        valid_keys = self.param.values().keys()
        if not callable(val):
            msg = f"Instead received {type(val)}"
        else:
            test_val = val()
            if not isinstance(test_val, dict):
                msg = f"but callable returned type {type(test_val)}."
            else:
                bad_keys = [k for k in test_val.keys() if k not in valid_keys]
                if bad_keys:
                    msg = f"but invalid output dictionary keys received: {bad_keys}."

        assert msg == "", (
            f"The `updatefunc` property of {type(self).__name__} must be a callable returning a "
            f"dictionary with a subset of following keys: {list(valid_keys)}.\n"
            f"{msg}"
        )
        return val


class Model3d(MagicParameterized):
    def __setattr__(self, name, value):
        if name == "data":
            value = self._validate_data(value)
        return super().__setattr__(name, value)

    showdefault = param.Boolean(
        default=True,
        doc="""Show/hide default 3D-model.""",
    )

    data = param.List(
        item_type=Trace3d,
        doc="""
        A trace or list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.
        """,
    )

    @staticmethod
    def _validate_trace(trace, **kwargs):
        updatefunc = None
        if trace is None:
            trace = Trace3d()
        if not isinstance(trace, Trace3d) and callable(trace):
            updatefunc = trace
            trace = Trace3d()
        if isinstance(trace, dict):
            trace = Trace3d(**trace)
        if isinstance(trace, Trace3d):
            trace.updatefunc = updatefunc
            if kwargs:
                trace.update(**kwargs)
            trace.update(trace.updatefunc())
        return trace

    def _validate_data(self, traces):
        if traces is None:
            traces = []
        elif not isinstance(traces, (list, tuple)):
            traces = [traces]
        new_traces = []
        for trace in traces:
            new_traces.append(self._validate_trace(trace))
        return new_traces

    def add_trace(self, trace=None, **kwargs):
        """Adds user-defined 3d model object which is positioned relatively to the main object to be
        displayed and moved automatically with it. This feature also allows the user to replace the
        original 3d representation of the object.

        trace: Trace3d instance, dict or callable
            Trace object. Can be a `Trace3d` instance or an dictionary with equivalent key/values
            pairs, or a callable returning the equivalent dictionary.

        backend: str
            Plotting backend corresponding to the trace. Can be one of `['matplotlib', 'plotly']`.

        constructor: str
            Model constructor function or method to be called to build a 3D-model object
            (e.g. 'plot_trisurf', 'Mesh3d). Must be in accordance with the given plotting backend.

        args: tuple, default=None
            Tuple or callable returning a tuple containing positional arguments for building a
            3D-model object.

        kwargs: dict or callable, default=None
            Dictionary or callable returning a dictionary containing the keys/values pairs for
            building a 3D-model object.

        coordsargs: dict, default=None
            Tells magpylib the name of the coordinate arrays to be moved or rotated,
                by default: `{"x": "x", "y": "y", "z": "z"}`, if False, object is not rotated.

        show: bool, default=None
            Show/hide model3d object based on provided trace.

        scale: float, default=1
            Scaling factor by which the trace vertices coordinates are multiplied.

        updatefunc: callable, default=None
            Callable object with no arguments. Should return a dictionary with keys from the
            trace parameters. If provided, the function is called at `show` time and updates the
            trace parameters with the output dictionary. This allows to update a trace dynamically
            depending on class attributes, and postpone the trace construction to when the object is
            displayed.
        """
        self.data = list(self.data) + [self._validate_trace(trace, **kwargs)]
        return self


class ColorSequence(param.List):
    def __set__(self, obj, val):
        self._validate_value(val, self.allow_None)
        val = [color_validator(v) for v in val]
        super().__set__(obj, val)


class Model3dData(param.List):
    def __set__(self, obj, val):
        self._validate_value(val, self.allow_None)
        val = [self._validate_trace(v) for v in val]
        super().__set__(obj, val)

    @staticmethod
    def _validate_trace(trace, **kwargs):
        updatefunc = None
        if trace is None:
            trace = Trace3d()
        if not isinstance(trace, Trace3d) and callable(trace):
            updatefunc = trace
            trace = Trace3d()
        if isinstance(trace, dict):
            trace = Trace3d(**trace)
        if isinstance(trace, Trace3d):
            trace.updatefunc = updatefunc
            if kwargs:
                trace.update(**kwargs)
            trace.update(trace.updatefunc())
        return trace


def convert_to_param(dict_, parent=None):
    """Convert nested defaults dict to nested MagicParameterized instances"""
    parent = "" if not parent else parent[0].upper() + parent[1:]
    params = {}
    for key, val in dict_.items():
        if not isinstance(val, dict):
            raise TypeError(f"{val} must be dict.")
        if "$type" in val:
            typ = None
            typ_str = str(val["$type"]).capitalize()
            args = {k: v for k, v in val.items() if k != "$type"}
            if typ_str == "Color":
                typ = param.Color
            elif typ_str == "List":
                it_typ = str(args.get("item_type", None)).capitalize()
                if it_typ == "Color":
                    args.pop("item_type", None)
                    typ = ColorSequence
                elif it_typ == "Trace3d":
                    args.pop("item_type", None)
                    typ = Model3dData
            else:
                typ = getattr(param, typ_str)
            if typ is not None:
                params[key] = typ(**args)
        else:
            name = parent + key[0].upper() + key[1:]
            val = convert_to_param(val, parent=name)
            params[key] = param.ClassSelector(class_=val, default=val())
    class_ = type(parent, (MagicParameterized,), params)
    return class_


default_settings = convert_to_param(
    magic_to_dict(DEFAULTS, separator="."), parent="Settings"
)

default_style_classes = {
    k: v.__class__
    for k, v in default_settings.display.style.param.values().items()
    if isinstance(v, param.Parameterized)
}
locals()["BaseStyle"] = base = default_style_classes.pop("base")
for fam, klass in default_style_classes.items():
    klass_name = f"{fam.capitalize()}Style"
    locals()[klass_name] = type(klass_name, (base, klass), {})

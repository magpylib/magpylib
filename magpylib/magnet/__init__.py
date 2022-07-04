"""
The `magpylib.magnet` subpackage contains all magnet classes.
"""
import importlib
from pkgutil import iter_modules

from magpylib._src import obj_classes

classes = []

for submodule in iter_modules(obj_classes.__path__):
    if submodule.name.startswith("class_magnet"):
        _, typ, cls_name = submodule.name.split("_")
        classes.append(cls_name)
        module = importlib.import_module(
            f"{obj_classes.__name__}.{submodule.name}", cls_name
        )
        vars()[cls_name] = getattr(module, cls_name)

__all__ = classes

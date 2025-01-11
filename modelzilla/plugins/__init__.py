import os
import traceback
from importlib import util
import inspect

import supervision as sv


InferenceResult = sv.Detections | sv.KeyPoints


class IPluginRegistry(type):
    plugins = {}

    def __init__(cls, name, bases, attrs):
        if name != "CLIPlugin":
            IPluginRegistry.plugins[name] = cls


class CLIPlugin(object, metaclass=IPluginRegistry):
    """Plugin class to be inherited by all plugin classes that want to be discovered
    by the plugin manager to be used by the CLI.
    """

    @classmethod
    def build_cmd_parser(cls, parser):
        """Automatically adds arguments to the parser based on the __init__ signature."""
        signature = inspect.signature(cls.__init__)
        for param_name, param in signature.parameters.items():
            if param_name == "self":  # Skip 'self' argument
                continue

            # Determine the type for argparse
            param_type = param.annotation if param.annotation != inspect._empty else str

            # Check if the parameter has a default value
            if param.default != inspect._empty:
                parser.add_argument(
                    f"--{param_name}",
                    type=param_type,
                    default=param.default,
                    help=f"Default: {param.default}",
                )
            else:
                parser.add_argument(
                    f"--{param_name}",
                    type=param_type,
                    required=True,
                    help="Required argument",
                )

    def inference(self) -> InferenceResult:
        """The inherited classes must implement the inference code here."""
        raise NotImplementedError


path = os.path.abspath(__file__)
current_file_path = os.path.dirname(path)


def discover_plugins(dirpath: str = None) -> dict:
    """Discover the plugin classes contained in Python files, given a
        list of directory names to scan. The default directory is the `plugins`
        folder from this library. If a directory is provided, it will be scanned
        for plugins too.

    Args:
        dirpath (str, optional): directory path to scan for plugins. Defaults to None.

    Returns:
        dict: Dictonary of plugin classes.
    """
    folders = [current_file_path]

    if dirpath is not None:
        folders.append(dirpath)

    def load_module(path):
        name = os.path.split(path)[-1]
        spec = util.spec_from_file_location(name, path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    for fol_path in folders:
        for fname in os.listdir(fol_path):
            if (
                not fname.startswith(".")
                and not fname.startswith("__")
                and fname.endswith(".py")
            ):
                try:
                    load_module(os.path.join(fol_path, fname))
                except Exception:
                    traceback.print_exc()

    return IPluginRegistry.plugins

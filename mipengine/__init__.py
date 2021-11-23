import glob
import importlib
from importlib import util
import os
from os.path import basename
from os.path import isfile

from .datatypes import DType
from .attrdict import AttrDict

__all__ = ["DType", "AttrDict", "ALGORITHMS_FOLDER_ENV_VARIABLE", "ALGORITHMS_FOLDER"]

ALGORITHMS_FOLDER_ENV_VARIABLE = "ALGORITHMS_FOLDER"
ALGORITHMS_FOLDER = "./mipengine/algorithms"

if os.getenv(ALGORITHMS_FOLDER_ENV_VARIABLE):
    ALGORITHMS_FOLDER = os.getenv(ALGORITHMS_FOLDER_ENV_VARIABLE)


def get_algorithm_modules():
    # Import all algorithms
    # Import all .py modules in the algorithms folder path
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?page=1&tab=votes#tab-top
    all_module_paths = glob.glob(f"{ALGORITHMS_FOLDER}/*.py")
    algorithm_module_paths = [
        module
        for module in all_module_paths
        if isfile(module) and not module.endswith("__init__.py")
    ]
    algorithm_names = [
        basename(module_path)[:-3] for module_path in algorithm_module_paths
    ]
    specs = [
        importlib.util.spec_from_file_location(algorithm_name, module_path)
        for algorithm_name, module_path in zip(algorithm_names, algorithm_module_paths)
    ]
    modules = {
        name: importlib.util.module_from_spec(spec)
        for name, spec in zip(algorithm_names, specs)
    }

    # Import modules
    [spec.loader.exec_module(module) for spec, module in zip(specs, modules.values())]

    return modules


algorithm_modules = get_algorithm_modules()

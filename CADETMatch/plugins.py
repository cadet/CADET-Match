import importlib
import importlib.util

from pathlib import Path

base = Path(__file__).parent

def load_plugin(path):
    module = '.'.join(path.relative_to(base).parts).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module, str(path))
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

def get_files(dir):
    return [path for path in dir.glob('*.py') if not path.name == '__init__.py']

def get_plugins(directory):
    plugins = base / directory
    temp = {}
    for path in get_files(plugins):
        plug = load_plugin(path)
        temp[plug.name] = plug
    return temp

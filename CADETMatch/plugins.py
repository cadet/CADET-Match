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

def call_plugin_function(plugin_path, attribute_name, *args, **kwargs):
    return getattr(load_plugin(plugin_path), attribute_name)(*args, **kwargs)

def call_plugin_by_name(name, directory, attribute_name, *args, **kwargs):
    plugins = base / directory
    for path in get_files(plugins):
        plugin_name = get_plugin_attribute(path, 'name')
        if plugin_name == name:
            return call_plugin_function(path, attribute_name, *args, **kwargs)

def call_plugin_by_id(name, attribute_name, *args, **kwargs):
    module = importlib.import_module(name)
    return getattr(module, attribute_name)(*args, **kwargs)

def call_plugins_by_name(directory, attribute_name, *args, **kwargs):
    plugins = base / directory
    temp = []
    for path in get_files(plugins):
        temp.append( call_plugin_function(path, attribute_name, *args, **kwargs) )
    return temp

def get_attribute_by_name(name, directory, attribute_name):
    plugins = base / directory
    for path in get_files(plugins):
        plugin_name = get_plugin_attribute(path, 'name')
        if plugin_name == name:
            return get_plugin_attribute(path, attribute_name)

def get_plugin_attribute(path, attribute_name):
    return getattr(load_plugin(path), attribute_name)

def get_files(dir):
    return [path for path in dir.glob('*.py') if not path.name == '__init__.py']

def get_plugin_names(directory):
    plugins = base / directory
    return [get_plugin_attribute(path, 'name') for path in get_files(plugins)]

def get_plugins(directory):
    plugins = base / directory
    temp = {}
    for path in get_files(plugins):
        temp[get_plugin_attribute(path, 'name')] = load_plugin(path)
    return temp

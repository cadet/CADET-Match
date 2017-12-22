import importlib
import os.path
import glob

def load_plugin(path):
    module_name = os.path.relpath(path, simulation_path).replace('/', '.').replace('.py', '')
    return importlib.import_module(module_name)

def call_plugin_function(plugin_path, attribute_name, *args, **kwargs):
    return getattr(load_plugin(plugin_path), attribute_name)(*args, **kwargs)

def call_plugin_by_name(name, directory, attribute_name, *args, **kwargs):
    for path in get_files(os.path.join(plugins, directory, '*.py')):
        plugin_name = get_plugin_attribute(path, 'name')
        if plugin_name == name:
            return call_plugin_function(path, attribute_name, *args, **kwargs)

def call_plugin_by_id(name, attribute_name, *args, **kwargs):
    module = importlib.import_module(name)
    return getattr(module, attribute_name)(*args, **kwargs)

def call_plugins_by_name(directory, attribute_name, *args, **kwargs):
    temp = []
    for path in get_files(os.path.join(plugins, directory, '*.py')):
        temp.append( call_plugin_function(path, attribute_name, *args, **kwargs) )
    return temp

def get_attribute_by_name(name, directory, attribute_name):
    for path in get_files(os.path.join(plugins, directory, '*.py')):
        plugin_name = get_plugin_attribute(path, 'name')
        if plugin_name == name:
            return get_plugin_attribute(path, attribute_name)

def get_plugin_attribute(path, attribute_name):
    return getattr(load_plugin(path), attribute_name)

def get_files(path):
    return [path for path in glob.glob(path) if not path.endswith('__init__.py')]

def get_plugin_names(directory):
    return [get_plugin_attribute(path, 'name') for path in get_files(os.path.join(plugins, directory, '*.py'))]

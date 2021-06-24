import ast
import inspect
import dis
from contextlib import contextmanager
from functools import reduce
from importlib import import_module
import os
from collections import namedtuple
from inspect import getsource, getsourcefile

Import = namedtuple('Import', ['module', 'name', 'alias'])


# TODO: Generalize attrs_used_by_method to attrs_used_by_func.


def _get_ast_root_from(o):
    source_str = None
    source_filepath = None
    if isinstance(o, str) and os.path.isfile(o):
        source_filepath = o
        with open(source_filepath) as fh:
            source_str = fh.read()
    elif not isinstance(o, ast.AST):  # not an AST node...
        source_filepath = getsourcefile(o)
        source_str = getsource(o)
        if not isinstance(source_filepath, str) and isinstance(source_str, str):
            raise ValueError('Unrecognized object format')

    return ast.parse(source=source_str, filename=source_filepath)


def _get_imports_from_ast_root(ast_root, recursive=False):
    for node in ast.iter_child_nodes(ast_root):
        module = None
        if isinstance(node, ast.Import):
            module = []
        elif isinstance(node, ast.ImportFrom):
            module = node.module.split('.')

        if module is not None:
            for n in node.names:
                yield Import(module, n.name.split('.'), n.asname)

        if recursive:
            yield from _get_imports_from_ast_root(node, recursive=recursive)


def get_imports_from_obj(o, recursive=False):
    """ Getting imports for an object (usually, module) """
    root = _get_ast_root_from(o)
    yield from _get_imports_from_ast_root(root, recursive)


def _process_duplicates(a, remove_duplicates=True):
    if remove_duplicates:
        return set(a)
        ## remove duplicates conserving order
        # return list(dict.fromkeys(a))
    else:
        return a


def get_class_that_defined_method(method):
    """
    Get class for unbound/bound method.
    """
    if inspect.ismethod(method):
        for cls in inspect.getmro(method.__self__.__class__):
            if cls.__dict__.get(method.__name__) is method:
                return cls
        method = method.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(method):
        cls = getattr(
            inspect.getmodule(method),
            method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
        )
        if isinstance(cls, type):
            return cls
    return getattr(method, '__objclass__', None)


def cls_and_method_name_of_method(method):
    return get_class_that_defined_method(method), method.__name__


def _alt_cls_and_method_name_of_method(
    method,
):  # TODO: Delete when determined to be of no additional value
    method_path = method.__qualname__.split('.')
    name = method_path[-1]
    cls = reduce(getattr, method_path[:-1], import_module(method.__module__))
    return cls, name


def list_func_calls(fn):
    """
    Extracts functions and methods called from fn
    :param fn: reference to function or method
    :return: a list of functions or methods names
    """
    funcs = []
    bytecode = dis.Bytecode(fn)
    instrs = list(reversed([instr for instr in bytecode]))
    for (ix, instr) in enumerate(instrs):
        if instr.opname == 'CALL_FUNCTION' or instr.opname == 'CALL_METHOD':
            load_func_instr = instrs[ix + instr.arg + 1]
            funcs.append(load_func_instr.argval)
    return [funcname for funcname in reversed(funcs)]


def attr_list(root, func_name):
    """
    Extracts attributes from ast tree processing only func_name function or method
    :param root: root node of ast tree
    :param func_name: name of the function
    :return: a list of attributes names
    """
    atts = []
    functions = sorted(
        {node.name for node in ast.walk(root) if isinstance(node, ast.FunctionDef)}
    )
    for root in ast.walk(root):
        if isinstance(root, ast.FunctionDef) and root.name == func_name:
            for child in ast.walk(root):
                if (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.ctx, ast.Load)
                    and child.attr not in functions
                ):
                    atts.append(child.attr)

    return atts


def _attrs_used_by_method(cls, method_name, remove_duplicates=True):
    """
    Util for attrs_used_by_method. Same output as attrs_used_by_method, but intput is (cls, method_name)
    """
    f = open(inspect.getsourcefile(cls), 'r')
    if f.mode == 'r':
        src = f.read()
    root = ast.parse(src)
    funcs = list_func_calls(getattr(cls, method_name))
    attrs = []
    for func in funcs:
        attrs = attrs + attr_list(root, func)

    return _process_duplicates(
        attrs + attr_list(root, method_name), remove_duplicates=True
    )


def attrs_used_by_method(method, remove_duplicates=True):
    """
    Extracts a list of cls attributes which are used by a method or method_name function
    Args:
        method: The method (object) to analyze

    Returns:
        A list of attribute names (of the class or instance thereof) that are accessed in the code of the said method.

    Example:

    Consider the method `A.target_method` coming from the following code in
    `i2.tests.footprints_test`:
    ```python
    def func(obj):
        \"\"\"Accesses attributes 'a' and 'b' of obj\"\"\"
        return obj.a + obj.b

    class A:
        e = 2

        def __init__(self, a=1, b=0, c=1, d=10):
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        def target_method(self, x):
            \"\"\"Accesses ['a', 'b', 'c', 'e']\"\"\"
            t = func(self)  # and this function will access some attributes!
            tt = self.other_method(t)
            return x * tt / self.e

        def other_method(self, x=1):
            \"\"\"Accesses ['c', 'e']\"\"\"
            w = self.c * 2  # c is accessed first
            return self.e + self.c * x - w  # and c is accessed again

        @classmethod
        def a_class_method(cls, y):
            \"\"\"Accesses ['e']\"\"\"
            return cls.e + y
    ```

    >>> from i2.tests.footprints_test import A
    >>> assert attrs_used_by_method(A.target_method) == {'a', 'b', 'c', 'e'}
    """
    return _attrs_used_by_method(
        *cls_and_method_name_of_method(method), remove_duplicates=remove_duplicates
    )


########## Dyanamic version #################################################################
class Tracker:
    """
    Tracks the attribute access right after `start_track` is set to True.

    Add this to __metaclass__ for any class that you need to track attributes for given a
    target method.
    """

    start_track = False

    def __init__(self, *args, **kwargs):
        self.attr_used = []
        self.attrs_to_ignore = []
        super().__init__(*args, **kwargs)

    def __getattribute__(self, item):
        """
        Inspect getter for tracking the attributes accessed.
        """
        if item not in ['on_access']:
            self.on_access(item)
        return super().__getattribute__(item)

    def __setattr__(self, key, value):
        """
        Inspect setter for tracking the attributes accessed.
        """
        if self.start_track:
            self.on_access(key)

        super().__setattr__(key, value)

    def on_access(self, key):
        """
        on attribute access, record attribute if and only if its not from
        core attribute or `attrs_to_ignore` set to class.
        """
        if (
            key in ['start_track', 'attr_used', 'on_access', 'attrs_to_ignore']
            or key in self.attrs_to_ignore
        ):
            return
        if self.start_track:
            self.attr_used.append(key)


@contextmanager
def start_tracking(tracker_instance):
    """
    Ctx manager to gracefully start/stop tracking.
    """
    tracker_instance.start_track = True
    yield tracker_instance
    tracker_instance.start_track = False


def attrs_used_by_method_computation(
    cls_method, init_kwargs=None, method_kwargs=None, remove_duplicates=True
):
    """
    Tracks the access to attributes within an execution.
    """
    if init_kwargs is None:
        init_kwargs = {}
    if method_kwargs is None:
        method_kwargs = {}

    method_class, method_name = cls_and_method_name_of_method(cls_method)
    tracker = type(
        'Tracker',
        (Tracker, method_class),
        dict(method_class.__dict__, **Tracker.__dict__),
    )(**init_kwargs)
    tracker.attrs_to_ignore = [
        func for func in dir(tracker) if callable(getattr(tracker, func))
    ]
    if hasattr(tracker, method_name):
        candidate_method = getattr(tracker, method_name)
        # Now, we want to track attributes.
        with start_tracking(tracker):
            candidate_method(**method_kwargs)

        return _process_duplicates(
            tracker.attr_used, remove_duplicates=remove_duplicates
        )

    else:
        # Error class/obj do not have that method.
        return 1


if __name__ == '__main__':

    def func(obj):
        return obj.a + obj.b

    class A:
        e = 2

        def __init__(self, a=0, b=0, c=1, d=10):
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        def target_func(self, x=3):
            t = func(self)
            tt = self.other_method(t)
            return x * tt / self.e

        target_method = target_func

        def other_method(self, x=1):
            return self.c * x

    from py2json.util import attrs_used_by_method

    print(attrs_used_by_method(A.target_func))

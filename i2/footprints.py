"""Analyzing what attributes of an input object a function actually uses"""


# --------------------------------------------------------------------------------------
# Tools to trace operations on an object.
# See https://github.com/i2mint/i2/issues/56.

import operator
from functools import partial
from i2.multi_object import Pipe
from i2.signatures import Sig, name_of_obj

# TODO: Maybe we should just use an explicit list of dunders instead of this dynamic
#  introspective approach.
dunder_filt = partial(filter, lambda xx: xx.startswith('__'))

_dunders = Pipe(dir, dunder_filt, set)


def module_if_string(x):
    if isinstance(x, str):
        return __import__(x)
    else:
        return x


dunders = Pipe(module_if_string, _dunders)


def dunders_diff(x, y):
    return dunders(x) - dunders(y)


def _method_sig(func, instance_arg='self'):
    """Replace the first argument of a function signature with an instance argument"""
    sig = Sig(func)
    first_param = sig.names[0]
    return Sig(func).ch_names(**{first_param: instance_arg})


exclude = {'__class_getitem__'}

# operator dunders not dunders of all modules (as represented by `typing` module)
_operator_dunders = {
    k: _method_sig(getattr(operator, k)) for k in dunders_diff('operator', 'typing')
}
# dict dunders that aren't dunders of all objects (as represented by `object` object)
_dict_dunders = {
    k: _method_sig(getattr(dict, k)) for k in (dunders_diff(dict, object) - exclude)
}
_rops = (
    set(
        '__radd__, __rsub__, __rmul__, __rdiv__, __rtruediv__, __rfloordiv__, __rmod__, '
        '__rdivmod__, __rpow__, __rlshift__, __rrshift__, __rand__, __rxor__, '
        '__ror__'.split()
    )
    - _dict_dunders.keys()
    - _operator_dunders.keys()
)
_rops = {k: Sig(lambda self, other: None) for k in _rops}
_dflt_methods = dict(_operator_dunders, **_dict_dunders, **_rops)


def trace_class_decorator(cls):
    def create_trace_method(name, signature=None):
        def method(self, *args):
            self.trace.append((name, *args))
            return self

        if signature is not None:
            method.__signature__ = signature


# TODO: Handle *args and **kwargs
def _dflt_method_factory(name, signature=None):
    """A factory for methods that trace the operations that are performed on an.
    The methods made here are specifically meant to be operator methods that have only
    positional arguments.
    """

    def method(self, *args):
        self.trace.append((name, *args))
        return self

    method.__name__ = name

    if signature is not None:
        method.__signature__ = signature

    return method


# TODO: Add method factory argument
def trace_class_decorator(
    cls,
    names_and_sigs=tuple(_dflt_methods.items()),
    method_factory=_dflt_method_factory,
):
    """A decorator that adds methods to a class that trace the operations that are
    performed on an instance of that class.
    """
    for name, sig in dict(names_and_sigs).items():
        setattr(cls, name, method_factory(name, sig))

    return cls


@trace_class_decorator
class MethodTrace:
    """A class that can be used to trace the methods that are called on it.

    See: https://github.com/i2mint/i2/issues/56 for more details.

    >>> t = MethodTrace()
    >>> ((t + 3) - 2) * 5 / 10  # doctest: +ELLIPSIS
    <MethodTrace with .trace = ('__add__', 3), ... ('__truediv__', 10)>
    >>> assert t.trace == [
    ...     ('__add__', 3), ('__sub__', 2), ('__mul__', 5), ('__truediv__', 10)
    ... ]
    >>>
    >>>
    >>> w = t[42]
    >>> t[42] = 'mol'  # an operation with two arguments
    >>> # ... and now an operation with no arguments:
    >>> ~t  # doctest: +ELLIPSIS
    <MethodTrace with .trace = ... ('__setitem__', 42, 'mol'), ('__invert__',)>
    >>>
    >>> assert t.trace == [
    ... ('__add__', 3), ('__sub__', 2), ('__mul__', 5), ('__truediv__', 10),
    ... ('__getitem__', 42), ('__setitem__', 42, 'mol'), ('__invert__',)
    ... ]
    >>>

    """

    def __init__(self):
        self.trace = []

    def __repr__(self):
        trace_str = ', '.join(map(lambda x: f'{x}', self.trace))
        return f'<{type(self).__name__} with .trace = {trace_str}>'

    # TODO: The following is a means to be able to trace all non-dunder methods.
    #  Not sure if we want this as a default, or an option.
    #  Make this work (works, but recursion error when unpickling.
    # def __getattr__(self, operation):
    #     def traced_operation(*args):
    #         self.trace.append((operation, *args))
    #         return self
    #
    #     return traced_operation


# --------------------------------------------------------------------------------------
# Tools to statically

import inspect


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
    if isinstance(method, property):
        return get_class_that_defined_method(method.fget), name_of_obj(method.fget)
    return get_class_that_defined_method(method), name_of_obj(method)


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
    if isinstance(method, property):
        return get_class_that_defined_method(method.fget), method.fget.__name__
    return get_class_that_defined_method(method), method.__name__


# --------------------------------------------------------------------------------------
# Static footprints

import ast
import dis
from functools import reduce
from importlib import import_module
import os
from collections import namedtuple
from inspect import getsource, getsourcefile

Import = namedtuple('Import', ['module', 'name', 'alias'])


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
    """Getting imports for an object (usually, module)"""
    root = _get_ast_root_from(o)
    yield from _get_imports_from_ast_root(root, recursive)


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
    for ix, instr in enumerate(instrs):
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


# TODO: Generalize attrs_used_by_method to attrs_used_by_func.


def _attrs_used_by_method(cls, method_name, remove_duplicates=True):
    """
    Util for attrs_used_by_method. Same output as attrs_used_by_method, but intput is (cls, method_name)
    """
    f = open(getsourcefile(cls), 'r')
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


# --------------------------------------------------------------------------------------
# Newer static footprints
# TODO: Merge with older attrs_used_by_method one above

import inspect
from functools import cached_property
from i2.signatures import Sig, resolve_function


def get_source(obj: object) -> str:
    """Get source string of a python object"""
    return inspect.getsource(resolve_function(obj))


# TODO: Break into two functions (one doing the work of the loop for a single method)
def object_dependencies(obj, *, get_source=get_source):
    import ast
    import inspect
    import textwrap
    from functools import cached_property

    dependency_dict = {}

    # Get the list of methods and properties
    if inspect.isclass(obj):
        members = inspect.getmembers(obj)
    elif hasattr(obj, '__class__'):
        members = inspect.getmembers(obj.__class__)
    else:
        return 'Invalid input. Please provide a class or an instance.'

    # Analyze each method or property
    for name, method in members:
        try:
            source_code = get_source(method)
        except TypeError:
            continue  # Skip if getsource fails

        source_code = textwrap.dedent(source_code)  # Remove leading indentation
        tree = ast.parse(source_code)

        first_arg = Sig(resolve_function(method)).names[0]  # Get first argument name

        called_methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.value.id == first_arg:
                        called_methods.append(node.func.attr)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id == first_arg:
                        called_methods.append(node.attr)

        dependency_dict[name] = set(set(called_methods))  # Remove duplicates

    return dependency_dict


# --------------------------------------------------------------------------------------
# Dynamic footprints

from contextlib import contextmanager


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
        # Now, we want to track attributes.
        with start_tracking(tracker):
            if isinstance(cls_method, property):
                candidate_method = cls_method.fget
                candidate_method(tracker)
            else:
                candidate_method = getattr(tracker, method_name)
                candidate_method(**method_kwargs)

        return _process_duplicates(
            tracker.attr_used, remove_duplicates=remove_duplicates
        )

    else:
        # Error class/obj do not have that method.
        return 1


# def test_attrs_used_by_method():
#     def func(obj):
#         return obj.a + obj.b
#
#     class A:
#         e = 2
#
#         def __init__(self, a=0, b=0, c=1, d=10):
#             self.a = a
#             self.b = b
#             self.c = c
#             self.d = d
#
#         def target_func(self, x=3):
#             t = func(self)
#             tt = self.other_method(t)
#             return x * tt / self.e
#
#         target_method = target_func
#
#         def other_method(self, x=1):
#             return self.c * x
#
#     from i2.footprints import attrs_used_by_method
#
#     print(attrs_used_by_method(A.target_func))

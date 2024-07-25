"""Analyzing what attributes of an input object a function actually uses"""

# --------------------------------------------------------------------------------------
# Tools to trace operations on an object.
# See https://github.com/i2mint/i2/issues/56.

import operator
from functools import partial, cached_property
import ast
from textwrap import dedent
from typing import List, Callable, Literal, Union, Container

from i2.util import ConditionalExceptionCatcher
from i2.multi_object import Pipe
from i2.signatures import Sig, name_of_obj, is_signature_error

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
        return f'<{name_of_obj(type(self))} with .trace = {trace_str}>'

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
            if cls.__dict__.get(name_of_obj(method)) is method:
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
            if cls.__dict__.get(name_of_obj(method)) is method:
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
    if isinstance(fn, cached_property):
        fn = fn.func
    bytecode = dis.Bytecode(fn)
    instrs = list(reversed([instr for instr in bytecode]))
    for ix, instr in enumerate(instrs):
        if instr.opname == 'LOAD_METHOD' or instr.opname == 'LOAD_GLOBAL':
            funcs.append(instr.argval)
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


class _DefinitionFinder(ast.NodeVisitor):
    def __init__(self, object_name):
        self.object_name = object_name
        self.definition_node = None

    # def visit_Assign(self, node):
    #     # Check if the target of assignment matches the object name
    #     for target in node.targets:
    #         if isinstance(target, ast.Name) and target.id == self.object_name:
    #             self.definition_node = node
    #     self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Check if the function name matches the object name
        if node.name == self.object_name:
            self.definition_node = node
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Check if the class name matches the object name
        if node.name == self.object_name:
            self.definition_node = node
        self.generic_visit(node)


def _get_definition_node(object_name, src_code):
    tree = ast.parse(dedent(src_code))
    finder = _DefinitionFinder(object_name)
    finder.visit(tree)
    return finder.definition_node


def _get_source_segment_from_node(node, src_code):
    """
    Extracts the source code segment for a given AST node from the source code string.
    """
    if not node:
        return None

    # Extract start and end line numbers from the node
    start_line = node.lineno
    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line

    # Split the source code into lines
    lines = src_code.splitlines()

    # Extract the lines corresponding to the node (adjusting for 0-based indexing)
    segment_lines = lines[start_line - 1 : end_line]

    # Join the lines back into a single string
    return '\n'.join(segment_lines)


def _get_definition_source(object_name: str, src_code: str):
    """
    Returns the source code string for the definition of the specified object within
    the given source code.

    >>> src_code = '''
    ... class MyClass:
    ...     x = 1
    ...     def my_method(self):
    ...         return self.x + 1
    ...
    ... def my_function():
    ...     pass
    ...
    ... a = 10
    ... '''
    >>> object_name = "MyClass"
    >>> assert _get_definition_source('MyClass', src_code).strip() == (
    ... '''
    ... class MyClass:
    ...     x = 1
    ...     def my_method(self):
    ...         return self.x + 1
    ... '''.strip()
    ... )
    """
    object_name, *more_names = object_name.split('.')

    node = _get_definition_node(object_name, src_code)
    if node is None:
        raise ValueError(f'Could not find definition for object {object_name}')
    src = _get_source_segment_from_node(node, src_code)
    src = dedent(src)

    if not more_names:
        return src
    else:
        return _get_definition_source('.'.join(more_names), src)


def _unwrap_object(o):
    if isinstance(o, cached_property):
        return o.func
    elif isinstance(o, property):
        return o.fget
    return o


from operator import attrgetter

qualname_of_obj = partial(name_of_obj, base_name_of_obj=attrgetter('__qualname__'))


def _get_source(o, src_code=None) -> str:
    if src_code is None:
        if isinstance(o, str):
            if os.path.isfile(o):
                with open(o) as f:
                    source_str = f.read()
            else:
                source_str = o  # consider the string to be the source code
            return source_str
        else:  # it's an object
            o = _unwrap_object(o)
            # get's it's source code
            try:
                with open(getsourcefile(o), 'r') as f:
                    source_str = f.read()
                return source_str
                # TODO: (somehow, simply return getsource(o) doesn't lead to passing tests)
            except OSError as e:
                try:
                    return getsource(o)
                except Exception as e:

                    raise ValueError(
                        f'Could not get source code for object {o}. '
                        '(This can happen if your object was defined in jupyter notebooks, '
                        'for example.)'
                        'Please provide the source code explicitly via the '
                        'src_code argument. '
                        'For example, if you are in a jupyter notebook or ipython, '
                        'you can specify `src_code=In[cell_index]` where `cell_index` '
                        'is the index of the cell where the object was defined.'
                    )
    else:
        # The reason for this case is that the source code of an object is not always
        # accessible via normal means (for example, in jupyter notebooks) so we need to pass it in.
        if isinstance(o, str):
            object_name = o
        else:
            object_name = qualname_of_obj(o)
        source_str = _get_definition_source(object_name, src_code)
        return source_str


def ensure_ast(o, src_code=None) -> ast.AST:
    """
    Casts input object `o` to a AST node.

    If the input is an AST node, it is returned as is.
    If the input is a filepath, the file is read and parsed as source code.
    If the input is a string, it is parsed as source code.
    If the input is an object, the source code is extracted and parsed.

    Let's get an AST for the ensure_ast function itself:

    >>> assert isinstance(ensure_ast(ensure_ast), ast.AST)

    Note that sometimes the source code of an object cannot be accessed via normal
    means (for example, in REPLs like jupyter notebooks) so we need to pass it in.

    >>> src_code = '''
    ... class MyClass:
    ...     x = 1
    ...     def my_method(self):
    ...         return self.x + 1
    ... a = 10
    ... '''
    >>> assert isinstance(ensure_ast('MyClass', src_code), ast.AST)

    """

    if isinstance(o, ast.AST):
        return o
    else:
        source_str = _get_source(o, src_code)
        return ast.parse(dedent(source_str))


class AttributeVisitor(ast.NodeVisitor):
    def __init__(self, object_name):
        self.object_name = object_name
        self.attributes = set()

    def visit_Attribute(self, node):
        # Check if the attribute access is for the target object
        if isinstance(node.value, ast.Name) and node.value.id == self.object_name:
            self.attributes.add(node.attr)
        # Continue traversing to find more attributes
        self.generic_visit(node)


# TODO: Clean this all up, it's horrible!
# TODO: Write teests for accessed_attributes using i2.tests.footprints_test -> A, B
def accessed_attributes(func, object_name=None):
    """
    Extracts the attributes accessed by a function or method.

    (This is a simpler, but narrow, version of `attrs_used_by_method`).

    >>> def func(a, b, c):
    ...     return a + b.bar + c
    ...
    >>> # Commenting out the testing code, as execution is not performed in the PCI
    >>> def foo(self):
    ...     a = 2
    ...     self.method(x)
    ...     y = self.prop
    ...     return a + func(x, self, y)
    ...
    >>> assert accessed_attributes(foo, 'self') == {'method', 'prop'}

    """
    if object_name is None:
        object_name = next(iter(Sig(func)), None)
        if object_name is None:
            raise ValueError(
                'Could not determine the object name. Please provide it explicitly.'
            )

    # Convert the function source to an AST
    func_name = qualname_of_obj(func)
    src = _get_definition_source(func_name, _get_source(func))
    node = ensure_ast(src)
    # func_src = ast.parse(inspect.getsource(func))
    # Initialize the visitor with the target object name
    visitor = AttributeVisitor(object_name)
    # Visit the AST to find accessed attributes
    visitor.visit(node)
    # Return the set of accessed attributes
    return visitor.attributes


def _is_method_like(
    name,
    obj,
    *,
    no_dunders=True,
    include=('__post_init__',),
    include_types=(Callable, property, cached_property),
    exclude_types=(staticmethod,),
):
    if name in include:
        return True
    elif no_dunders and name.startswith('__') and name.endswith('__'):
        return False
    return isinstance(obj, include_types) and not isinstance(obj, exclude_types)


def init_argument_names(cls, *, no_error_action=None) -> List[str]:
    """
    Get the list of argument names

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class DataClass:
    ...     x: str
    ...     y: float = 2
    ...     z = 3
    ...
    >>> init_argument_names(DataClass)
    ['x', 'y']

    Note: Some builtin types don't have signatures, so we get:

    ```
    ValueError: no signature found for builtin type ...
    ```

    By default, we handle this by returning an empty list, but a callable
    no_error_action will call that function and return its result.
    Anything else will result in raising the error.

    """
    try:
        return Sig(cls).names
    except ValueError as e:
        # Some builtin types don't have signatures, so we get:
        #   ValueError: no signature found for builtin type ...
        # By default, we handle this by returning an empty list.
        if no_error_action is None:
            return []
        elif callable(no_error_action):
            return no_error_action()
        else:
            raise e


ExcludeNames = Union[Container, Callable]


def _get_class_attributes(
    cls: type,
    filt=_is_method_like,
    *,
    exclude_names: ExcludeNames = init_argument_names,
):
    if isinstance(exclude_names, Callable):
        exclude_names = exclude_names(cls)

    for name, obj in cls.__dict__.items():
        if filt(name, obj) and name not in exclude_names:
            yield obj


skip_signature_errors = ConditionalExceptionCatcher(ValueError, is_signature_error)


def attribute_dependencies(
    cls: type,
    filt=_is_method_like,
    *,
    name_of_obj=name_of_obj,
    exclude_names: ExcludeNames = init_argument_names,
):
    """
    Extracts (method_name, accessed_attributes) pairs for a class or instance thereof.

    :param cls: The class or instance to analyze
    :param filt: A function that filters the attributes to consider
    :param name_of_obj: A function that returns the name of an object
    :param exclude_names: A list of names to exclude from the analysis or a function that
        returns such a list given the class
    :return: A generator of (method_name, accessed_attributes) pairs

    """
    for func in _get_class_attributes(cls, filt=filt, exclude_names=exclude_names):
        with skip_signature_errors:
            yield name_of_obj(func), accessed_attributes(func)


def _attrs_used_by_method(cls, method_name, *, src_code=None):
    """
    Util for attrs_used_by_method. Same output as attrs_used_by_method, but intput is (cls, method_name)
    """
    root = ensure_ast(cls, src_code)
    funcs = list_func_calls(getattr(cls, method_name))
    attrs = []
    for func in funcs:
        attrs = attrs + attr_list(root, func)

    return _process_duplicates(
        attrs + attr_list(root, method_name), remove_duplicates=True
    )


def attrs_used_by_method(method, *, src_code=None):
    """
    Extracts a list of cls attributes which are used by a method or method_name function

    Note: The function tries to analyzed the source code deeply, gathering indirect
    references to the instance attributes. As a result, it is not very robust.
    You may want to check out the simpler (but narrow) function: `accessed_attributes`.

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
        *cls_and_method_name_of_method(method), src_code=src_code
    )


# --------------------------------------------------------------------------------------
# Newer static footprints
# TODO: Merge with older attrs_used_by_method one above

import inspect
from i2.signatures import Sig, resolve_function


def get_source(obj: object) -> str:
    """Get source string of a python object"""
    return inspect.getsource(resolve_function(obj))


# TODO: Break into two functions (one doing the work of the loop for a single method)
# TODO: Routing pattern. Extract conditional logic to make it parametrizable
def object_dependencies(obj, *, get_source=get_source):
    import ast
    import inspect
    import textwrap
    from functools import cached_property

    dependency_dict = {}

    if inspect.isclass(obj):
        members = inspect.getmembers(obj)
    elif hasattr(obj, '__class__'):
        members = inspect.getmembers(obj.__class__)
    else:
        return 'Invalid input. Please provide a class or an instance.'

    for name, method in members:
        try:
            source_code = get_source(method)
            first_arg = Sig(resolve_function(method)).names[0]
        except TypeError:
            continue

        source_code = textwrap.dedent(source_code)
        tree = ast.parse(source_code)

        called_methods = []
        assigned_attributes = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if getattr(target.value, 'id', None) == first_arg:
                            assigned_attributes.add(target.attr)

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if getattr(node.func.value, 'id', None) == first_arg:
                        called_methods.append(node.func.attr)

            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id == first_arg:
                        called_methods.append(node.attr)

        # Remove attributes that are only assigned to
        called_methods = set(called_methods) - assigned_attributes

        dependency_dict[name] = called_methods

    return dependency_dict


def _dict_to_graph(
    graph: dict,
    *,
    edge_connector: str,
    graph_template: str,
    display: Callable = None,
    indent: str = '    ',
    prefix: str = '',
    suffix: str = '',
) -> str:
    """Helper for dict_to_graph"""
    if not display:
        display = lambda x: x
    elif display is True:
        display = print
    assert callable(display), f'display should be callable, boolean or None: {display=}'
    lines = []
    for from_node, to_nodes in graph.items():
        for to_node in to_nodes:
            lines.append(f'{indent}{from_node}{edge_connector}{to_node};')
    graph_code = '\n'.join(lines)
    return display(graph_template.format(code=f'{prefix}{graph_code}{suffix}'))


def dict_to_graph(
    graph: dict,
    from_key_to_values: bool = True,
    *,
    kind: Literal['graphviz', 'mermaid'] = 'graphviz',
    indent: str = '    ',
    prefix: str = '',
    suffix: str = '',
    display: Union[bool, Callable] = False,
) -> str:
    """A function to convert a dictionary to a graphviz string.

    You privide a graph in the form of a ``{from_node: to_nodes, ...`` or
    ``{to_node: from_nodes, ...``` dictionary, and will get a graphviz string.
    You can use this to visualize a graph (e.g. a dependency graph) in a jupyter notebook.

    :param graph: The graph, in the form of a to convert to graphviz.
    :param from_key_to_values: Whether the keys of the graph are from nodes or to nodes.
    :param kind: The kind of graphviz string to return. Either "graphviz" or "mermaid".
    :param indent: The indent to use for the graphviz string.
    :param graphviz_template: The template to use for the graphviz string.
    :param display: Whether to display the graphviz string as a graph in a jupyter notebook. Requires graphviz.
    :return: The graphviz string.

    Example (but bear in mind the order of the nodes in graphviz_str may be different):

    >>> graph_dict = {
    ...     "A": ["B", "C"],
    ...     "B": ["D"],  # note that "D" is not mentioned as a key
    ...     "C": ["D", "E", "F"],
    ...     "E": [],
    ... }
    >>> # Keys are from nodes
    >>> graphviz_str = dict_to_graph(graph_dict)
    >>> print(graphviz_str)  # doctest: +SKIP
    digraph G {
        "A" -> "B";
        "A" -> "C";
        "B" -> "D";
        "C" -> "D";
        "C" -> "E";
        "C" -> "F";
    }

    >>> # Keys are to nodes
    >>> graphviz_str = dict_to_graph(graph_dict, from_key_to_values=False)
    >>> print(graphviz_str)  # doctest: +SKIP
    digraph G {
        "B" -> "A";
        "C" -> "A";
        "D" -> "B";
        "D" -> "C";
        "E" -> "C";
    }

    The default ``kind`` is graphviz, but you can also use mermaid:

    >>> graphviz_str = dict_to_graph(graph_dict, kind="mermaid")
    >>> print(graphviz_str)  # doctest: +SKIP
    graph TD
        A --> B;
        A --> C;
        B --> D;
        C --> D;
        C --> E;
        C --> F;

    """
    # TODO: Could make these specs open-closed (routing pattern)
    if kind == 'graphviz':
        graph_template = f'digraph G {{{{\n{{code}}\n}}}}'
        edge_connector = ' -> '
        if display is True:
            try:
                from graphviz import Source

                display = Source
            except ImportError:
                ImportError('You need to `pip install graphviz` to display the graph')

    elif kind == 'mermaid':
        graph_template = f'graph TD\n{{code}}\n'
        edge_connector = ' --> '
        if display is True:
            try:
                from kroki import diagram_image  # pip install kroki

                display = lambda x: diagram_image(x, diagram_type='mermaid')
            except ImportError:
                ImportError('You need to `pip install kroki` to display the graph')

    else:
        raise ValueError(f'Invalid kind specified: {kind}')

    if not from_key_to_values:
        graph = {
            to_node: [from_node for from_node in graph if to_node in graph[from_node]]
            for to_node in set(val for vals in graph.values() for val in vals)
        }

    return _dict_to_graph(
        graph,
        edge_connector=edge_connector,
        graph_template=graph_template,
        indent=indent,
        prefix=prefix,
        suffix=suffix,
        display=display,
    )


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

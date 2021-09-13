"""
Tools to provide meta-interfaces of python objects.
See also:
    py2misc/py2store/tree_store.py
"""
import inspect
from inspect import signature, Parameter
from typing import Mapping

from i2.util import lazyprop, FrozenDict, get_function_body

inspect_empty = Parameter.empty


class NotFoundType:
    def __bool__(self):
        return False

    def __repr__(self):
        return 'NotFound'


# Note, if the "empty" classes the are used to check if an object is empty are not singletons, could lead to bugs
# TODO: Look into metaprogramming tricks for this.
not_found = NotFoundType()
no_name = type('NoName', (NotFoundType,), {})()
inspect_is_empty = inspect._empty


def is_not_empty(obj):
    if obj is inspect_is_empty or isinstance(obj, NotFoundType):
        return False
    else:
        return True


def _is_property(attr_name, attr_val):
    return not attr_name.startswith('_') and isinstance(attr_val, (property, lazyprop))


def _property_names_of(obj):
    if not isinstance(obj, type):
        obj = type(obj)
    for attr_name in dir(obj):
        attr_val = getattr(obj, attr_name)
        if _is_property(attr_name, attr_val):
            yield attr_name


def name_of_obj(o):
    """
    Tries to find the (or "a") name for an object, even if `__name__` doesn't exist.

    >>> name_of_obj(map)
    'map'
    >>> name_of_obj([1, 2, 3])
    'list'
    >>> name_of_obj(print)
    'print'
    >>> name_of_obj(lambda x: x)
    '<lambda>'
    >>> from functools import partial
    >>> name_of_obj(partial(print, sep=','))
    'print'
    """
    if hasattr(o, '__name__'):
        return o.__name__
    elif hasattr(o, '__class__'):
        name = name_of_obj(o.__class__)
        if name == 'partial':
            if hasattr(o, 'func'):
                return name_of_obj(o.func)
        return name
    else:
        return no_name


class AttrFromKey:
    def __init__(self, d):
        self._d = d

    def __getattr__(self, k):
        return self._d.__getitem__(k)

    __getitem__ = __getattr__


class KeyFromAttr:
    def __init__(self, d):
        self._d = d

    def __getitem__(self, k):
        return getattr(self._d, k)

    __getattr__ = __getitem__


class _ParameterMintLazy:  # Note: Experimental
    _attrs = ['name', 'default', 'annotation']

    #     find = DelegatedAttribute('_params', '__getattribute__')
    #     __getattribute__ = DelegatedAttribute('_params', '__getattribute__')
    def __init__(self, param, _attrs=None):
        self._param = param
        if _attrs is not None:
            # usually used to get full attrs: ['name', 'kind', 'default', 'annotation']
            self._attrs = _attrs

    def items(self):
        for k in self._attrs:
            v = getattr(self._param, k)
            if v is not inspect_empty:
                yield k, v

    def __getitem__(self, k):
        return getattr(self._param, k)

    def __contains__(self, k):
        try:
            _ = self[k]
            return True
        except Exception:
            return False

    def __getattr__(self, k):  # TODO: Use descriptor to make it faster
        return getattr(self._param, k)

    def __getstate__(self):
        return {k: v for k, v in self.items()}


class ParameterMint:
    _attrs = ['name', 'kind', 'default', 'annotation']

    def __init__(self, param, position=None):
        if hasattr(param, '__getitem__'):
            param = AttrFromKey(param)

        for attr in self._attrs:
            try:
                val = getattr(param, attr, inspect_empty)
            except KeyError:
                val = inspect_empty
            setattr(self, attr, val)
        if position is not None:
            self.position = position

    def __getitem__(self, k):
        return getattr(self, k)

    def __contains__(self, k):
        try:
            _ = self[k]
            return True
        except Exception:
            return False

    def items(self):
        for k in self._attrs:
            v = getattr(self, k)
            if v is not inspect_empty:
                yield k, v
        if hasattr(self, 'position'):
            yield 'position', self.position

    def __getstate__(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        d = dict()
        for k, v in self.items():
            if k == 'kind':
                v = v.name
            d[k] = v
        return str(d)


dflt_params = FrozenDict({})


class ParametersMint(Mapping):
    """
    Get mint of the parameters of a callable.

    >>> import inspect
    >>> from pprint import pprint
    >>>
    >>> def g(a, b: 'some_type', c=1, d: int = 1) -> float:
    ...     return a * b * c * d
    >>> mint = ParametersMint(inspect.signature(g).parameters)
    >>> # mint is a mapping (like a read-only dict), so...
    >>> list(mint)
    ['a', 'b', 'c', 'd']
    >>>
    >>> for arg_spec in mint.values():
    ...     print(arg_spec)
    {'name': 'a', 'kind': 'POSITIONAL_OR_KEYWORD', 'position': 0}
    {'name': 'b', 'kind': 'POSITIONAL_OR_KEYWORD', 'annotation': 'some_type', 'position': 1}
    {'name': 'c', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 1, 'position': 2}
    {'name': 'd', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 1, 'annotation': <class 'int'>, 'position': 3}
    >>> t = list(mint.items())
    >>> t[0]
    ('a', {'name': 'a', 'kind': 'POSITIONAL_OR_KEYWORD', 'position': 0})
    >>> t[1]
    ('b', {'name': 'b', 'kind': 'POSITIONAL_OR_KEYWORD', 'annotation': 'some_type', 'position': 1})
    >>>
    >>> mint = ParametersMint(inspect.signature(g).parameters)
    >>> pprint(dict(mint))
    {'a': {'name': 'a', 'kind': 'POSITIONAL_OR_KEYWORD', 'position': 0},
     'b': {'name': 'b', 'kind': 'POSITIONAL_OR_KEYWORD', 'annotation': 'some_type', 'position': 1},
     'c': {'name': 'c', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 1, 'position': 2},
     'd': {'name': 'd', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 1, 'annotation': <class 'int'>, 'position': 3}}

    >>>
    >>> # and now, some cannibalistic fun...
    >>> pprint(dict(ParametersMint(inspect.signature(pprint).parameters)))
    {'compact': {'name': 'compact', 'kind': 'KEYWORD_ONLY', 'default': False, 'position': 5},
     'depth': {'name': 'depth', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': None, 'position': 4},
     'indent': {'name': 'indent', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 1, 'position': 2},
     'object': {'name': 'object', 'kind': 'POSITIONAL_OR_KEYWORD', 'position': 0},
     'sort_dicts': {'name': 'sort_dicts', 'kind': 'KEYWORD_ONLY', 'default': True, 'position': 6},
     'stream': {'name': 'stream', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': None, 'position': 1},
     'width': {'name': 'width', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 80, 'position': 3}}
    >>> pprint(dict(ParametersMint(inspect.signature(ParametersMint).parameters)))
    {'args': {'name': 'args', 'kind': 'VAR_POSITIONAL', 'position': 0},
     'kwds': {'name': 'kwds', 'kind': 'VAR_KEYWORD', 'position': 1}}
    >>> pprint(dict(ParametersMint(inspect.signature(ParametersMint.__init__).parameters)))
    {'params': {'name': 'params', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': FrozenDict({}), 'position': 1},
     'self': {'name': 'self', 'kind': 'POSITIONAL_OR_KEYWORD', 'position': 0}}
    >>> pprint(dict(ParametersMint(inspect.signature(ParametersMint.__new__).parameters)))
    {'args': {'name': 'args', 'kind': 'VAR_POSITIONAL', 'position': 1},
     'cls': {'name': 'cls', 'kind': 'POSITIONAL_OR_KEYWORD', 'position': 0},
     'kwds': {'name': 'kwds', 'kind': 'VAR_KEYWORD', 'position': 2}}
    """

    def __init__(self, params=dflt_params):
        self._param_names = list()
        for i, (k, v) in enumerate(params.items()):
            self._param_names.append(k)
            setattr(self, k, ParameterMint(v, position=i))

    def __getitem__(self, k):
        return getattr(self, k)

    def __iter__(self):
        yield from self._param_names

    def items(self):
        for k in self._param_names:
            yield k, getattr(self, k)

    def __len__(self):
        c = 0
        for k in self.__iter__():
            c += 1
        return c

    def __getstate__(self):
        return {k: v.__getstate__() for k, v in self.items()}

    def __setstate__(self, state):
        return type(self)(state)

    def to_dict(self):
        return self.__getstate__()

    def __repr__(self):
        arg_repr = str({k: v.__repr__() for k, v in self.items()})
        return f'{self.__class__.__name__}({arg_repr})'


class Mint(Mapping):
    """
    Get a Mint object of a python object.
    A Mint will provide parameters that provide (meta-)information about the interface of the python object.

    >>> from pprint import pprint
    >>> # Mint of a function
    >>> def f(my_arg: int = 7) -> int:
    ...     return my_arg + 10
    >>> mint = Mint(f)
    >>> mint.obj_name, mint.type_name, mint.module_name, mint.obj_name
    ('f', 'function', 'i2.base', 'f')
    >>> # Mint of a module
    >>> import os as myos
    >>> mint = Mint(myos)
    >>> mint.obj_name, mint.type_name, mint.module_name, mint.obj_name
    ('os', 'module', 'os', 'os')
    >>> assert set(list(mint)) == {'module_name', 'module', 'type_name', 'obj_name'}
    >>> # Mint of a variable
    >>> v = 10
    >>> mint = Mint(v)
    >>> mint.obj_name, mint.type_name, mint.module_name, mint.obj_name
    (NotFound, 'int', NotFound, NotFound)
    >>> assert set(list(mint)) == {'type_name'}  # see that there's only one non-null attr!
    """

    def __init__(self, obj, attrs=None):
        self._obj = obj
        if attrs is None:
            attrs = frozenset(_property_names_of(type(self)))
        else:
            self_type = type(self)
            for attr in attrs:
                if not _is_property(attr, getattr(self_type, attr, None)):
                    raise AttributeError(
                        f"{self.__class__} doesn't have attribute: {attr}"
                    )
        self._attrs = attrs

    def __len__(self):
        return self._length

    def __iter__(self):
        yield from self._non_empty_attrs

    def __getitem__(self, k):
        if k in self._non_empty_attrs:
            return getattr(self, k)
        else:
            raise KeyError('No such key: {k}')

    def items(self):
        for attr in self._non_empty_attrs:
            yield attr, getattr(self, attr)

    @lazyprop
    def _non_empty_attrs(self):
        non_empty_attrs = list()
        for attr in self._attrs:
            attr_val = getattr(self, attr)
            if is_not_empty(attr_val):
                non_empty_attrs.append(attr)
        return non_empty_attrs

    @lazyprop
    def _length(self):
        return len(self._non_empty_attrs)

    @lazyprop
    def obj_name(self):
        return getattr(self._obj, '__name__', not_found)

    @lazyprop
    def type_name(self):
        return getattr(type(self._obj), '__name__', not_found)

    @lazyprop
    def module(self):
        return inspect.getmodule(self._obj) or not_found

    @lazyprop
    def module_name(self):
        return getattr(self.module, '__name__', not_found)


class MintOfCallableMixin:
    @lazyprop
    def _signature(self):
        """Here's some doc"""
        return signature(self._obj)

    @lazyprop
    def parameters(self):
        return ParametersMint(self._signature.parameters)

    @lazyprop
    def return_annotation(self):
        return self._signature.return_annotation or not_found

    @lazyprop
    def doc_string(self):
        return inspect.getdoc(self._obj) or not_found

    @lazyprop
    def comments_preceeding_def(self):
        return inspect.getcomments(self._obj) or not_found

    @lazyprop
    def default_of(self):
        d = dict()
        for k, v in self.parameters.items():
            if 'default' in v:
                d[k] = v['default']
        return d

    @lazyprop
    def annotation_of(self):
        d = dict()
        for k, v in self.parameters.items():
            if 'annotation' in v:
                d[k] = v['annotation']
        return d


class MintOfDocMixin:
    @lazyprop
    def _parsed_doc(self):
        return 'Not yet implemented (correctly)'
        from i2.scrap import parse_mint_doc

        # return parse_mint_doc(self.doc_string)


class MintOfCallable(Mint, MintOfCallableMixin, MintOfDocMixin):
    """
    Get a Mint object of a python object.
    A Mint will provide parameters that provide (meta-)information about the interface of the python object.

    >>> from pprint import pprint
    >>> def f(my_arg: int = 7) -> int:
    ...     return my_arg + 10
    >>> f.__doc__ = 'some documentation'
    >>>
    >>> mint = MintOfCallable(f)
    >>> mint.obj_name
    'f'
    >>> mint.type_name
    'function'
    >>> mint.module_name
    'i2.base'
    >>> mint.parameters.my_arg
    {'name': 'my_arg', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 7, 'annotation': <class 'int'>, 'position': 0}
    >>> mint.doc_string
    'some documentation'
    >>> mint.return_annotation
    <class 'int'>
    >>> def g(a, b: 'some_string_id_of_a_custom_type', c=1, d: int = 1) -> float:
    ...     return a * b * c * d
    >>> pprint(dict(MintOfCallable(g).parameters))
    {'a': {'name': 'a', 'kind': 'POSITIONAL_OR_KEYWORD', 'position': 0},
     'b': {'name': 'b', 'kind': 'POSITIONAL_OR_KEYWORD', 'annotation': 'some_string_id_of_a_custom_type', 'position': 1},
     'c': {'name': 'c', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 1, 'position': 2},
     'd': {'name': 'd', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 1, 'annotation': <class 'int'>, 'position': 3}}
    """

    pass


# class FunctionBuilderMint(Mint):
#     name = lazyprop(Mint.obj_name)
#     module = lazyprop(Mint.module_name)
#     module.__doc__ = 'Name of the module from which this function was imported.'
#     doc = lazyprop(MintOfCallableMixin.doc_string)
#
#     @lazyprop
#     def _signature(self):
#         """ Here's some doc """
#         return signature(self._obj)
#
#     @lazyprop
#     def body(self):
#         """String version of the code representing the body
#         of the function. Defaults to ``'pass'``, which will result
#         in a function which does nothing and returns ``None``."""
#         return get_function_body(self._obj)
#
#     @lazyprop
#     def args(self):
#         return [p['name'] for p in self._signature.parameters]
#
#     @lazyprop
#     def varargs(self):
#         return [p['name'] for p in self._signature.parameters]
#
# """    name (str): Name of the function.
#     doc (str): `Docstring`_ for the function, defaults to empty.
#     module (str): Name of the module from which this function was
#         imported. Defaults to None.
#     body (str): String version of the code representing the body
#         of the function. Defaults to ``'pass'``, which will result
#         in a function which does nothing and returns ``None``.
#     args (list): List of argument names, defaults to empty list,
#         denoting no arguments.
#     varargs (str): Name of the catch-all variable for positional
#         arguments. E.g., "args" if the resultant function is to have
#         ``*args`` in the signature. Defaults to None.
#     varkw (str): Name of the catch-all variable for keyword
#         arguments. E.g., "kwargs" if the resultant function is to have
#         ``**kwargs`` in the signature. Defaults to None.
#     defaults (dict): A mapping of argument names to default values.
#     kwonlyargs (list): Argument names which are only valid as
#         keyword arguments. **Python 3 only.**
#     kwonlydefaults (dict): A mapping, same as normal *defaults*,
#         but only for the *kwonlyargs*. **Python 3 only.**
#     annotations (dict): Mapping of type hints and so
#         forth. **Python 3 only.**
#     filename (str): The filename that will appear in
#         tracebacks. Defaults to "boltons.funcutils.FunctionBuilder".
#     indent (int): Number of spaces with which to indent the
#         function *body*. Values less than 1 will result in an error.
#     dict (dict): Any other attributes which should be added to the
#         functions compiled with this FunctionBuilder."""

"""Misc util objects"""
from operator import attrgetter, itemgetter
import inspect
import re
import itertools
import sys
import functools
import types
from typing import (
    Mapping,
    Callable,
    Any,
    MutableMapping,
    Union,
    Optional,
    Iterable,
    TypeVar,
)

T = TypeVar('T')


class OverwritesForbidden(ValueError):
    """Raise when a user is not allowed to overwrite a mapping's key"""


def is_lambda(func):
    return getattr(func, '__name__', None) == '<lambda>'


# TODO: Fragile. Make more robust.
def lambda_code(lambda_func) -> str:
    """Extract code of expression from lambda function.
    For lambda code-extraction see:
    https://stackoverflow.com/questions/73980648/how-to-transform-a-lambda-function-into-a-pickle-able-function

    """
    func_str = str(inspect.getsourcelines(lambda_func)[0])
    return func_str.strip("['\\n']").split(' = ')[1]


# TODO: Only works with lambdas so either assert function is a lambda on construction
#  or make it work with functions more generally.
class PicklableLambda:
    """
    Wraps a lambda function to make it picklable (through extracting its code)
    Also, provide it with a name, optionally.

    >>> f = lambda x, y=0: x + y
    >>> ff = PicklableLambda(f)
    >>> import pickle
    >>> fff = pickle.loads(pickle.dumps(ff))
    >>> assert fff(2, 3) == ff(2, 3) == f(2, 3)

    For lambda code-extraction see:
    https://stackoverflow.com/questions/73980648/how-to-transform-a-lambda-function-into-a-pickle-able-function

    """

    def __init__(self, func, name=None):
        self.func = func
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = name or getattr(func, '__name__', type(self).__name__)

    def __getstate__(self):
        return lambda_code(self.func), self.__name__

    def __setstate__(self, state):
        func_code, name = state
        func = eval(func_code)  # scary?
        self.__init__(func, name)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f'<{type(self).__name__}({self.__name__})>'


def ensure_identifiers(
    *objs: Iterable[T],
    get_identfiers: Callable[[T], Iterable[str]] = str.split,
    is_identifier: Callable[[str], bool] = str.isidentifier,
):
    """Ensure an iterable of identifiers

    >>> list(ensure_identifiers('these', 'are', 'valid', 'identifiers'))
    ['these', 'are', 'valid', 'identifiers']

    By default, ``ensure_identifiers`` will apply ``str.split`` to each ``obj`` of
    ``objs`` (assumed to be strings!) so that it can extract identifiers from
    space-separated strings of identifiers:

    >>> list(ensure_identifiers('these are valid identifiers'))
    ['these', 'are', 'valid', 'identifiers']

    You can control this functionality through the ``get_identfiers`` argument, for
    example, disallowing such splitting, or enabling the extraction of identifiers
    from other objects than strings.

    >>> list(ensure_identifiers(
    ...     {'this': 0, 'works': 1}, {'too': 2},
    ...     get_identfiers=list
    ... ))
    ['this', 'works', 'too']

    You can also control the ``is_identifier`` validatation function:

    >>> def less_than_6_chars(s): return len(s) < 6
    >>> list(ensure_identifiers('okay', 'too_long', is_identifier=less_than_6_chars))
    Traceback (most recent call last):
      ...
    ValueError: too_long isn't an identifier according toless_than_6_chars

    """
    for obj in objs:
        for identifier in get_identfiers(obj):
            if is_identifier(identifier):
                yield identifier
            else:
                raise ValueError(
                    f"{identifier} isn't an identifier according to"
                    f"{getattr(is_identifier, '__name__', str(is_identifier))}"
                )


def insert_name_based_objects_in_scope(
    *names,
    factory: Callable[[str], Any],
    scope: MutableMapping,
    allow_overwrites: bool = False,
):
    """
    Make several string-parametrized objects and insert them in a scope (e.g. locals()).

    This is useful when to avoid (error-prone) situations where we want the name we
    assign an object to, to be aligned with it's internal name, such as::

        foo = Factory('foo', ...)
        bar = Factory('bar', ...)
        baz = Factory('baz', ...)

    :param names: Identifier (valid python variable name) strings.
        These are used both as arguments of the ``factory`` and as keys for the
        ``scope`` the object the factory makes will be inserted under.
    :param factory: A function that takes a (valid python identifier) string and
        returns an object parametrized by that string.
    :param scope: The ``MutableMapping`` we want to insert the objects in.
    :param allow_overwrites: Whether the objects we create can overwrite existing
        objects the ``scope`` may already have. If we don't allow overwrites and we
        try to write under an existing key, a ``OverwritesForbidden`` error will be
        raised. This also includes the situation where we have some duplicates in
        ``names``.
    :return: None (this function has the side effect of inserting items in ``scope``.

    One of the (controversal) uses of ``insert_name_based_objects_in_scope`` is to be
    able to make several string-parametrized

    >>> from collections import namedtuple
    >>> from functools import partial
    >>>
    >>> factory = partial(namedtuple, field_names='apple banana')
    >>> insert_namedtuples_in_locals = partial(insert_name_based_objects_in_scope,
    ...     factory=factory, scope=locals(), allow_overwrites=True
    ... )
    >>> insert_namedtuples_in_locals('foo bar', 'baz')

    And now ``foo` exists!

    >>> 'foo' in locals()
    True
    >>> foo(1,2)
    foo(apple=1, banana=2)

    And so does ``bar`` and ``baz``:

    >>> bar(3, banana=4)
    bar(apple=3, banana=4)
    >>> baz(apple=3, banana=4)
    baz(apple=3, banana=4)
    """
    for name in ensure_identifiers(*names):
        if name not in scope or allow_overwrites:
            scope[name] = factory(name)
        else:
            raise OverwritesForbidden(
                f'This key already exisited and is not allowed to be overwritten'
            )


class Literal:
    """An object to indicate that the value should be considered literally.

    >>> t = Literal(42)
    >>> t.get_val()
    42
    >>> t()
    42

    """

    def __init__(self, val):
        self.val = val

    def get_val(self):
        """Get the value wrapped by Literal instance.

        One might want to use ``literal.get_val()`` instead ``literal()`` to get the
        value a ``Literal`` is wrapping because ``.get_val`` is more explicit.

        That said, with a bit of hesitation, we allow the ``literal()`` form as well
        since it is useful in situations where we need to use a callback function to
        get a value.
        """
        return self.val

    __call__ = get_val


def dflt_idx_preprocessor(obj, idx):
    if isinstance(idx, str) and str.isdigit(idx):
        idx = int(idx)
    if isinstance(idx, int) or isinstance(obj, Mapping):
        return obj[idx]
    elif hasattr(obj, idx):
        return getattr(obj, idx)
    else:
        raise KeyError(f"Couldn't extract a {idx} from object {obj}")


def path_extractor(tree, path, getter=dflt_idx_preprocessor, *, path_sep='.'):
    """Get items from a tree-structured object from a sequence of tree-traversal indices.

    :param tree: The object you want to extract values from:
        Can be any object you want, as long as the indices listed by path and how to get
        the items indexed are well specified by ``path`` and ``getter``.
    :param path: An iterable of indices that define how to traverse the tree to get
        to desired item(s). If this iterable is a string, the ``path_sep`` argument
        will be used to transform it into a tuple of string indices.
    :param getter: A ``(tree, idx)`` function that specifies how to extract item ``idx``
        from the ``tree`` object.
    :param path_sep: The string separator to use if ``path`` is a string
    :return: The ``tree`` item(s) referenced by ``path``


    >>> tree = {'a': {'b': [0, {'c': [1, 2, 3]}]}}
    >>> path_extractor(tree, path=['a'])
    {'b': [0, {'c': [1, 2, 3]}]}
    >>> path_extractor(tree, path=['a', 'b'])
    [0, {'c': [1, 2, 3]}]
    >>> path_extractor(tree, path=['a', 'b', 1])
    {'c': [1, 2, 3]}
    >>> path_extractor(tree, path=['a', 'b', 1, 'c'])
    [1, 2, 3]
    >>> path_extractor(tree, path=('a', 'b', 1, 'c', 2))
    3

    You could do the same by specifying the path as a dot-separated string.

    >>> path_extractor(tree, 'a.b.1.c.2')
    3

    You can use any separation you want.

    >>> path_extractor(tree, 'a/b/1/c/2', path_sep='/')
    3

    You can also use `*` to indicate that you want to keep all the nodes of a given
    level.

    >>> tree = {'a': [{'b': [1, 10]}, {'b': [2, 20]}, {'b': [3, 30]}]}
    >>> path_extractor(tree, 'a.*.b.1')
    [10, 20, 30]

    A generalization of `*` is to specify a callable which will be intepreted as
    a filter function.

    >>> tree = {'a': [{'b': 1}, {'c': 2}, {'b': 3}, {'b': 4}]}
    >>> path_extractor(tree, ['a', lambda x: 'b' in x])
    [{'b': 1}, {'b': 3}, {'b': 4}]
    >>> path_extractor(tree, ['a', lambda x: 'b' in x, 'b'])
    [1, 3, 4]
    """
    if isinstance(path, str):
        path = path.split(path_sep)
    if len(path) == 0:
        return tree
    else:
        idx, *path = path  # extract path[0] as idx & update path to path[1:]
        if isinstance(idx, str) and idx == '*':
            idx = lambda x: True  # use a filter function (but filter everything in)
        if callable(idx) and not isinstance(idx, Literal):
            # If idx is a non-literal callable, consider it as a filter to be applied
            # to iter(tree)
            # TODO: https://github.com/i2mint/i2/issues/27
            return [
                path_extractor(sub_tree, path, getter) for sub_tree in filter(idx, tree)
            ]
        else:
            if isinstance(idx, Literal):
                # Use of Literal is meant get out of trouble if we want to use a
                # callable as an actual index, not as a filter.
                idx = idx.get_val()
            tree = getter(tree, idx)
            return path_extractor(tree, path, getter)


# Note: Specialization of path_extractor for Mappings
def dp_get(d, dot_path):
    """Get stuff from a dict (or any Mapping), using dot_paths (i.e. 'foo.bar' instead of
     ['foo']['bar'])

    >>> d = {'foo': {'bar': 2, 'alice': 'bob'}, 3: {'pi': 3.14}}
    >>> assert dp_get(d, 'foo') == {'bar': 2, 'alice': 'bob'}
    >>> assert dp_get(d, 'foo.bar') == 2
    >>> assert dp_get(d, 'foo.alice') == 'bob'
    """
    return path_extractor(d, dot_path, lambda d, k: d[k])


class lazyprop:
    """
    A descriptor implementation of lazyprop (cached property) from David Beazley's "Python Cookbook" book.
    It's

    >>> class Test:
    ...     def __init__(self, a):
    ...         self.a = a
    ...     @lazyprop
    ...     def len(self):
    ...         print('generating "len"')
    ...         return len(self.a)
    >>> t = Test([0, 1, 2, 3, 4])
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4]}
    >>> t.len
    generating "len"
    5
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4], 'len': 5}
    >>> t.len
    5
    >>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
    >>> t.a = [0, 1, 2]  # if we change a...
    >>> t.len  # ... we still get the old cached value of len
    5
    >>> del t.len  # if we delete the len prop
    >>> t.len  # ... then len being recomputed again
    generating "len"
    3
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class FrozenHashError(TypeError):
    pass


class FrozenDict(dict):
    """An immutable dict subtype that is hashable and can itself be used
    as a :class:`dict` key or :class:`set` entry. What
    :class:`frozenset` is to :class:`set`, FrozenDict is to
    :class:`dict`.

    There was once an attempt to introduce such a type to the standard
    library, but it was rejected: `PEP 416 <https://www.python.org/dev/peps/pep-0416/>`_.

    Because FrozenDict is a :class:`dict` subtype, it automatically
    works everywhere a dict would, including JSON serialization.

    """

    __slots__ = ('_hash',)

    def updated(self, *a, **kw):
        """Make a copy and add items from a dictionary or iterable (and/or
        keyword arguments), overwriting values under an existing
        key. See :meth:`dict.update` for more details.
        """
        data = dict(self)
        data.update(*a, **kw)
        return type(self)(data)

    @classmethod
    def fromkeys(cls, keys, value=None):
        # one of the lesser known and used/useful dict methods
        return cls(dict.fromkeys(keys, value))

    def __repr__(self):
        cn = self.__class__.__name__
        return '%s(%s)' % (cn, dict.__repr__(self))

    def __reduce_ex__(self, protocol):
        return type(self), (dict(self),)

    def __hash__(self):
        try:
            ret = self._hash
        except AttributeError:
            try:
                ret = self._hash = hash(frozenset(self.items()))
            except Exception as e:
                ret = self._hash = FrozenHashError(e)

        if ret.__class__ is FrozenHashError:
            raise ret

        return ret

    def __copy__(self):
        return self  # immutable types don't copy, see tuple's behavior

    # block everything else
    def _raise_frozen_typeerror(self, *a, **kw):
        'raises a TypeError, because FrozenDicts are immutable'
        raise TypeError('%s object is immutable' % self.__class__.__name__)

    __setitem__ = __delitem__ = update = _raise_frozen_typeerror
    setdefault = pop = popitem = clear = _raise_frozen_typeerror

    del _raise_frozen_typeerror


frozendict = FrozenDict  # alias to align with frozenset

########################################################################################################################


function_type = type(
    lambda x: x
)  # using this instead of callable() because classes are callable, for instance


class NoDefault(object):
    def __repr__(self):
        return 'no_default'


no_default = NoDefault()


class imdict(dict):
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


def inject_method(self, method_function, method_name=None):
    """
    method_function could be:
        * a function
        * a {method_name: function, ...} dict (for multiple injections)
        * a list of functions or (function, method_name) pairs
    """
    if isinstance(method_function, function_type):
        if method_name is None:
            method_name = method_function.__name__
        setattr(self, method_name, types.MethodType(method_function, self))
    else:
        if isinstance(method_function, dict):
            method_function = [
                (func, func_name) for func_name, func in method_function.items()
            ]
        for method in method_function:
            if isinstance(method, tuple) and len(method) == 2:
                self = inject_method(self, method[0], method[1])
            else:
                self = inject_method(self, method)

    return self


########################################################################################################################


def get_function_body(func):
    source_lines = inspect.getsourcelines(func)[0]
    source_lines = itertools.dropwhile(lambda x: x.startswith('@'), source_lines)
    line = next(source_lines).strip()
    if not line.startswith('def ') and not line.startswith('class'):
        return line.rsplit(':')[-1].strip()
    elif not line.endswith(':'):
        for line in source_lines:
            line = line.strip()
            if line.endswith(':'):
                break
    # Handle functions that are not one-liners
    first_line = next(source_lines)
    # Find the indentation of the first line
    indentation = len(first_line) - len(first_line.lstrip())
    return ''.join(
        [first_line[indentation:]] + [line[indentation:] for line in source_lines]
    )


class ExistingArgument(ValueError):
    pass


class MissingArgument(ValueError):
    pass


def _default_sentinel_repr_method(self):
    return '%s(%r)' % (self.__class__.__name__, self.__name__)


def mk_sentinel(
    name,
    boolean_value: bool = False,
    repr_: Union[str, Callable] = _default_sentinel_repr_method,
    *,
    module: Optional[str] = None,
):
    """Creates and returns a new **instance** of a new class, suitable for usage as a
    "sentinel" since it is a kind of singleton (there can be only one instance of it.)

    A frequent use case for sentinels are where we want to indicate that something is
    missing. Often, we use ``None`` for this, but sometimes ``None`` is a valid value in
    our context (see for example the ``inspect.Parameter.empty`` sentinel to indicate
    that an argument doesn't have a default or annotation).
    Other times, we may want to distinguish different kinds of "nothing".

    ``mk_sentinel`` can help you create such sentinels, takes care of annoying details
    like pickability and allows you to control how to resolve your sentinel to a boolean.

    :param name: The name of your sentinel. Will be used for ``__name__`` attribute.
    :param boolean_value: The boolean value that the sentinel instance should resolve to.
    :param repr_: The method or string that should be used for the repr.
    :param module:
    :return: A sentinel instance

    >>> Empty = mk_sentinel('Empty')
    >>> Empty
    Sentinel('Empty')

    By default, the boolean resolution of a sentinel is ``False``. Meaning:

    >>> Nothing = mk_sentinel('Nothing')
    >>> bool(Nothing)
    False

    This is consistent with ``None``, so that you can check that an object ``x`` is not
    ``Nothing`` by doing ``if x: ...`` or idioms like:

    >>> x = Nothing
    >>> x = x or 'default'
    >>> x
    'default'

    (Though note that in situations where other elements that cast to ``False`` are
    valid values for ``x`` (like ``0``, ``None``, or ``False`` itself), it's safer to use
    ``if x is not Nothing: ...``.)

    Anyway, I digress.
    Point is that in some situations, the semantics  or usage of your sentinel is better
    align with True. You can control what the boolean resolution of your
    sentinel should be through the ``boolean_value`` argument:

    >>> Empty = mk_sentinel('Empty', boolean_value=True)
    >>> bool(Empty)
    True

    You can also control what you see in the repr, specifying a string value;

    >>> Empty = mk_sentinel('undefined', repr_='undefined')
    >>> Empty
    undefined

    or a method;

    >>> Empty = mk_sentinel('Empty', repr_=lambda self: f"<{self.__name__}>")
    >>> Empty
    <Empty>

    And yes, even though we used a lambda here, it's still picklable:

    >>> import pickle

    >>> Empty = mk_sentinel('Empty', repr_='Empty', module=__name__)
    >>> pickle.loads(pickle.dumps(Empty))  # doctest: +SKIP
    Empty

    Talking about pickle, here's some more info on that:

    >>> unpickled_Empty = pickle.loads(pickle.dumps(Empty))  # doctest: +SKIP
    >>> # The unpickled version is "equal" to the original:
    >>> unpickled_Empty == Empty  # doctest: +SKIP
    True
    >>> # the types are the same too:
    >>> type(unpickled_Empty) == type(Empty)    # doctest: +SKIP
    True
    >>>
    >>>

    Note that though two sentinels might have the same name, they're not equal:

    >>> Empty = mk_sentinel('Empty')
    >>> AnotherEmptyWithSameName = mk_sentinel('Empty')
    >>> Empty
    Sentinel('Empty')
    >>> AnotherEmptyWithSameName
    Sentinel('Empty')
    >>> # but...
    >>> AnotherEmptyWithSameName == Empty
    False
    >>> # Note even the types are the same!
    >>> type(AnotherEmptyWithSameName) == type(Empty)
    False

    One thing that makes the pickle work is that we took care of sticking in a
    ``__module__`` for you. ``mk_sentinel`` figures this out by some dark magic
    involving looking into the system's "frames" etc. This may not always work since
    some systems (e.g. ``pypy``) may use different "under-the-hood" methods.

    But if you want to control the value of ``__module__`` yourself, you can, simply
    but indicating what the module of the sentinel is.
    Usually, you'll just specify it as ``module=__name__``, which will stick the
    name of the module you're defining the sentinel in for you!

    >>> MySentinel = mk_sentinel('MySentinel', module=__name__)

    Thanks: Inspired greately from the ``make_sentinel`` function of ``boltons``:
    See https://boltons.readthedocs.io/.

    """

    class Sentinel(object):
        def __init__(self):
            self.__name__ = name

        if callable(repr_):
            __repr__ = repr_
        else:

            def __repr__(self):
                return repr_

        def __reduce__(self):
            return self.__name__

        def __bool__(self):
            return boolean_value

    if module is None:
        # TODO: Try to use something else than hidden _getframe
        # TODO: extract this module resolver so can be reused (_getframe(2)?)
        frame = sys._getframe(1)
        module = frame.f_globals.get('__name__')

    if not module or module not in sys.modules:
        raise ValueError(
            'Pickleable sentinel objects can only be made from top-level module scopes'
        )
    Sentinel.__module__ = module

    return Sentinel()


def _indent(text, margin, newline='\n', key=bool):
    'based on boltons.strutils.indent'
    indented_lines = [
        (margin + line if key(line) else line) for line in text.splitlines()
    ]
    return newline.join(indented_lines)


NO_DEFAULT = mk_sentinel('NO_DEFAULT', boolean_value=False)


from inspect import formatannotation


def inspect_formatargspec(
    args,
    varargs=None,
    varkw=None,
    defaults=None,
    kwonlyargs=(),
    kwonlydefaults={},
    annotations={},
    formatarg=str,
    formatvarargs=lambda name: '*' + name,
    formatvarkw=lambda name: '**' + name,
    formatvalue=lambda value: '=' + repr(value),
    formatreturns=lambda text: ' -> ' + text,
    formatannotation=formatannotation,
):
    """Copy formatargspec from python 3.7 standard library.
    Python 3 has deprecated formatargspec and requested that Signature
    be used instead, however this requires a full reimplementation
    of formatargspec() in terms of creating Parameter objects and such.
    Instead of introducing all the object-creation overhead and having
    to reinvent from scratch, just copy their compatibility routine.
    """

    def formatargandannotation(arg):
        result = formatarg(arg)
        if arg in annotations:
            result += ': ' + formatannotation(annotations[arg])
        return result

    specs = []
    if defaults:
        firstdefault = len(args) - len(defaults)
    for i, arg in enumerate(args):
        spec = formatargandannotation(arg)
        if defaults and i >= firstdefault:
            spec = spec + formatvalue(defaults[i - firstdefault])
        specs.append(spec)
    if varargs is not None:
        specs.append(formatvarargs(formatargandannotation(varargs)))
    else:
        if kwonlyargs:
            specs.append('*')
    if kwonlyargs:
        for kwonlyarg in kwonlyargs:
            spec = formatargandannotation(kwonlyarg)
            if kwonlydefaults and kwonlyarg in kwonlydefaults:
                spec += formatvalue(kwonlydefaults[kwonlyarg])
            specs.append(spec)
    if varkw is not None:
        specs.append(formatvarkw(formatargandannotation(varkw)))
    result = '(' + ', '.join(specs) + ')'
    if 'return' in annotations:
        result += formatreturns(formatannotation(annotations['return']))
    return result


class FunctionBuilder(object):
    """The FunctionBuilder type provides an interface for programmatically
    creating new functions, either based on existing functions or from
    scratch.

    Note: Based on https://boltons.readthedocs.io

    Values are passed in at construction or set as attributes on the
    instance. For creating a new function based of an existing one,
    see the :meth:`~FunctionBuilder.from_func` classmethod. At any
    point, :meth:`~FunctionBuilder.get_func` can be called to get a
    newly compiled function, based on the values configured.

    >>> fb = FunctionBuilder('return_five', doc='returns the integer 5',
    ...                      body='return 5')
    >>> f = fb.get_func()
    >>> f()
    5
    >>> fb.varkw = 'kw'
    >>> f_kw = fb.get_func()
    >>> f_kw(ignored_arg='ignored_val')
    5

    Note that function signatures themselves changed quite a bit in
    Python 3, so several arguments are only applicable to
    FunctionBuilder in Python 3. Except for *name*, all arguments to
    the constructor are keyword arguments.

    Args:
        name (str): Name of the function.
        doc (str): `Docstring`_ for the function, defaults to empty.
        module (str): Name of the module from which this function was
            imported. Defaults to None.
        body (str): String version of the code representing the body
            of the function. Defaults to ``'pass'``, which will result
            in a function which does nothing and returns ``None``.
        args (list): List of argument names, defaults to empty list,
            denoting no arguments.
        varargs (str): Name of the catch-all variable for positional
            arguments. E.g., "args" if the resultant function is to have
            ``*args`` in the signature. Defaults to None.
        varkw (str): Name of the catch-all variable for keyword
            arguments. E.g., "kwargs" if the resultant function is to have
            ``**kwargs`` in the signature. Defaults to None.
        defaults (tuple): A tuple containing default argument values for
            those arguments that have defaults.
        kwonlyargs (list): Argument names which are only valid as
            keyword arguments. **Python 3 only.**
        kwonlydefaults (dict): A mapping, same as normal *defaults*,
            but only for the *kwonlyargs*. **Python 3 only.**
        annotations (dict): Mapping of type hints and so
            forth. **Python 3 only.**
        filename (str): The filename that will appear in
            tracebacks. Defaults to "boltons.funcutils.FunctionBuilder".
        indent (int): Number of spaces with which to indent the
            function *body*. Values less than 1 will result in an error.
        dict (dict): Any other attributes which should be added to the
            functions compiled with this FunctionBuilder.

    All of these arguments are also made available as attributes which
    can be mutated as necessary.

    .. _Docstring: https://en.wikipedia.org/wiki/Docstring#Python

    """

    _argspec_defaults = {
        'args': list,
        'varargs': lambda: None,
        'varkw': lambda: None,
        'defaults': lambda: None,
        'kwonlyargs': list,
        'kwonlydefaults': dict,
        'annotations': dict,
    }

    @classmethod
    def _argspec_to_dict(cls, f):
        argspec = inspect.getfullargspec(f)
        return dict((attr, getattr(argspec, attr)) for attr in cls._argspec_defaults)

    _defaults = {
        'doc': str,
        'dict': dict,
        'is_async': lambda: False,
        'module': lambda: None,
        'body': lambda: 'pass',
        'indent': lambda: 4,
        'annotations': dict,
        'filename': lambda: 'boltons.funcutils.FunctionBuilder',
    }

    _defaults.update(_argspec_defaults)

    _compile_count = itertools.count()

    def __init__(self, name, **kw):
        self.name = name
        for a, default_factory in self._defaults.items():
            val = kw.pop(a, None)
            if val is None:
                val = default_factory()
            setattr(self, a, val)

        if kw:
            raise TypeError('unexpected kwargs: %r' % kw.keys())
        return

    # def get_argspec(self):  # TODO

    def get_sig_str(self, with_annotations=True):
        """Return function signature as a string.

        with_annotations is ignored on Python 2.  On Python 3 signature
        will omit annotations if it is set to False.
        """
        if with_annotations:
            annotations = self.annotations
        else:
            annotations = {}

        return inspect_formatargspec(
            self.args, self.varargs, self.varkw, [], self.kwonlyargs, {}, annotations
        )

    _KWONLY_MARKER = re.compile(
        r'''
    \*     # a star
    \s*    # followed by any amount of whitespace
    ,      # followed by a comma
    \s*    # followed by any amount of whitespace
    ''',
        re.VERBOSE,
    )

    def get_invocation_str(self):
        kwonly_pairs = None
        formatters = {}
        if self.kwonlyargs:
            kwonly_pairs = dict((arg, arg) for arg in self.kwonlyargs)
            formatters['formatvalue'] = lambda value: '=' + value

        sig = inspect_formatargspec(
            self.args,
            self.varargs,
            self.varkw,
            [],
            kwonly_pairs,
            kwonly_pairs,
            {},
            **formatters,
        )
        sig = self._KWONLY_MARKER.sub('', sig)
        return sig[1:-1]

    @classmethod
    def from_func(cls, func):
        """Create a new FunctionBuilder instance based on an existing
        function. The original function will not be stored or
        modified.
        """
        # TODO: copy_body? gonna need a good signature regex.
        # TODO: might worry about __closure__?
        if not callable(func):
            raise TypeError('expected callable object, not %r' % (func,))

        if isinstance(func, functools.partial):
            kwargs = {
                'name': func.__name__,
                'doc': func.__doc__,
                'module': getattr(func, '__module__', None),  # e.g., method_descriptor
                'annotations': getattr(func, '__annotations__', {}),
                'dict': getattr(func, '__dict__', {}),
            }

        kwargs.update(cls._argspec_to_dict(func))

        if inspect.iscoroutinefunction(func):
            kwargs['is_async'] = True

        return cls(**kwargs)

    def get_func(self, execdict=None, add_source=True, with_dict=True):
        """Compile and return a new function based on the current values of
        the FunctionBuilder.

        Args:
            execdict (dict): The dictionary representing the scope in
                which the compilation should take place. Defaults to an empty
                dict.
            add_source (bool): Whether to add the source used to a
                special ``__source__`` attribute on the resulting
                function. Defaults to True.
            with_dict (bool): Add any custom attributes, if
                applicable. Defaults to True.

        To see an example of usage, see the implementation of
        :func:`~boltons.funcutils.wraps`.
        """
        execdict = execdict or {}
        body = self.body or self._default_body

        tmpl = 'def {name}{sig_str}:'
        tmpl += '\n{body}'

        if self.is_async:
            tmpl = 'async ' + tmpl

        body = _indent(self.body, ' ' * self.indent)

        name = self.name.replace('<', '_').replace('>', '_')  # lambdas
        src = tmpl.format(
            name=name,
            sig_str=self.get_sig_str(with_annotations=False),
            doc=self.doc,
            body=body,
        )
        self._compile(src, execdict)
        func = execdict[name]

        func.__name__ = self.name
        func.__doc__ = self.doc
        func.__defaults__ = self.defaults
        func.__kwdefaults__ = self.kwonlydefaults
        func.__annotations__ = self.annotations

        if with_dict:
            func.__dict__.update(self.dict)
        func.__module__ = self.module
        # TODO: caller module fallback?

        if add_source:
            func.__source__ = src

        return func

    def get_defaults_dict(self):
        """Get a dictionary of function arguments with defaults and the
        respective values.
        """
        ret = dict(
            reversed(list(zip(reversed(self.args), reversed(self.defaults or []))))
        )
        kwonlydefaults = getattr(self, 'kwonlydefaults', None)
        if kwonlydefaults:
            ret.update(kwonlydefaults)
        return ret

    def get_arg_names(self, only_required=False):
        arg_names = tuple(self.args) + tuple(getattr(self, 'kwonlyargs', ()))
        if only_required:
            defaults_dict = self.get_defaults_dict()
            arg_names = tuple([an for an in arg_names if an not in defaults_dict])
        return arg_names

    def add_arg(self, arg_name, default=NO_DEFAULT, kwonly=False):
        """Add an argument with optional *default* (defaults to
        ``funcutils.NO_DEFAULT``). Pass *kwonly=True* to add a
        keyword-only argument
        """
        if arg_name in self.args:
            raise ExistingArgument(
                'arg %r already in func %s arg list' % (arg_name, self.name)
            )
        if arg_name in self.kwonlyargs:
            raise ExistingArgument(
                'arg %r already in func %s kwonly arg list' % (arg_name, self.name)
            )
        if not kwonly:
            self.args.append(arg_name)
            if default is not NO_DEFAULT:
                self.defaults = (self.defaults or ()) + (default,)
        else:
            self.kwonlyargs.append(arg_name)
            if default is not NO_DEFAULT:
                self.kwonlydefaults[arg_name] = default
        return

    def remove_arg(self, arg_name):
        """Remove an argument from this FunctionBuilder's argument list. The
        resulting function will have one less argument per call to
        this function.

        Args:
            arg_name (str): The name of the argument to remove.

        Raises a :exc:`ValueError` if the argument is not present.

        """
        args = self.args
        d_dict = self.get_defaults_dict()
        try:
            args.remove(arg_name)
        except ValueError:
            try:
                self.kwonlyargs.remove(arg_name)
            except (AttributeError, ValueError):
                # py2, or py3 and missing from both
                exc = MissingArgument(
                    'arg %r not found in %s argument list:'
                    ' %r' % (arg_name, self.name, args)
                )
                exc.arg_name = arg_name
                raise exc
            else:
                self.kwonlydefaults.pop(arg_name, None)
        else:
            d_dict.pop(arg_name, None)
            self.defaults = tuple([d_dict[a] for a in args if a in d_dict])
        return

    def _compile(self, src, execdict):

        filename = '<%s-%d>' % (self.filename, next(self._compile_count),)
        try:
            code = compile(src, filename, 'single')
            exec(code, execdict)
        except Exception:
            raise
        return execdict

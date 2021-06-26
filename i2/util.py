import inspect
import re
import itertools
import functools
import types


def dp_get(d, dot_path):
    """
    Get stuff from a dict, using dot_paths (i.e. 'foo.bar' instead of ['foo']['bar'])

    >>> d = {'foo': {'bar': 2, 'alice': 'bob'}, 3: {'pi': 3.14}}
    >>> assert dp_get(d, 'foo') == {'bar': 2, 'alice': 'bob'}
    >>> assert dp_get(d, 'foo.bar') == 2
    >>> assert dp_get(d, 'foo.alice') == 'bob'
    """
    components = dot_path.split('.')
    dd = d.get(components[0])
    for comp in components[1:]:
        dd = dd.get(comp)
    return dd


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


def make_sentinel(name='_MISSING', var_name=None):
    """Creates and returns a new **instance** of a new class, suitable for
    usage as a "sentinel", a kind of singleton often used to indicate
    a value is missing when ``None`` is a valid input.

    Args:
        name (str): Name of the Sentinel
        var_name (str): Set this name to the name of the variable in
            its respective module enable pickleability.

    >>> make_sentinel(var_name='_MISSING')
    _MISSING

    The most common use cases here in boltons are as default values
    for optional function arguments, partly because of its
    less-confusing appearance in automatically generated
    documentation. Sentinels also function well as placeholders in queues
    and linked lists.

    .. note::

      By design, additional calls to ``make_sentinel`` with the same
      values will not produce equivalent objects.

      >>> make_sentinel('TEST') == make_sentinel('TEST')
      False
      >>> type(make_sentinel('TEST')) == type(make_sentinel('TEST'))
      False

    """

    class Sentinel(object):
        def __init__(self):
            self.name = name
            self.var_name = var_name

        def __repr__(self):
            if self.var_name:
                return self.var_name
            return '%s(%r)' % (self.__class__.__name__, self.name)

        if var_name:

            def __reduce__(self):
                return self.var_name

        def __nonzero__(self):
            return False

        __bool__ = __nonzero__

    return Sentinel()


def _indent(text, margin, newline='\n', key=bool):
    'based on boltons.strutils.indent'
    indented_lines = [
        (margin + line if key(line) else line) for line in text.splitlines()
    ]
    return newline.join(indented_lines)


NO_DEFAULT = make_sentinel(var_name='NO_DEFAULT')


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
            **formatters
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

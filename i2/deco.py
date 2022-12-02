"""Decorator tools"""
import inspect
from collections import defaultdict
from contextlib import suppress
from functools import wraps, partial
from inspect import Signature, signature, Parameter
from itertools import chain
from typing import Callable, Tuple, Dict, Any, TypeVar

from i2.signatures import Sig, kind_forgiving_func, name_of_obj

# keep the imports below here because might be referenced (or take care of refs)
# from i2.signatures import ch_signature_to_all_pk, params_of, copy_func

T = TypeVar('T')  # Can be anything


def identity(obj: T) -> T:
    return obj


# ---------------------------------------------------------------------------------------
# The following is an exact copy of the `functools.wraps`, but for some reason, it
# solves an issue of double_up_as_factory on jypyter.
# See description of problem here:
# https://stackoverflow.com/questions/73243479/mysterious-effect-a-decorator-that-breaks-juypyters-ability-to-tab-complete-a

from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES, update_wrapper


def wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES):
    return partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)


def _resolve_inclusion(include, exclude, super_set):
    if isinstance(include, str):
        include = include.split()
    if isinstance(exclude, str):
        exclude = exclude.split()
    if exclude is None:
        exclude = set()
    include = [x for x in (include or super_set) if x not in exclude]
    return include


# ---------------------------------------------------------------------------------------
class FuncFactory:
    """Make a function factory.

    but more convenient and helpful (e.g. is picklable, produces functions with
    signatures, etc.)

    One can use ``functools.partials`` to fix, or change, the defaults of
    arguments of a function ``func`` thereby creating a different function.

    >>> def foo(a, b, *, c=2) -> float:
    ...     return a * b + c
    >>> foo(10, 2)
    22
    >>> foo(10, b=2, c=3)
    23
    >>> from functools import partial
    >>> new_foo = partial(foo, b=2, c=3)  # change default c=3 and add one: b=2
    >>> new_foo(10)  # now the function can be called with one argument (couldn't before)
    23

    In essence, ``FuncFactory`` is equivalent to:

    ```
    FuncFactory = lambda func: lambda *args, **kwargs: partial(func, *args, **kwargs)
    ```

    but more convenient and helpful. For one, it doesn't use ``lambda``, so is picklable.
    It also has a more helpful signature:

    >>> factory = FuncFactory(foo)
    >>> factory
    <FuncFactory(foo)>(a, b, *, c=2) -> Callable[..., float]

    (Note that the repr even reuses ``foo``'s return annotation to tell us that our
    factory will return a callable that returns that type (if the annotation is a type).

    An instance of ``FuncFactory`` is a factory of functions, that is, it can make
    functions for you based on the instance's underlying ``func``:

    >>> f = factory(b=2, c=3)
    >>> f(10)
    23

    Note that:

    >>> ff = factory(2, 3)  # equivalent to ``factory(a=2, b=3)``
    >>> ff(c=10)
    16

    Further, you can tell ``FuncFactory`` to ``include`` or ``exclude`` specific
    arguments, using their names or indices to specify them.

    >>> factory_no_a = FuncFactory(foo, exclude=['a'])
    >>> factory_no_a
    <FuncFactory(foo)>(b, *, c=2) -> Callable[..., float]
    >>> g = factory_no_a(2, 3)  # equivalent to ``factory(b=2, c=3)`` as no ``a`` here
    >>> g(10)
    23

    Recipe: Say you're normalizing some data accessor into callback functions and you
    want to create functions that provide a specific object when called (with no args).
    Sure, you can do this by specifying ``lambda: obj`` every time, but lambdas can be
    problematic (e.g. their not picklable).

    Here's another solution:

    >>> def identity(obj):
    ...     return obj
    >>> func_returning_obj = FuncFactory(identity)
    >>> get_42 = func_returning_obj(42)
    >>> get_42()
    42

    Note: A convenience property has been added to implement this recipe:

    >>> get_42, get_hello = map(FuncFactory.func_returning_obj, (42, 'hello'))
    >>> get_42()
    42
    >>> get_hello()
    'hello'

    """

    def __init__(self, func, *, include=(), exclude=()):
        self.func = func
        func_sig = Sig(func)
        include, exclude = map(func_sig.get_names, (include, exclude))
        include = _resolve_inclusion(include, exclude, func_sig.names)
        self.include = include

        factory_sig = Sig(func_sig, return_annotation=Callable[..., Any])
        factory_sig = factory_sig[self.include]
        # previous I did the following, but don't know why:
        # factory_sig = factory_sig - sig.names[0]
        # Now I know: Because in the func(obj, **params) case it makes more sense
        # that the sig be **params
        # This led to extend to the include/exclude functionality
        # TODO: Consider adding some init control over what arguments of the function we expose for binding.
        if func_sig.return_annotation is not func_sig.empty:
            try:
                factory_sig = Sig(
                    factory_sig,
                    return_annotation=Callable[..., func_sig.return_annotation],
                )
            except TypeError:
                pass

        self.func_sig = func_sig
        self.factory_sig = factory_sig
        self.__signature__ = factory_sig

    def _process_args_and_kwargs(self, args, kwargs):
        _kwargs = self.factory_sig.kwargs_from_args_and_kwargs(
            args, kwargs, allow_partial=True, ignore_kind=True
        )
        __args, __kwargs = self.func_sig.args_and_kwargs_from_kwargs(
            _kwargs, allow_partial=True, ignore_kind=False
        )
        return __args, __kwargs

    def __call__(self, *args, **kwargs):
        args, kwargs = self._process_args_and_kwargs(args, kwargs)
        return partial(self.func, *args, **kwargs)

    def __repr__(self):
        return f'<FuncFactory({name_of_obj(self.func)})>{self.__signature__}'

    @classmethod
    def func_returning_obj(cls, obj):
        return cls(identity)(obj)


def _double_up_as_factory(wrapped=None, *args, __decorator_func=None, **kwargs):
    """Util for double_up_as_factory, ``__decorator_func`` to be partialized"""
    if args:
        raise RuntimeError(
            f'You need to specify decorator arguments as keyword-only.'
            f'You specified positional arguments: {args=}'
        )
    if wrapped is None:  # then we want a factory
        return partial(__decorator_func, **kwargs)
    else:
        return __decorator_func(wrapped, *args, **kwargs)


def double_up_as_factory(decorator_func):
    """Repurpose a decorator both as it's original form, and as a decorator factory.
    That is, from a decorator that is defined do ``wrapped_func = decorator(func, **params)``,
    make it also be able to do ``wrapped_func = decorator(**params)(func)``.

    Note: You'll only be able to do this if all but the first argument are keyword-only,
    and the first argument (the function to decorate) has a default of ``None`` (this is for your own good).
    This is validated before making the "double up as factory" decorator.

    >>> @double_up_as_factory
    ... def decorator(func=None, *, multiplier=2):
    ...     def _func(x):
    ...         return func(x) * multiplier
    ...     return _func
    ...
    >>> def foo(x):
    ...     return x + 1
    ...
    >>> foo(2)
    3
    >>> wrapped_foo = decorator(foo, multiplier=10)
    >>> wrapped_foo(2)
    30
    >>>
    >>> multiply_by_3 = decorator(multiplier=3)
    >>> wrapped_foo = multiply_by_3(foo)
    >>> wrapped_foo(2)
    9
    >>>
    >>> @decorator(multiplier=3)
    ... def foo(x):
    ...     return x + 1
    ...
    >>> foo(2)
    9

    Note that to be able to use double_up_as_factory, your first argument (the object to be wrapped) needs to default
    to None and be the only argument that is not keyword-only (i.e. all other arguments need to be keyword only).

    >>> @double_up_as_factory
    ... def decorator_2(func, *, multiplier=2):
    ...     '''Should not be able to be transformed with double_up_as_factory'''
    Traceback (most recent call last):
      ...
    AssertionError: First argument of the decorator function needs to default to None. Was <class 'inspect._empty'>
    >>> @double_up_as_factory
    ... def decorator_3(func=None, multiplier=2):
    ...     '''Should not be able to be transformed with double_up_as_factory'''
    Traceback (most recent call last):
      ...
    AssertionError: All arguments (besides the first) need to be keyword-only

    """

    def validate_decorator_func(decorator_func):
        first_param, *other_params = signature(decorator_func).parameters.values()
        assert first_param.default is None, (
            f'First argument of the decorator function needs to default to None. '
            f'Was {first_param.default}'
        )
        assert all(
            p.kind in {p.KEYWORD_ONLY, p.VAR_KEYWORD} for p in other_params
        ), f'All arguments (besides the first) need to be keyword-only'
        return True

    validate_decorator_func(decorator_func)

    return wraps(decorator_func)(
        partial(_double_up_as_factory, __decorator_func=decorator_func)
    )


# TODO: Review, doc, and make public.
#   Namely, it may be cleaner to have the `argname=argtrans, ...` interface
#   and offer utils to make an argtrans be the combination of a condition and a trans!
@double_up_as_factory
def _conditional_arg_trans(func=None, **condition_and_trans_of_argname):
    """

    See also: ensure_iterable_args, for a more complex example of how to use it.
    """

    @wraps(func)
    def _func(*args, **kwargs):
        sig = Sig(func)
        kwargs = sig.kwargs_from_args_and_kwargs(args, kwargs)
        for argname, (condition, trans) in condition_and_trans_of_argname.items():
            if argname in kwargs and condition(kwargs[argname]):
                # Note: Using tuple instead of list because quicker & safer (non-mutable)
                kwargs[argname] = trans(kwargs[argname])
        args, kwargs = sig.args_and_kwargs_from_kwargs(kwargs)
        return func(*args, **kwargs)

    return _func


def _tuplize(x):
    return (x,)


def _mk_type_condition(condition):
    """If condition is a type, make an isinstance function to check for that type"""
    curryable_isinstance = kind_forgiving_func(isinstance)
    if isinstance(condition, type) or isinstance(condition, tuple):
        type_condition = condition
        condition = partial(curryable_isinstance, class_or_tuple=type_condition)
    return condition


def ensure_iterable_args(func=None, **condition_of_argname):
    """Wrap a function so that specific arguments are assured to be iterable if
    they meet specific conditions.

    The condition, in the example below, is being a string.
    Note that in general, the condition needs to be a boolean function.
    The explicit form of our example would say `names=lambda x: isinstance(x, str)`,
    but `ensure_iterable_args` allows the convenience of just specifying the type,
    or a tuple of types, and the actually boolean function will be made for you.

    >>> @ensure_iterable_args(names=str)
    ... def greet_people(names, greeting='Hello'):
    ...     for name in names:
    ...         yield f"{greeting} {name}!"
    >>> assert list(greet_people(['Alice', 'Bob'])) == ['Hello Alice!', 'Hello Bob!']
    >>> assert list(greet_people('Alice')) == ['Hello Alice!']

    Note that to decorate a function, you can also use the form:

    >>> greet_people = ensure_iterable_args(greet_people, names=str)
    """
    condition_and_trans_of_argname = {
        argname: (_mk_type_condition(condition), _tuplize)
        for argname, condition in condition_of_argname.items()
    }
    return _conditional_arg_trans(func, **condition_and_trans_of_argname)


def transparently_wrapped(func):
    @wraps(func)
    def transparently_wrapped_func(*args, **kwargs):
        return func(args, **kwargs)

    return transparently_wrapped_func


def mk_args_kwargs_merger(func):
    """
    Make a function that will return a dict containing all {argname: argval} pairs from a function's call.
    That is, it merges all non-keyword arguments with the keyword-arguments, with the right name, so that
    the arguments can be handled more uniformly.
    :param func: The function that will be called, whose signature should be looked at to make the
        merging function
    :return: A function merge_args_and_kwargs(args, kwargs) that can be used to merge arguments

    >>> def func(a, b, c=3):
    ...     return a * (b + c)
    >>> merger = mk_args_kwargs_merger(func)
    >>> dict(merger([1], {'b': 10}))
    {'a': 1, 'b': 10}
    >>> dict(merger([], {'a': 1, 'b': 10}))
    {'a': 1, 'b': 10}
    >>> dict(merger([1, 10], {}))
    {'a': 1, 'b': 10}
    >>> dict(merger([], {}))
    {}
    >>> # Usage demo:
    >>> assert func(*[1], **{'b': 10}) == func(**merger([1], {'b': 10}))
    >>> assert func(*[], **{'a': 1, 'b': 10}) == func(**merger([], {'a': 1, 'b': 10}))
    >>> assert func(**{'a': 1, 'b': 10}) == func(**merger([], {'a': 1, 'b': 10}))
    """

    def merge_args_and_kwargs(args, kwargs):
        if len(args) > 0:
            return inspect.signature(func).bind_partial(*args, **kwargs).arguments
        else:
            return kwargs

    return merge_args_and_kwargs


def kwargs_for_func(*funcs, **kwargs):
    """
    :param funcs:
    :param kwargs:
    :return:

    >>> from i2.tests.objects_for_testing import formula1, sum_of_args, mult, add
    >>> def print_dict(d):  # just a util for this doctest
    ...     from pprint import pprint
    ...     pprint({k.__name__: d[k] for k in sorted(d, key=lambda x: x.__name__)})
    >>> print_dict(kwargs_for_func(formula1, mult, add,
    ...                           w=1, x=2, z=3, a=4, b=5)) # doctest: +NORMALIZE_WHITESPACE
    {'add': {'a': 4, 'b': 5},
     'formula1': {'w': 1, 'x': 2, 'z': 3},
     'mult': {'x': 2}}
    """
    return dict((func, Sig(func).source_kwargs(**kwargs)) for func in funcs)


def assert_attrs(attrs):
    """
    Asserts, at construction time, that the class contains a specific set of attributes
    :param attrs: An attribute name (string) or a list of attribute names whose existence needs to be enforced.
    :return: A class decorator that will enforce the existence of the attrs when an instance is made

    >>> @assert_attrs('foo')
    ... class A:
    ...     bar = 10
    ...
    >>> try:
    ...     a = A()
    ... except AttributeError:
    ...     print("AttributeError, as expected, because missing the foo attribute")
    AttributeError, as expected, because missing the foo attribute
    >>> @assert_attrs('foo')
    ... class B:
    ...     def foo(self): pass
    >>> b = B()
    >>>
    >>> class A:
    ...     bar = 10
    >>> class B:
    ...     def foo(self): pass
    >>>
    >>> @assert_attrs(['foo', 'bar'])
    ... class C(A, B):
    ...     pass
    >>> c = C()
    """
    if isinstance(attrs, str):
        attrs = [attrs]

    def _assert_attrs(klass):
        @wraps(klass)
        def get_instance(*args, **kw):
            for attr in attrs:
                if not hasattr(klass, attr):
                    raise AttributeError(
                        'class {} needs to have a {} attribute:'.format(
                            klass.__name__, attr
                        )
                    )
            return klass(*args, **kw)

        return get_instance

    return _assert_attrs


def preprocess_arguments(pre):
    """Apply a function to args, kwargs and use the transformed in the decorated function"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            args, kwargs = pre(*args, **kwargs)
            return func(*args, **kwargs)

        return wraps(func)(wrapper)

    return decorator


def preprocess(pre):
    def decorator(func):
        if inspect.ismethod(func):

            def wrapper(self, *args, **kwargs):
                return func(self, pre(*args, **kwargs))

        else:

            def wrapper(*args, **kwargs):
                return func(pre(*args, **kwargs))

        return wraps(func)(wrapper)

    return decorator


def _return_annotation_of(func):
    """Return annotation of callable (if type, will return type systematically)

    >>> def foo() -> bool: ...
    >>> assert _return_annotation_of(foo) == bool
    >>> assert _return_annotation_of(zip) == zip
    >>> assert _return_annotation_of(print) == Parameter.empty
    """
    if isinstance(
        func, type
    ):  # TODO: Verify rule (are there commmon enough meta tricks that need to be handled?)
        return func
    else:
        try:
            return signature(func).return_annotation
        except ValueError:  # some builtins don't have signatures
            return Parameter.empty


class OutputPostProcessingError(RuntimeError):
    ...


def postprocess(post, caught_post_errors=(Exception,), verbose_error_message=False):
    """Add some post-processing after a function

    :param post: The function to apply to the output

    >>> list_range = postprocess(list)(range)
    >>> list_range(4)
    [0, 1, 2, 3]
    >>> sum_range = postprocess(sum)(range)
    >>> sum_range(4)
    6

    Note: The decorator also sticks the return annotation of the post function on the wrapped one.

    Use cases:

    - Changing a generator into a container returning function
        In many situations, writing a generator is simpler than writing a function
        that accumulates a list or a dict etc.
        So here, you just write the generator and tag this decorator on top, to get the same effect.

    >>> from inspect import signature
    >>> @postprocess(dict)
    ... def bar(x):
    ...     for i in range(x):
    ...         yield str(i), i
    >>> bar(3)
    {'0': 0, '1': 1, '2': 2}
    >>> signature(bar)
    <Signature (x) -> dict>
    >>>
    >>> @postprocess(list)
    ... def foo(x):
    ...     for i in range(x):
    ...         yield i
    >>> foo(3)
    [0, 1, 2]
    >>> from inspect import signature
    >>> signature(foo)
    <Signature (x) -> list>

    - Triggering something (like logging, or forwarding) when a function returns

    >>> def log_this(x):
    ...     print(f"Logging {x}")
    ...     return x
    >>> logged_foo = postprocess(log_this)(foo)
    >>> t = logged_foo(2)
    Logging [0, 1]
    >>> assert t == [0, 1]

    - Using a function that does a lot to make several functions that do less.
        (e.g. Extracting/making a python object from a function returning a raw http response_

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            try:
                return post(output)
            except caught_post_errors as e:
                msg = f'Error when postprocessing output with post func: {func}'
                if verbose_error_message:
                    msg += '\n' + f'  output={output}'
                    if (
                        isinstance(verbose_error_message, int)
                        and verbose_error_message > 1
                    ):
                        msg += (
                            '\n'
                            + '  which was obtained by func(*args, **kwargs) where:'
                        )
                        msg += (
                            '\n' + f'    args: {args}' + '\n' + f'    kwargs: {kwargs}'
                        )
                msg += '\n' + f'Error is: {e}'
                raise OutputPostProcessingError(msg)

        return_annot = _return_annotation_of(post)
        with suppress(
            ValueError
        ):  # intended to catch cases where wrapper doesn't have a signature
            wrapper_signature = signature(wrapper)
            sig = Signature(
                wrapper_signature.parameters.values(), return_annotation=return_annot,
            )
            wrapper.__signature__ = sig

        return wraps(func)(wrapper)

    return decorator


def input_output_decorator(preprocess=None, postprocess=None):
    """
    Makes a decorator that preprocesses inputs and postprocesses outputs.
    Use it if you want to transform the input of a function or method before calling it, or if you want
    to transform the returned value before returning it.
    :param preprocess: Function to be applied to input
    :param postprocess: Function to be applied to output
    :return: a decorator that preprocesses inputs and postprocesses outputs
    See also: preprocess and postprocess decorators if you need only to pre or post process!

    >>> # Examples with "normal functions"
    >>> def f(x=3):
    ...     '''Some doc...'''
    ...     return x + 10
    >>> ff = input_output_decorator()(f)
    >>> print((ff(5.0)))
    15.0
    >>> ff = input_output_decorator(preprocess=int)(f)
    >>> print((ff(5.0)))
    15
    >>> ff = input_output_decorator(preprocess=int, postprocess=lambda x: "Hello {}!".format(x))(f)
    >>> print((ff('5')))
    Hello 15!
    >>> ff = input_output_decorator(postprocess=lambda x: "Hello {}!".format(x))(f)
    >>> print((ff(5.0)))
    Hello 15.0!
    >>> print((ff.__doc__))
    Some doc...
    >>>
    >>> # examples with methods (bounded, class methods, static methods
    >>> class F:
    ...     '''This is not what you'd expect: The doc of the class, not the function'''
    ...     def __init__(self, y=10):
    ...         '''Initialize'''
    ...         self.y = y
    ...     def __call__(self, x=3):
    ...         '''Some doc...'''
    ...         return self.y + x
    ...     @staticmethod
    ...     def static_method(x, y):
    ...         return "What {} {} you have".format(x, y)
    ...     @classmethod
    ...     def class_method(cls, x):
    ...         return "{} likes {}".format(cls.__name__, x)
    >>>
    >>> f = F()
    >>> ff = input_output_decorator()(f)
    >>> print((ff(5.0)))
    15.0
    >>> ff = input_output_decorator(preprocess=int)(f)
    >>> print((ff(5.0)))
    15
    >>> ff = input_output_decorator(preprocess=int, postprocess=lambda x: "Hello {}!".format(x))(f)
    >>> print((ff('5')))
    Hello 15!
    >>> ff = input_output_decorator(postprocess=lambda x: "Hello {}!".format(x))(f)
    >>> print((ff(5.0)))
    Hello 15.0!
    >>> print((ff.__doc__))
    This is not what you'd expect: The doc of the class, not the function

    # >>>
    # >>> f.static_method = input_output_decorator(preprocess=lambda x: '"' + x + '"',
    # ...                                          postprocess=lambda x: x + '!!!')(f.static_method)
    # >>> print(ff.static_method('big', 'eyes'))
    # What big "eyes" you have!!!
    """

    def decorator(func):
        if preprocess and postprocess:

            def func_wrapper(*args, **kwargs):
                return postprocess(func(preprocess(*args, **kwargs)))

        elif preprocess:  # a preprocess but no postprocess

            def func_wrapper(*args, **kwargs):
                return func(preprocess(*args, **kwargs))

        elif postprocess:  # a postprocess but no preprocess

            def func_wrapper(*args, **kwargs):
                return postprocess(func(*args, **kwargs))

        else:  # neither pre nor post processing, so leave func as is
            return func

        return wraps(func)(func_wrapper)

    decorator.preprocess = preprocess
    decorator.postprocess = postprocess

    return decorator


def transform_args(dflt_trans_func=None, /, **trans_func_for_arg):
    """
    Make a decorator that transforms function arguments before calling the function.
    Works with plain functions and bounded methods.
    For example:
        * original argument: a relative path --> used argument: a full path
        * original argument: a pickle filepath --> used argument: the loaded object
    :param rootdir: rootdir to be used for all name arguments of target function
    :param name_arg: the position (int) or argument name of the argument containing the name
    :return: a decorator

    >>> # Example with a plain function
    >>> def f(a, b, c='default_c'):
    ...     return "a={a}, b={b}, c={c}".format(a=a, b=b, c=c)
    >>> def prepend_root(x):
    ...     return 'ROOT/' + x
    >>>
    >>> def test(f):
    ...     assert f('foo', 'bar', 3) == 'a=foo, b=bar, c=3'
    ...     ff = transform_args()(f)  # no transformation specification, so function is unchanged
    ...     assert ff('foo', 'bar', c=3) == 'a=foo, b=bar, c=3'
    ...     ff = transform_args(a=prepend_root)(f)  # prepend root to a
    ...     assert ff('foo', c=3, b='bar') == 'a=ROOT/foo, b=bar, c=3'  # note: testing different order of args
    ...     ff = transform_args(b=prepend_root)(f)  # prepend root to b
    ...     assert ff(c=3, b='bar', a='foo') == 'a=foo, b=ROOT/bar, c=3'  # note: testing different order of args
    ...     ff = transform_args(a=prepend_root, b=prepend_root)(f)  # prepend root to a and b
    ...     assert ff('foo', 'bar', 3) == 'a=ROOT/foo, b=ROOT/bar, c=3'
    ...     assert ff('foo', 'bar') == 'a=ROOT/foo, b=ROOT/bar, c=default_c'  # defaults still work
    >>>
    >>> test(f)
    >>>
    >>> # Example with bounded method, wrapping from instance
    >>> class A:
    ...     def __init__(self, sep=''):
    ...         self.sep = sep
    ...     def f(self, a, b, c='default_c'):
    ...         return f"a={a}{self.sep} b={b}{self.sep} c={c}"
    >>>
    >>> a = A(sep=',')
    >>> test(a.f)
    >>>
    >>> # Example with bounded method, wrapping from class
    >>> A.f = transform_args(a=prepend_root, b=prepend_root)(A.f)
    >>> a = A(sep=',')
    >>> assert a.f('foo', 'bar', 3) == 'a=ROOT/foo, b=ROOT/bar, c=3'
    >>> assert a.f('foo', 'bar') == 'a=ROOT/foo, b=ROOT/bar, c=default_c'  # defaults still work
    """

    def transform_args_decorator(func):
        get_kwargs = mk_args_kwargs_merger(func)

        if (
            len(trans_func_for_arg) == 0 and not dflt_trans_func
        ):  # if no transformations were specified...
            return func  # just return the function itself
        elif dflt_trans_func is not None:
            assert callable(
                dflt_trans_func
            ), 'The dflt_trans_func needs to be a callable'

            @wraps(func)
            def transform_args_wrapper(*args, **kwargs):
                val_of_argname = get_kwargs(args, kwargs)
                val_of_argname = {
                    argname: dflt_trans_func(val)
                    for argname, val in val_of_argname.items()
                }

                # apply transform functions to argument values
                return func(**val_of_argname)

            return transform_args_wrapper
        else:

            @wraps(func)
            def transform_args_wrapper(*args, **kwargs):
                # get a {argname: argval, ...} dict from *args and **kwargs
                # Note: Didn't really need an if/else here but I am assuming that...
                # Note: ... getcallargs gives us an overhead that can be avoided if there's only keyword args.

                val_of_argname = get_kwargs(args, kwargs)

                for argname, trans_func in trans_func_for_arg.items():
                    if argname in val_of_argname:
                        val_of_argname[argname] = trans_func(val_of_argname[argname])
                # apply transform functions to argument values
                return func(**val_of_argname)

            return transform_args_wrapper

    transform_args_decorator.dflt_trans_func = dflt_trans_func
    transform_args_decorator.trans_func_for_arg = trans_func_for_arg

    return transform_args_decorator


def wrap_method_output(wrapper_func):
    def _wrap_output(wrapped):
        @wraps(wrapped)
        def _wrapped(self, *args, **kwargs):
            return wrapper_func(wrapped(self, *args, **kwargs))

        return _wrapped

    return _wrap_output


def wrap_class_methods(
    _return_a_copy_of_the_class=True,
    _raise_error_if_non_existent_method=True,
    **wrapper_for_method,
):
    """
    Make a decorator that wraps specific methods.

    IMPORTANT: The decorator will by default return a copy of the class. This might incur some run time overhead.
    If this is desirable, for example, when you want to create several decorations of a same class.
    If you want to change the class itself (e.g. you're only loading it once in a module, and decorating it), then
    specify _return_a_copy_of_the_class=False

    Note that _return_a_copy_of_the_class=True has a side effect of building russian dolls of essentially subclasses
    of the class, which may have some undesirable results if repeated too many times.

    :param _return_a_copy_of_the_class: Specifies whether to
        return a copy of the class (_return_a_copy_of_the_class=True, the default),
        or change the actual loaded class itself (_return_a_copy_of_the_class=False)
    :param wrapper_for_method: method_name=wrapper_function pairs.
    :return: A class wrapper. That is, a decorator that takes a class and returns a decorated version of it
        (or decaorates "in-place" if _return_a_copy_of_the_class=False

    SEE ALSO:
        * wrap_method_output: The function that is called for every method we wrap.
        * transform_class_method_input_and_output: A wrap_class_methods that is specialized for input arg and output
            transformation.

    >>> from functools import wraps
    >>> class A:
    ...     def __init__(self, a=10):
    ...         self.a = a
    ...     def add(self, x):
    ...         return self.a + x
    ...     def multiply(self, x):
    ...         return self.a * x
    ...
    >>> a = A()
    >>> a.add(2)
    12
    >>> a.multiply(2)
    20
    >>>
    >>> def log_calls(func):
    ...     name = func.__name__
    ...     @wraps(func)
    ...     def _func(self, *args, **kwargs):
    ...         print("Calling {} with\\n  args={}\\n  kwargs={}".format(name, args, kwargs))
    ...         return func(self, *args, **kwargs)
    ...     return _func
    ...
    >>> AA = wrap_class_methods(**{k: log_calls for k in ['add', 'multiply']})(A)
    >>> a = AA()
    >>> a.add(x=3)
    Calling add with
      args=()
      kwargs={'x': 3}
    13
    >>> a.multiply(3)
    Calling multiply with
      args=(3,)
      kwargs={}
    30
    """

    def class_wrapper(cls):
        if _return_a_copy_of_the_class:
            _cls = type('_' + cls.__name__, cls.__bases__, dict(cls.__dict__))
            # class _cls(cls):
            #     pass
        else:
            _cls = cls
        for method, wrapper in wrapper_for_method.items():
            if hasattr(_cls, method):
                setattr(_cls, method, wrapper(getattr(_cls, method)))
            elif _raise_error_if_non_existent_method:
                raise ValueError(
                    f"{getattr(_cls, '__name__', str(cls))} has no '{method}' method!"
                )
        return _cls

    return class_wrapper


def mk_input_and_output_method_wrapper(method_output_trans=None, **arg_trans):
    def wrap_method(method_func):
        wrapped_method = transform_args(**arg_trans)(method_func)
        if method_output_trans is not None:
            return wrap_method_output(method_output_trans)(wrapped_method)
        else:
            return wrapped_method

    return wrap_method


def transform_class_method_input_and_output(
    cls, method, method_output_trans=None, **arg_trans
):
    wrapped_method = transform_args(**arg_trans)(getattr(cls, method))
    if method_output_trans is not None:
        setattr(
            cls, method, wrap_method_output(method_output_trans)(wrapped_method),
        )
    else:
        setattr(cls, method, wrapped_method)


def wrap_class_methods_input_and_output(
    _return_a_copy_of_the_class=True,
    _raise_error_if_non_existent_method=True,
    **method_trans_spec,
):
    """
    Make a decorator that wraps specific methods, transforming specific argument values a nd output values.

    IMPORTANT: The decorator will by default return a copy of the class. This might incur some run time overhead.
    If this is desirable, for example, when you want to create several decorations of a same class.
    If you want to change the class itself (e.g. you're only loading it once in a module, and decorating it), then
    specify _return_a_copy_of_the_class=False

    :param _return_a_copy_of_the_class: Specifies whether to
        return a copy of the class (_return_a_copy_of_the_class=True, the default),
        or change the actual loaded class itself (_return_a_copy_of_the_class=False)
    :param method_trans_spec: method_name=trans_specs_for_method pairs.
        The trans_specs_for_method is a dict that is understood by transform_class_method_input_and_output.
        Except for one special case, it's keys are argument names and values are callables to call on those
        arguments' values.
        The special case is method_output_trans. This specifies that the callable it points to should be called
        on output of method. Here's one recipe for outputs: If the output of a function is an iterable and you want
        to apply a function trans to each element of the output, specify method_output_trans=lambda x: map(trans, x).
    :return: A wrapped class

    SEE ALSO:
        * mk_method_trans_spec_from_methods_specs_dict: a utility to make method_trans_spec more easily
        * transform_class_method_input_and_output: The function that is called for every method we wrap.

    In the following, we will show two examples.
    - The first is a toy example to demonstrate the basic functionality.
    - The second demonstrates a more involved case, but is still a silly example.
    - The third demonstrates more the type of application we'd use wrap_class_methods_input_and_output for in real life.

    # FIRST EXAMPLE
    We make an Ops class that wraps Counter, allowing one to add items and show the counts of items added.

    >>> from collections import UserDict
    >>> import re
    >>> from collections import Counter
    >>>
    >>> class Ops:
    ...     def __init__(self):
    ...         self.counter = Counter()
    ...     def add_item(self, item):
    ...         self.counter.update({item: 1})
    ...     def show(self):
    ...         return self.counter
    >>> # Here's an example of what Ops does
    >>> ops = Ops()
    >>> for item in ['this', 'is', 'that', 'and', 'that', 'is', 'this']:
    ...     ops.add_item(item)
    ...
    >>> ops.show()
    Counter({'this': 2, 'is': 2, 'that': 2, 'and': 1})
    >>>
    >>> # But say we don't want to count actual words added, but just the first two letters of these words,
    >>> # and say we want to show() to return the dict, not the Counter.
    >>> NewOps = wrap_class_methods_input_and_output(
    ...     _return_a_copy_of_the_class=False,
    ...     add_item=dict(item=lambda x: x[:2]),  # intercept items fed to add_item and keep only 2 first letters
    ...     show=dict(method_output_trans=dict)  # intercept output of show method, converting to dict
    ... )(Ops)
    >>> # let's try it out!
    >>> ops = NewOps()
    >>> for item in ['this', 'is', 'that', 'and', 'that', 'is', 'this']:
    ...     ops.add_item(item)
    ...
    >>> ops.show()
    {'th': 4, 'is': 2, 'an': 1}
    >>> # See that we specified _return_a_copy_of_the_class=False?
    >>> # Now look at what happens if we try to use Ops, the original class, again. It behaves like NewOps.
    >>> # That's usually not the behavior we want, so be careful!
    >>> ops = Ops()
    >>> for item in ['this', 'is', 'that', 'and', 'that', 'is', 'this']:
    ...     ops.add_item(item)
    ...
    >>> ops.show()
    {'th': 4, 'is': 2, 'an': 1}
    >>>
    >>>

    # SECOND EXAMPLE

    Wrap a dict (or rather, the safer collections.UserDict), doing weird things to the input and output
    keys and values

    >>> val_in_trans = lambda x: 'hello {}'.format(x)  # prepend "hello " to incoming values
    >>> val_out_trans = lambda x: re.sub('hello', 'hi', x)  # replace "hello" by "hi" in output values
    >>> key_in_trans = lambda x: '__' + x  # prepend incoming keys with double underscore
    >>> key_out_trans = lambda x: x[2:]  # remove the first two characters (underscores) from keys when output
    >>>
    >>> methods_specs_dict = {
    ...     ('__contains__', '__getitem__', '__setitem__', '__delitem__'): dict(key=key_in_trans),
    ...     '__setitem__': dict(item=val_in_trans),
    ...     '__iter__': dict(method_output_trans=lambda x: map(key_out_trans, x)),
    ...     '__getitem__': dict(method_output_trans=val_out_trans)
    ... }
    >>>
    >>> methods_specs_dict = mk_method_trans_spec_from_methods_specs_dict(methods_specs_dict)
    >>>
    >>> @wrap_class_methods_input_and_output(**methods_specs_dict)
    ... class AA(UserDict):
    ...     pass
    ...
    >>> aa = AA()
    >>> aa['foo'] = 'shoo'  # store 'shoo' under 'foo'
    >>> # the __str__ method isn't wrapped, so we see the actual STORED keys and values
    >>> # we see that __foo, not foo is the actual key, and "hello shoo" the value:
    >>> assert str(aa) == "{'__foo': 'hello shoo'}"
    >>> assert 'foo' in aa  # yet from the interface, it looks like 'foo' is a key of aa...
    >>> assert '__foo' not in aa  # ... and '__foo' is not a key.
    >>> aa['foo'] = 'bar'  # let's replace the value of 'foo'
    >>> assert str(aa) == "{'__foo': 'hello bar'}"  # see what's stored
    >>> aa['star'] = 'wars'  # let's add another
    >>> assert list(aa) == ['foo', 'star']  # what are the keys? (this uses __iter__ under the hood)
    >>> # In the following, we'll use methods keys(), values(), and items(), none of which we wrapped.
    >>> # And yet, they work as expected, since they pass on their work to methods we wrapped.
    >>> assert list(aa.keys()) == ['foo', 'star']  # another way to get keys
    >>> # see here that when we ask for values, we don't get what we asked to store, ...
    >>> # ... nor what is actually stored, but something else
    >>> assert list(aa.values()) == ['hi bar', 'hi wars']
    >>> assert str(list(aa.items())) == "[('foo', 'hi bar'), ('star', 'hi wars')]"  # the keys and values we get from items()
    >>> assert str(aa) == "{'__foo': 'hello bar', '__star': 'hello wars'}"  # what is actually stored
    >>> del aa['foo']  # testing deletion of a key
    >>> assert str(aa) == "{'__star': 'hello wars'}"  # it worked!
    >>>
    >>>

    # THIRD EXAMPLE

    Here again, we'll wrap UserDict. But instead of being silly, we'll pretend we need to store waveforms
    in binary format (so input values will have to be wrapped), but still retrieving these waveforms as lists
    (so output values will have to be wrapped).
    Additionally, we'll pretend we're working with wav files within some root directory, but don't
    want the root dir or the '.wav' extension to appear in our keys. So we'll have to wrap input and output keys.
    Of course, this is just pretend. Don't use this with real waveforms. It won't work.

    >>> root = '/ROOT/DIR/'
    >>> abs_path_of_rel_path = lambda rel_path: root + rel_path + '.wav'  # transform a relative path to an absolute one
    >>> rel_path_of_abs_path = lambda x: x.replace(root, '').replace('.wav', '')  # transform an absolute path to a relative one
    >>> list_to_bytes = bytes
    >>> bytes_to_list = list
    >>>
    >>> methods_specs_dict = {
    ...     ('__contains__', '__getitem__', '__setitem__', '__delitem__'): dict(key=abs_path_of_rel_path),
    ...     '__setitem__': dict(item=list_to_bytes),
    ...     '__iter__': dict(method_output_trans=lambda x: map(rel_path_of_abs_path, x)),
    ...     '__getitem__': dict(method_output_trans=bytes_to_list)
    ... }
    >>>
    >>> methods_specs_dict = mk_method_trans_spec_from_methods_specs_dict(methods_specs_dict)
    >>>
    >>> @wrap_class_methods_input_and_output(**methods_specs_dict)
    ... class Wf(UserDict):
    ...     pass
    ...
    >>> year = [2, 0, 1, 9]
    >>> down = [5, 4, 3, 2, 1]
    >>>
    >>> wf = Wf()
    >>> wf['year'] = year
    >>> print(str(wf).replace("b'", "'"))
    {'/ROOT/DIR/year.wav': '\\x02\\x00\\x01\\t'}
    >>> 'year' in wf
    True
    >>> wf['down'] = down
    >>> print(str(wf).replace("b'", "'"))
    {'/ROOT/DIR/year.wav': '\\x02\\x00\\x01\\t', '/ROOT/DIR/down.wav': '\\x05\\x04\\x03\\x02\\x01'}
    >>> list(wf.keys())
    ['year', 'down']
    >>> list(wf.values())
    [[2, 0, 1, 9], [5, 4, 3, 2, 1]]
    >>> list(wf.items())
    [('year', [2, 0, 1, 9]), ('down', [5, 4, 3, 2, 1])]
    >>> len(wf)
    2
    >>> del wf['year']
    >>> len(wf)
    1
    >>> list(wf.items())
    [('down', [5, 4, 3, 2, 1])]
    """

    wrapper_for_method = {
        method: mk_input_and_output_method_wrapper(**method_trans)
        for method, method_trans in method_trans_spec.items()
    }
    return wrap_class_methods(
        _return_a_copy_of_the_class=_return_a_copy_of_the_class,
        _raise_error_if_non_existent_method=True,
        **wrapper_for_method,
    )

    # def class_wrapper(cls):
    #     if _return_a_copy_of_the_class:
    #         class _cls(cls):
    #             pass
    #     else:
    #         _cls = cls
    #     for method, method_trans in method_trans_spec.items():
    #         if hasattr(_cls, method):
    #             transform_class_method_input_and_output(_cls, method, **method_trans)
    #         elif _raise_error_if_non_existent_method:
    #             if hasattr(cls, '__name__'):
    #                 class_name = cls.__name__
    #             else:
    #                 class_name = str(cls)
    #             raise ValueError("{} has no '{}' method!".format(class_name, method))
    #     return _cls
    #
    # return class_wrapper


def add_method(obj, method_func, method_name=None, class_name=None):
    """
    Dynamically add a method to an object.

    :param obj: The object to add a method to
    :param method_func: The function to use as a method. The first argument must be the object itself
        (usually called self)
    :param method_name: The desired function name. If None, will take method_func.__name__
    :param class_name:  The desired class name. If None, will take type(obj).__name__
    :return: the object, but with the additional method (or a different function for it)

    >>> class A:
    ...     def __init__(self, x=10):
    ...         self.x = x
    >>> def times(self, y):
    ...     return self.x * y
    >>> def plus(self, y):
    ...     return self.x + y
    >>> a = A(x=10)
    >>> a = add_method(a, plus, '__call__')  # add a __call__ method, assigning it to plus
    >>> a(2)
    12
    >>> a = add_method(a, times, '__call__')  # reassign the __call__ method to times instead
    >>> a(2)
    20
    >>> a = add_method(a, plus, '__getitem__')  # assign the method __getitem__ to plus
    >>> a[2]  # see that it works
    12
    >>> a(2)  # and that we still have our __call__ method
    20
    """
    if isinstance(method_func, str):
        method_name = method_func
        method_func = getattr(obj, method_name)
    if method_name is None:
        method_name = method_func.__name__

    base = type(obj)

    if class_name is None:
        class_name = base.__name__
    bases = (base.__bases__[1:]) + (base,)
    bases_names = set(map(lambda x: x.__name__, bases))
    if class_name in bases_names:
        for i in range(6):
            class_name += '_'
            if not class_name in bases_names:
                break
        else:
            raise ValueError(
                "can't find a name for class that is not taken by bases. Consider using explicit name"
            )

    new_keys = set(dir(obj)) - set(chain(*[dir(b) for b in bases]))

    d = {a: getattr(obj, a) for a in new_keys}
    d[method_name] = method_func

    return type(class_name, bases, d)()


def transform_instance_method_input_and_output(
    obj, method, method_output_trans=None, **arg_trans
):
    from warnings import warn

    warn('Not sure transform_instance_method_input_and_output works yet')
    wrapped_method = transform_args(**arg_trans)(getattr(type(obj), method))
    if method_output_trans is not None:
        obj = add_method(
            obj,
            wrap_method_output(method_output_trans)(wrapped_method),
            method_name=method,
        )
    else:
        obj = add_method(obj, wrapped_method, method_name=method)
    return obj


def wrap_instance_methods(
    _return_a_copy_of_the_class=True,
    _raise_error_if_non_existent_method=True,
    **method_trans_spec,
):
    def obj_wrapper(obj):
        for method, method_trans in method_trans_spec.items():
            if hasattr(obj, method):
                obj = transform_instance_method_input_and_output(
                    obj, method, **method_trans
                )
            elif _raise_error_if_non_existent_method:
                if hasattr(obj.__class__, '__name__'):
                    class_name = obj.__name__
                else:
                    class_name = str(obj)
                raise ValueError("{} has no '{}' method!".format(class_name, method))
        return obj

    return obj_wrapper


def mk_method_trans_spec_from_methods_specs_dict(methods_specs_dict):
    """
    Utility to make inputs for wrap_class_methods_input_and_output more easily.
    :param methods_specs_dict: a dict where
        keys are method names (either a single string, or a tuple of strings)
        values are the trans_spec dicts that should be associated to those methods
    :return: A dict in the method_trans_spec (input of wrap_class_method) format.

    >>> methods_specs_dict = {}
    >>> methods_specs_dict['foo'] = {'x': str, 'y': int}
    >>> methods_specs_dict[('foo', 'bar')] = {'z': list, 'method_output_trans': float}
    >>> methods_specs_dict[('bar', )] = {'zz': int}
    >>> method_trans_spec = mk_method_trans_spec_from_methods_specs_dict(methods_specs_dict)
    >>> list(method_trans_spec.keys())
    ['foo', 'bar']
    >>> method_trans_spec['foo']
    {'x': <class 'str'>, 'y': <class 'int'>, 'z': <class 'list'>, 'method_output_trans': <class 'float'>}
    >>> method_trans_spec['bar']
    {'z': <class 'list'>, 'method_output_trans': <class 'float'>, 'zz': <class 'int'>}
    """
    method_trans_spec = defaultdict(dict)
    for methods, specs in methods_specs_dict.items():
        if isinstance(methods, str):
            methods = (methods,)
        for method in methods:
            method_trans_spec[method].update(specs)
    return dict(method_trans_spec)


Args = Tuple
Kwargs = Dict
WhatToLog = Callable[[Callable, Args, Kwargs], Any]


def _special_str(x: Any, max_len=100) -> str:
    """A util function for _call_signature"""
    if isinstance(x, str):
        return "'" + x + "'"
    else:
        x_str = str(x)
        if len(x_str) > max_len:
            type_str = getattr(type(x), '__name__', str(type(x)))
            if hasattr(x, '__repr__'):
                value_str = x.__repr__()
            else:
                value_str = x_str
            x_str = '{}({}...)'.format(type_str, value_str[:20])
        return x_str


def _call_signature(func: Callable, args: Args, kwargs: Kwargs) -> str:
    """
    A util to make a string representation of a call of a function func with given args and kwargs.
    Meant to be the default mk_log_str of mk_call_logger.
    :param func: A callable
    :param args: A tuple of positional arguments
    :param kwargs: A dict of key=val arguments
    :return: A string to represent all of that.

    >>> args = (2, 'sdf', list(range(1000)))
    >>> kwargs = {'z': 'boo', 'zzz': 10}
    >>> print(_call_signature(_call_signature, args, kwargs))
    _call_signature(2, 'sdf', list([0, 1, 2, 3, 4, 5, 6...), z='boo', zzz=10)
    """
    args_signature = ', '.join(map(_special_str, args))
    kwargs_signature = ', '.join(
        ('{}={}'.format(k, _special_str(v)) for k, v in kwargs.items())
    )
    return '{func_name}({signature})'.format(
        func_name=func.__name__,
        signature=', '.join([args_signature, kwargs_signature]),
    )


def mk_call_logger(
    logger=print,
    what_to_log: WhatToLog = _call_signature,
    log_output=False,
    func_is_bounded=False,
):
    """
    Makes a decorator that logs each call to the wrapped function.
    :param logger: The actual function that logs stuff. Default is print. The "stuff" it logs is given by
        the what_to_log argument (a function).
    :param what_to_log: A function taking inputs (func, args, kwargs) of the call, and returning something to log
        (usually, and by default, a string)
    :param func_is_bounded: Whether the function is bounded (like a method) or not
    :return: A decorator

    >>> # Example of use on (unbounded) function, with default args
    >>> @mk_call_logger()
    ... def useless_computation(x, y=2, z='foo'):
    ...     return z * (x + y)
    ...
    >>> _ = useless_computation(3, y=1, z='ha')
    useless_computation(3, y=1, z='ha')

    The same example, but with output logging too

    >>> @mk_call_logger(log_output=True)
    ... def useless_computation(x, y=2, z='foo'):
    ...     return z * (x + y)
    >>> _ = useless_computation(3, y=1, z='ha')
    useless_computation(3, y=1, z='ha')
    -> hahahaha

    And now a bit more involved...

    >>>
    >>> # Example of use on class method, with a different what_to_log function.
    >>> class A:
    ...     def __init__(self, a=10):
    ...         self.a = a
    ...     def add(self, x):
    ...         return self.a + x
    ...     def multiply(self, x):
    ...         return self.a * x
    ...
    >>> def _name_args_kwargs(func, args, kwargs) -> str:
    ...     return "Calling {} with\\n  args={}\\n  kwargs={}".format(func.__name__, args, kwargs)
    ...
    >>>
    >>> log_calls = mk_call_logger(what_to_log=_name_args_kwargs, func_is_bounded=True)
    >>> for method in ['add', 'multiply']:
    ...     A_method = getattr(A, method)
    ...     setattr(A, method, mk_call_logger(what_to_log=_name_args_kwargs, func_is_bounded=True)(A_method))
    ...
    >>>
    >>> a = A()
    >>> a.add(x=2)
    Calling add with
      args=()
      kwargs={'x': 2}
    12
    >>> a.multiply(2)
    Calling multiply with
      args=(2,)
      kwargs={}
    20
    """
    if log_output is True:
        log_output = '-> {}'.format
    assert log_output is False or callable(log_output)

    if not func_is_bounded:

        def log_calls(func):
            @wraps(func)
            def _func(*args, **kwargs):
                logger(what_to_log(func, args, kwargs))
                out = func(*args, **kwargs)
                log_output and logger(log_output(out))
                return out

            _func._logged_with = logger

            return _func

    else:

        def log_calls(func):
            @wraps(func)
            def _func(self, *args, **kwargs):
                logger(what_to_log(func, args, kwargs))
                out = func(self, *args, **kwargs)
                log_output and logger(log_output(out))
                return out

            _func._logged_with = logger

            return _func

    return log_calls


def get_callable_from_factory_if_no_arguments(func_or_factory_thereof: Callable):
    """Will return the input itself if it's a callable with at least one argument.
    If not, it will consider it to be a factory, call it to get the actual
    callable object that the user presumably is seeking to get"""
    assert callable(
        func_or_factory_thereof
    ), f'{func_or_factory_thereof} needs to be callable'
    if len(Sig(func_or_factory_thereof)) == 0:
        # if func_or_factory_thereof has no arguments, assume it's a factory
        func = func_or_factory_thereof()
        # and make sure that now we have arguments
        if not isinstance(func, Callable) or not len(Sig(func)) > 0:
            raise ValueError(
                'Your func_or_factory_thereof had no arguments, so I assumed it '
                "was a factory, called it, but that didn't produce a "
                "callable with at least one argument. So I don't know what to do."
            )
    else:
        func = func_or_factory_thereof
    return func


## This one didn't actually handle position only correctly (just signature)
# def old_ch_func_to_all_pk(func):
#     """Returns a copy of the function where all arguments are of the PK kind.
#     (PK: Positional_or_keyword)
#
#     :param func: A callable
#     :return:
#
#     >>> from py2http.decorators import signature, ch_func_to_all_pk
#     >>>
#     >>> def f(a, /, b, *, c=None, **kwargs): ...
#     ...
#     >>> print(signature(f))
#     (a, /, b, *, c=None, **kwargs)
#     >>> ff = old_ch_func_to_all_pk(f)
#     >>> print(signature(ff))
#     (a, b, c=None, **kwargs)
#     >>> def g(x, y=1, *args, **kwargs): ...
#     ...
#     >>> print(signature(g))
#     (x, y=1, *args, **kwargs)
#     >>> gg = old_ch_func_to_all_pk(g)
#     >>> print(signature(gg))
#     (x, y=1, args=(), **kwargs)
#     """
#     func = tuple_the_args(func)
#     sig = signature(func)
#     func.__signature__ = ch_signature_to_all_pk(sig)
#     return func

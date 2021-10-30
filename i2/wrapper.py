"""A wrapper object and tools to work with it"""

from functools import wraps, partial
from inspect import Parameter, signature
from typing import Mapping, Callable
from types import MethodType

from i2.signatures import Sig

_empty = Parameter.empty


def double_up_as_factory(decorator_func: Callable):
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

    @wraps(decorator_func)
    def _double_up_as_factory(wrapped=None, **kwargs):
        if wrapped is None:  # then we want a factory
            return partial(decorator_func, **kwargs)
        else:
            return decorator_func(wrapped, **kwargs)

    return _double_up_as_factory


def transparent_ingress(*args, **kwargs):
    """
    >>> transparent_ingress(1, 2, test=1)
    ((1, 2), {'test': 1})
    """

    return args, kwargs


def transparent_egress(output):
    """
    >>> transparent_egress('unnecessary_doctest')
    'unnecessary_doctest'
    """

    return output


"""
    How Wrap works:

    ```
     *interface_args, **interface_kwargs
                     │
                     ▼
    ┌───────────────────────────────────┐
    │              ingress              │
    └───────────────────────────────────┘
                     │
                     ▼
          *func_args, **func_kwargs
                     │
                     ▼
    ┌───────────────────────────────────┐
    │               func                │
    └───────────────────────────────────┘
                     │
                     ▼
                 func_output
                     │
                     ▼
    ┌───────────────────────────────────┐
    │              egress               │
    └───────────────────────────────────┘
                     │
                     ▼
                final_output
    ```
"""


class Wrap:
    """A callable function wrapper with interface modifiers.

    :param func: The wrapped function
    :param ingress: The incoming data transformer. It determines the argument properties
        (name, kind, default and annotation) as well as the actual input of the
        wrapped function.
    :param egress: The outgoing data transformer. It also takes precedence over the
        wrapped function to determine the return annotation of the ``Wrap`` instance
    :return: A callable instance wrapping ``func``

    Some examples:

    >>> from inspect import signature
    >>> from i2 import Sig

    >>> def func(a, b):
    ...     return a * b

    >>> wrapped_func = wrap(func)  # no transformations: wrapped_func is the same as func
    >>> assert wrapped_func(2, 'Hi') == func(2, 'Hi') == 'HiHi'

    Modifying the first argument

    >>> def ingress(a, b):
    ...   return (2 * a, b), dict()
    >>> wrapped_func = wrap(func, ingress=ingress)  # first variable is now multiplied by 2
    >>> wrapped_func(2, 'Hi')
    'HiHiHiHi'

    Same using keyword args, we need to use tuple to represent an empty tuple

    >>> def ingress(a, b):
    ...   return tuple(), dict(a=2 * a, b=b) # Note that b MUST be present as well, or an error will be raised
    >>> wrapped_func = wrap(func, ingress=ingress)  # first variable is now multiplied by 2
    >>> wrapped_func(2, 'Hi')
    'HiHiHiHi'

    Using both args and kwargs

    >>> def ingress(a, b):
    ...   return (2 * a, ), dict(b=b)
    >>> wrapped_func = wrap(func, ingress=ingress)  # first variable is now multiplied by 2
    >>> wrapped_func(2, 'Hi')
    'HiHiHiHi'

    We can use ingress to ADD parameters to func

    >>> def ingress(a, b, c):
    ...   return (a, b + c), dict()
    >>> wrapped_func = wrap(func, ingress=ingress)
    >>> # now wrapped_func takes three arguments
    >>> wrapped_func(2, 'Hi', 'world!')
    'Hiworld!Hiworld!'

    Egress is a bit more straightforward, it simply applies to the output of the
    wrapped function. We can use ingress to ADD parameters to func

    >>> def egress(output):
    ...   return output + ' ITSME!!!'
    >>> wrapped_func = wrap(func, ingress=ingress, egress=egress)
    >>> # now wrapped_func takes three arguments
    >>> wrapped_func(2, 'Hi', 'world!')
    'Hiworld!Hiworld! ITSME!!!'


    A more involved example:

    >>> def ingress(a, b: str, c="hi"):
    ...     return (a + len(b) % 2,), dict(string=f"{c} {b}")
    ...
    >>> def func(times, string):
    ...     return times * string
    ...
    >>> wrapped_func = wrap(func, ingress)
    >>> assert wrapped_func(2, "world! ", "Hi") == "Hi world! Hi world! Hi world! "
    >>>
    >>> wrapped_func = wrap(func, egress=len)
    >>> assert wrapped_func(2, "co") == 4 == len("coco") == len(func(2, "co"))
    >>>
    >>> wrapped_func = wrap(func, ingress, egress=len)
    >>> assert (
    ...     wrapped_func(2, "world! ", "Hi")
    ...     == 30
    ...     == len("Hi world! Hi world! Hi world! ")
    ... )

    .. seealso::

        ``wrap`` function.

    """

    def __init__(self, func, ingress=None, egress=None):
        self.func = func
        wraps(func)(self)  # TODO: should we really copy everything by default?

        # remember the actual value of ingress and egress (for reduce to reproduce)
        self._ingress = ingress
        self._egress = egress

        return_annotation = _empty
        if ingress is not None:
            self.ingress = ingress
            return_annotation = Sig(func).return_annotation
        else:
            self.ingress = transparent_ingress

        if egress is not None:
            self.egress = egress
            egress_return_annotation = Sig(egress).return_annotation
            if egress_return_annotation is not Parameter.empty:
                return_annotation = egress_return_annotation
        else:
            self.egress = transparent_egress

        self.__signature__ = Sig(ingress, return_annotation=return_annotation)
        self.__wrapped__ = func
        # TODO: Pros and cons analysis of pointing __wrapped__ to func. partial uses
        #  .func, but wraps looks for __wrapped__

    def __call__(self, *ingress_args, **ingress_kwargs):
        func_args, func_kwargs = self.ingress(*ingress_args, **ingress_kwargs)
        return self.egress(self.func(*func_args, **func_kwargs))

    def __reduce__(self):
        return type(self), (self.func, self._ingress, self._egress)

    def __get__(self, instance, owner):
        return MethodType(self, instance)


def wrap(func, ingress=None, egress=None):
    """Wrap a function, optionally transforming interface, input and output.

    :param func: The wrapped function
    :param ingress: The incoming data transformer. It determines the argument properties
        (name, kind, default and annotation) as well as the actual input of the
        wrapped function.
    :param egress: The outgoing data transformer. It also takes precedence over the
        wrapped function to determine the return annotation of the ``Wrap`` instance
    :return: A callable instance wrapping ``func``

    Consider the following function.

    >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
    ...     return w + x * y ** z
    ...
    >>> assert f(0) == 8
    >>> assert f(1,2) == 17 == 1 + 2 * 2 ** 3

    See that ``f`` is restricted to use ``z`` as keyword only argument kind:

    >>> f(1, 2, 3, 4)
    Traceback (most recent call last):
      ...
    TypeError: f() takes from 1 to 3 positional arguments but 4 were given

    and ``w`` has position only argument kind:

    >>> f(w=1, x=2, y=3, z=4)
    Traceback (most recent call last):
      ...
    TypeError: f() got some positional-only arguments passed as keyword arguments: 'w'

    Say we wanted a version of this function that didn't have the argument kind
    restrinctions, where the annotation of ``x`` was ``int`` and where the default
    of ``z`` was ``10`` instead of ``3``, and doesn't have an annotation.
    We can do so using the following ingress function:

    >>> def ingress(w, x: int = 1, y: int=2, z = 10):
    ...     return (w,), dict(x=x, y=y, z=z)

    The ingress function serves two purposes:

    - Redefining the signature (i.e. the argument names, kinds, defaults,
    and annotations (not including the return annotation, which is taken care of by the
    egress argument).

    - Telling the wrapper how to get from that interface to the interface of the
    wrapped function.

    If we also wanted to add a return_annotation, we could do so via an ``egress``
    function argument:

    >>> def egress(wrapped_func_output) -> float:
    ...     return wrapped_func_output  # because here we don't want to do anything extra

    Now we can use these ingress and egress functions to get the version of ``f`` of
    our dreams:

    >>> h = wrap(f, ingress, egress)

    Let's see what the signature of our new function looks like:

    >>> from inspect import signature
    >>> str(signature(h))
    '(w, x: int = 1, y: int = 2, z=10) -> float'

    Now let's see that we can actually use this new function ``h``, without the
    restrictions of argument kind, getting the same results as the wrapped ``f``,
    but with default ``z=10``.

    What we wanted (but couldn't) do with ``f``:

    >>> h(1, 2, 3, 4)  # == 1 + 2 * 3 ** 4
    163
    >>> h(w=1, x=2, y=3, z=4)
    163

    >>> assert h(0) == h(0, 1) == h(0, 1, 2) == 0 + 1 * 2 ** 10 == 2 ** 10 == 1024

    For more examples, see also the

    .. seealso::

        ``Wrap`` class.

    """
    return Wrap(func, ingress, egress)


class Ingress:
    def __init__(self, inner_sig, *, conserve_kind=False, **changes_for_name):
        self.inner_sig = Sig(inner_sig)
        self.outer_sig = self.inner_sig.ch_names(**changes_for_name)
        if conserve_kind is not True:
            self.outer_sig = self.outer_sig.ch_kinds_to_position_or_keyword()
        self.changes_for_name = changes_for_name
        self.old_name_for_new_name = invert_map(changes_for_name)
        self.outer_sig(self)

    def __call__(self, *ingress_args, **ingress_kwargs):
        func_kwargs = self.outer_sig.kwargs_from_args_and_kwargs(
            ingress_args, ingress_kwargs
        )
        func_kwargs = dict(
            items_with_mapped_keys(func_kwargs, self.old_name_for_new_name)
        )
        return self.inner_sig.args_and_kwargs_from_kwargs(func_kwargs)


def items_with_mapped_keys(d: dict, key_mapper):
    for k, v in d.items():
        # key_mapper.get(k, k) will give the new key name if present, else will use
        # the old
        yield key_mapper.get(k, k), v


def invert_map(d: dict):
    new_d = {v: k for k, v in d.items()}
    if len(new_d) == len(d):
        return new_d
    else:
        raise ValueError(f'There are duplicate keys so I can invert map: {d}')


from i2.signatures import parameter_to_dict


def parameters_to_dict(parameters):
    return {name: parameter_to_dict(param) for name, param in parameters.items()}


# TODO: Fits global pattern -- merge
class InnerMapIngress:
    def __init__(self, inner_sig, *, _allow_reordering=False, **changes_for_name):
        """

        :param inner_sig:
        :param _allow_reordering:
        :param changes_for_name:

        Say we wanted a version of this function that didn't have the argument kind
        restrinctions, where the annotation of ``x`` was ``int`` and where the default
        of ``z`` was ``10`` instead of ``3``, and doesn't have an annotation.

        Let's take the same function mentioned in the docs of ``wrap``

        >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
        ...     return w + x * y ** z

        In order to get a version of this function we wanted (more lenient kinds,
        with some annotations and a default change), we used the ingress function:

        >>> def ingress_we_used(w, x: int = 1, y: int=2, z = 10):
        ...     return (w,), dict(x=x, y=y, z=z)

        Defining an ingress function this way is usually the simplest way to get a
        a wrapped function. But in some cases we need to build the ingress function
        using some predefined rule/protocol. For those cases, ``InnerMapIngress``
        could come in handy.

        You would build your ``ingress`` function like this, for example:

        >>> from inspect import Parameter, signature
        >>> PK = Parameter.POSITIONAL_OR_KEYWORD
        >>> not_specified = Parameter.empty
        >>> ingress = InnerMapIngress(
        ...     f,
        ...     w=dict(kind=PK),  # change kind to PK
        ...     x=dict(annotation=int),  # change annotation from float to int
        ...     y=dict(annotation=int),  # add annotation int
        ...     # change kind to PK, default to 10, and remove annotation:
        ...     z=dict(kind=PK, default=10, annotation=not_specified),
        ... )
        >>> assert (
        ...     str(signature(ingress))
        ...     == str(signature(ingress_we_used))
        ...     == '(w, x: int = 1, y: int = 2, z=10)'
        ... )
        >>> assert (
        ...     ingress(0,1,2,3)
        ...     == ingress_we_used(0,1,2,3)
        ...     == ((0,), {'x': 1, 'y': 2, 'z': 3})
        ... )

        """
        self.inner_sig = Sig(inner_sig)

        self.outer_sig = self.inner_sig.modified(
            _allow_reordering=_allow_reordering, **changes_for_name
        )

        outer_name_for_inner_name = {
            inner_name: change['name']
            for inner_name, change in changes_for_name.items()
            if 'name' in change
        }
        self.inner_name_for_outer_name = invert_map(outer_name_for_inner_name)
        self.outer_sig(self)

    def __call__(self, *ingress_args, **ingress_kwargs):
        # Get the all-keywords version of the arguments (args,kwargs->kwargs)
        func_kwargs = self.outer_sig.kwargs_from_args_and_kwargs(
            ingress_args, ingress_kwargs, apply_defaults=True
        )

        # Modify the keys of func_kwargs so they reflect the inner signature's names
        # That is, map outer names to inner names.
        func_kwargs = dict(
            items_with_mapped_keys(func_kwargs, self.inner_name_for_outer_name)
        )

        # Return an (args,kwargs) pair the respects the inner function's
        # argument kind restrictions.
        return self.inner_sig.args_and_kwargs_from_kwargs(func_kwargs)

    @classmethod
    def from_signature(cls, inner_sig, outer_sig, _allow_reordering=False):
        """

        :param inner_sig:
        :param outer_sig:
        :param _allow_reordering:
        :return:

        >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
        ...     return w + x * y ** z

        >>> def g(w, x: int = 1, y: int=2, z = 10):
        ...     return w + x * y ** z
        ...
        >>> ingress = InnerMapIngress.from_signature(f, g)
        >>> Sig(ingress)
        <Sig (w, x: int = 1, y: int = 2, z=10)>
        >>>
        >>>
        >>>
        >>> h = wrap(f, ingress=InnerMapIngress.from_signature(f, g))
        >>> assert h(0) == g(0) == 1024 == 0 + 1 * 2 ** 10
        >>> assert h(1,2) == g(1,2) == 2049 == 1 + 2 * 2 ** 10
        >>> assert h(1,2,3,4) == g(1,2,3,4) == 1 + 2 * 3 ** 4
        >>>
        >>> assert h(w=1,x=2,y=3,z=4) == g(1,2,3,4) == 1 + 2 * 3 ** 4
        """
        outer_sig = Sig(outer_sig)
        return cls(
            inner_sig,
            _allow_reordering=_allow_reordering,
            **parameters_to_dict(outer_sig.parameters),
        )


# TODO: Fits global pattern -- merge
class ArgNameMappingIngress:
    def __init__(self, inner_sig, *, conserve_kind=False, **outer_name_for_inner_name):
        self.inner_sig = Sig(inner_sig)
        self.outer_sig = self.inner_sig.ch_names(**outer_name_for_inner_name)
        if conserve_kind is not True:
            self.outer_sig = self.outer_sig.ch_kinds_to_position_or_keyword()
        self.inner_name_for_outer_name = invert_map(outer_name_for_inner_name)
        self.outer_sig(self)

    def __call__(self, *ingress_args, **ingress_kwargs):
        # Get the all-keywords version of the arguments (args,kwargs->kwargs)
        func_kwargs = self.outer_sig.kwargs_from_args_and_kwargs(
            ingress_args, ingress_kwargs
        )
        # Modify the keys of func_kwargs so they reflect the inner signature's names
        # That is, map outer names to inner names.
        func_kwargs = dict(
            items_with_mapped_keys(func_kwargs, self.inner_name_for_outer_name)
        )
        # Return an (args,kwargs) pair the respects the inner function's
        # argument kind restrictions.
        return self.inner_sig.args_and_kwargs_from_kwargs(func_kwargs)


def mk_ingress_from_name_mapper(func, name_mapper: Mapping, *, conserve_kind=False):
    return ArgNameMappingIngress(func, conserve_kind=conserve_kind, **name_mapper)


# class Ingress:
#     @classmethod
#     def name_map(cls, wrapped, **new_names):
#         """"""
#
#     @classmethod
#     def defaults(cls, wrapped, **defaults):
#         """"""
#
#     @classmethod
#     def order(cls, wrapped, arg_order):
#         """"""
#
#     @classmethod
#     def factory(cls, wrapped, **func_for_name):
#         """"""


def nice_kinds(func):
    """Wraps the func so it will only have POSITIONAL_OR_KEYWORD argument kinds.

    The original purpose of this function is to remove argument-kind restriction
    annoyances when doing functional manipulations such as:

    >>> from functools import partial
    >>> isinstance_of_str = partial(isinstance, class_or_tuple=str)
    >>> isinstance_of_str('I am a string')
    Traceback (most recent call last):
      ...
    TypeError: isinstance() takes no keyword arguments

    Here, instead, we can just get a kinder version of the function and do what we
    want to do:

    >>> _isinstance = nice_kinds(isinstance)
    >>> isinstance_of_str = partial(_isinstance, class_or_tuple=str)
    >>> isinstance_of_str('I am a string')
    True
    >>> isinstance_of_str(42)
    False

    """
    from i2 import Sig, call_somewhat_forgivingly

    sig = Sig(func)
    sig = sig.ch_kinds(**{name: Sig.POSITIONAL_OR_KEYWORD for name in sig.names})

    @wraps(func)
    def _func(*args, **kwargs):
        return call_somewhat_forgivingly(func, args, kwargs, enforce_sig=sig)

    _func.__signature__ = sig
    return _func


# ---------------------------------------------------------------------------------------
# wrap tools


def arg_val_converter(func, **conversion_for_arg):
    return Wrap(func, ingress=ArgValConverterIngress(func, **conversion_for_arg))


def arg_val_converter_ingress(func, __strict=True, **conversion_for_arg):
    sig = Sig(func)
    if __strict:
        conversion_names_that_are_not_func_args = conversion_for_arg.keys() - sig.names
        assert not conversion_names_that_are_not_func_args, (
            'Some of the arguments you want to convert are not argument names '
            f'for the function: {conversion_names_that_are_not_func_args}'
        )

    @sig
    def ingress(*args, **kwargs):
        # TODO: Make a helper function for this ak -> k -> proc -> ak pattern
        kwargs = sig.kwargs_from_args_and_kwargs(args, kwargs)
        kwargs = dict(convert_dict_values(kwargs, conversion_for_arg))
        args, kwargs = sig.args_and_kwargs_from_kwargs(kwargs)
        return args, kwargs

    return ingress


# TODO: Fits global pattern -- merge
class ArgValConverterIngress:
    def __init__(self, func, __strict=True, **conversion_for_arg):
        sig = Sig(func)
        if __strict:
            conversion_names_that_are_not_func_args = (
                conversion_for_arg.keys() - sig.names
            )
            assert not conversion_names_that_are_not_func_args, (
                'Some of the arguments you want to convert are not argument names '
                f'for the function: {conversion_names_that_are_not_func_args}'
            )
        self.sig = sig
        self.conversion_for_arg = conversion_for_arg
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        # TODO: Make a helper function for this ak -> k -> proc -> ak pattern
        kwargs = self.sig.kwargs_from_args_and_kwargs(args, kwargs)
        kwargs = dict(convert_dict_values(kwargs, self.conversion_for_arg))
        args, kwargs = self.sig.args_and_kwargs_from_kwargs(kwargs)
        return args, kwargs


def convert_dict_values(to_convert: dict, key_to_conversion_function: dict):
    for k, v in to_convert.items():
        if k in key_to_conversion_function:
            conversion_func = key_to_conversion_function[k]
            yield k, conversion_func(v)  # converted kv pair
        else:
            yield k, v  # unconverted kv pair


# TODO: Test for performance an ask about readability
def _alt_convert_dict_values(to_convert: dict, key_to_conversion_function: dict):
    for k, v in to_convert.items():
        conversion_func = key_to_conversion_function.get(k, lambda x: x)
        yield k, conversion_func(v)

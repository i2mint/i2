"""A wrapper object and tools to work with it"""

from functools import wraps
from inspect import Parameter, Signature
from typing import Mapping
from i2 import Sig

_empty = Parameter.empty


def transparent_ingress(*args, **kwargs):
    return args, kwargs


def transparent_egress(output):
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

from types import MethodType


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
    >>>
    >>> def ingress(a, b: str, c="hi"):
    ...     return (a + len(b) % 2,), dict(string=f"{c} {b}")
    ...
    >>> def func(times, string):
    ...     return times * string
    ...
    >>> wrapped_func = wrap(func)  # no transformations
    >>> assert wrapped_func(2, "co") == "coco" == func(2, "co")
    >>>
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


class Ingress:
    @classmethod
    def name_map(cls, wrapped, **new_names):
        """"""

    @classmethod
    def defaults(cls, wrapped, **defaults):
        """"""

    @classmethod
    def order(cls, wrapped, arg_order):
        """"""

    @classmethod
    def factory(cls, wrapped, **func_for_name):
        """"""


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
# tests


def _test_ingress(a, b: str, c='hi'):
    return (a + len(b) % 2,), dict(string=f'{c} {b}')


def _test_func(times, string):
    return times * string


def test_wrap():
    import pickle
    from inspect import signature

    func = _test_func

    # Just wrapping the func gives you a sort of copy of the func.
    wrapped_func = wrap(func)  # no transformations
    assert wrapped_func(2, 'co') == 'coco' == func(2, 'co')

    # If you give the wrap an ingress function
    ingress = _test_ingress
    wrapped_func = wrap(func, ingress)
    # it will use it to (1) determine the signature of the wrapped_func
    assert (
        str(signature(wrapped_func)) == "(a, b: str, c='hi')"
    )  # "(a, b: str, c='hi')"
    # and (2) to map inputs
    assert wrapped_func(2, 'world! ', 'Hi') == 'Hi world! Hi world! Hi world! '

    # An egress function can be used to transform outputs
    wrapped_func = wrap(func, egress=len)
    assert wrapped_func(2, 'co') == 4 == len('coco') == len(func(2, 'co'))

    # Both ingress and egress can be used in combination
    wrapped_func = wrap(func, ingress, egress=len)
    assert (
        wrapped_func(2, 'world! ', 'Hi') == 30 == len('Hi world! Hi world! Hi world! ')
    )

    # A wrapped function is pickle-able (unlike the usual way decorators are written)

    unpickled_wrapped_func = pickle.loads(pickle.dumps(wrapped_func))
    assert (
        unpickled_wrapped_func(2, 'world! ', 'Hi')
        == 30
        == len('Hi world! Hi world! Hi world! ')
    )


def _test_foo(a, b: int, c=7):
    return a + b * c


def _test_bar(a, /, b: int, *, c=7):
    return a + b * c


def test_mk_ingress_from_name_mapper():
    import pickle
    from inspect import signature

    foo = _test_foo
    # Define the mapping (keys are inner and values are outer names)
    name_mapper = dict(a='aa', c='cc')
    # Make an ingress function with that mapping
    ingress = mk_ingress_from_name_mapper(foo, name_mapper)
    # Use the ingress function to wrap a function
    wrapped_foo = wrap(foo, ingress)
    # See that the signature of the wrapped func uses the mapped arg names
    assert (
        str(signature(wrapped_foo)) == str(signature(ingress)) == '(aa, b: int, cc=7)'
    )
    # And that wrapped function does compute correctly
    assert (
        foo(1, 2, c=4)
        == foo(a=1, b=2, c=4)
        == wrapped_foo(aa=1, b=2, cc=4)
        == wrapped_foo(1, 2, cc=4)
    )
    # The ingress function returns args and kwargs for wrapped function
    assert ingress('i was called aa', b='i am b', cc=42) == (
        (),
        {'a': 'i was called aa', 'b': 'i am b', 'c': 42},
    )
    # See above that the args is empty. That will be the case most of the time.
    # Keyword arguments will be favored when there's a choice. If wrapped
    # function uses position-only arguments though, ingress will have to use them
    bar = _test_bar
    assert str(signature(bar)) == '(a, /, b: int, *, c=7)'
    ingress_for_bar = mk_ingress_from_name_mapper(bar, name_mapper)
    assert ingress_for_bar('i was called aa', b='i am b', cc=42) == (
        ('i was called aa',),
        {'b': 'i am b', 'c': 42},
    )
    wrapped_bar = wrap(bar, ingress_for_bar)
    assert (
        bar(1, 2, c=4)
        == bar(1, b=2, c=4)
        == wrapped_bar(1, b=2, cc=4)
        == wrapped_bar(1, 2, cc=4)
    )

    # Note that though bar had a positional only and a keyword only argument,
    # we are (by default) free of argument kind constraints in the wrapped function:
    # We can can use a positional args on `cc` and keyword args on `aa`
    assert str(signature(wrapped_bar)) == '(aa, b: int, cc=7)'
    assert wrapped_bar(1, 2, 4) == wrapped_bar(aa=1, b=2, cc=4)

    # If you want to conserve the argument kinds of the wrapped function, you can
    # specify this with `conserve_kind=True`:
    ingress_for_bar = mk_ingress_from_name_mapper(bar, name_mapper, conserve_kind=True)
    wrapped_bar = wrap(bar, ingress_for_bar)
    assert str(signature(wrapped_bar)) == '(aa, /, b: int, *, cc=7)'

    # A wrapped function is pickle-able (unlike the usual way decorators are written)
    unpickled_wrapped_foo = pickle.loads(pickle.dumps(wrapped_foo))
    assert (
        str(signature(unpickled_wrapped_foo))
        == str(signature(ingress))
        == '(aa, b: int, cc=7)'
    )
    assert (
        foo(1, 2, c=4)
        == foo(a=1, b=2, c=4)
        == unpickled_wrapped_foo(aa=1, b=2, cc=4)
        == unpickled_wrapped_foo(1, 2, cc=4)
    )


test_wrap()
test_mk_ingress_from_name_mapper()

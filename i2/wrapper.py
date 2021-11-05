"""A wrapper object and tools to work with it"""

from functools import wraps, partial
from inspect import Parameter, signature
from typing import Mapping, Callable
from types import MethodType

from i2.signatures import Sig
from i2.multi_object import Pipe

empty = Parameter.empty


def identity(x):
    """Transparent function, returning what's been input"""
    return x


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


# TODO: Put in module docs when md docs work!
"""
    How the `Wrap` class works:

    ```
          *outer_args, **outer_kwargs
                     │
                     ▼
    ┌───────────────────────────────────┐
    │              ingress              │
    └───────────────────────────────────┘
                     │
                     ▼
          *inner_args, **inner_kwargs
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
    
    How the `Ingress` class (ingress templated function maker) works:
    
    ```
          *outer_args, **outer_kwargs
                     │
                     ▼
    ┌───────────────────────────────────┐
    │          outer_sig_bind           │
    └───────────────────────────────────┘
                     │
                     ▼
              outer_all_kwargs
                     │
                     ▼
    ┌───────────────────────────────────┐
    │            kwargs_trans           │
    └───────────────────────────────────┘
                     │
                     ▼
              inner_all_kwargs
                     │
                     ▼
    ┌───────────────────────────────────┐
    │          inner_sig_bind           │
    └───────────────────────────────────┘
                     │
                     ▼
          *inner_args, **inner_kwargs
    ```
    
"""


class Wrap:
    """A function wrapper with interface modifiers.

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

        return_annotation = empty
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


def append_empty_args(func):
    """To use to transform an ingress function that only returns kwargs to one that
    returns the normal form of ingress functions: ((), kwargs)"""

    @wraps(func)
    def _func(*args, **kwargs):
        return (), func(*args, **kwargs)

    return _func


class Ingress:
    """The Ingress class offers a template for creating ingress classes.

    Note that when writing a decorator with i2.wrapper, you're usually better off
    writing an ingress function for the purpose. As a result, your code will usually
    be less complex, easier to read, and more efficient than using the Ingress class.

    So why use the Ingress class at all? For one, because it'll take care of some common
    mechanics for you, so once you understand how to use it,
    you'll probably create a correct wrapper faster.

    Further, if you're writing a general wrapping tool (e.g. your own currying machine,
    some rule-based input casting function, etc.) then you'll find that using
    Ingres will usually with on the complexity, readability and/or efficiency front.

    >>> from i2.wrapper import Ingress, wrap
    >>> from inspect import signature
    >>> from i2.wrapper import InnerMapIngress
    >>>
    >>> # Ingress = InnerMapIngress
    >>>
    >>> def f(w, /, x: float, y=2, *, z: int = 3):
    ...     return f"(w:={w}) + (x:={x}) * (y:={y}) ** (z:={z}) == {w + x * y ** z}"
    >>>
    >>> f(0, 1)
    '(w:=0) + (x:=1) * (y:=2) ** (z:=3) == 8'

    Let’s say you wanted to dispatch this function to a command line interface,
    or a webservice where all arguments are taken from the url.
    The problem here is that this means that all incoming values will be strings
    in that case.
    Say you wanted all input values to be cast to ints. In that case you could do:

    >>> trans_all_vals_to_ints = lambda d: {k: int(v) for k, v in d.items()}
    >>>
    >>> cli_f = wrap(
    ...     f,
    ...     ingress=Ingress(signature(f), kwargs_trans=trans_all_vals_to_ints)
    ... )
    >>>
    >>> cli_f("2", "3", "4")
    '(w:=2) + (x:=3) * (y:=4) ** (z:=3) == 194'

    In a more realistic situation, you'd want to have more control over this value
    transformation.

    Say you wanted to convert to int if it's possible, try float if not,
    and just leave the string alone otherwise.

    >>> def _try_casting_to_numeric(x):
    ...     try:
    ...         return int(x)
    ...     except ValueError:
    ...         try:
    ...             return float(x)
    ...         except ValueError:
    ...             return x
    ...
    >>> def cast_numbers(d: dict):
    ...     return {k: _try_casting_to_numeric(v) for k, v in d.items()}
    >>>
    >>> cli_f = wrap(f, ingress=Ingress(signature(f), kwargs_trans=cast_numbers))
    >>>
    >>> cli_f("2", "3.14", "4")
    '(w:=2) + (x:=3.14) * (y:=4) ** (z:=3) == 202.96'

    Let's say that our values transformations are not all 1-to-1 as in the examples
    above.
    Instead, they can be

    - `1-to-many` (e.g. the outer 'w' is used to compute the inner `w` and `x`)

    - `many-to-1 (e.g. the outer `x` and `y` are used to compute inner `y`)

    ```
      w   x   y   z
     / \   \ /    |
    w   x   y     z
    ```

    >>> def kwargs_trans(outer_kw):
    ...     return dict(
    ...         # e.g. 1-to-many: one outer arg (w) producing two inner args (w, and y)
    ...         w=outer_kw['w'] * 2,
    ...         x=outer_kw['w'] * 3,
    ...         # e.g. many-to-1: two outer args (x and y) producing one inner arg (y)
    ...         y=outer_kw['x'] + outer_kw['y'],
    ...         # Note that no z is mentioned: This means we're just leaving it alone
    ...     )
    ...
    >>>
    >>> ingress = Ingress(signature(f), kwargs_trans=kwargs_trans)
    >>> assert ingress(2, x=3, y=4) == ((4,), {'x': 6, 'y': 7, 'z': 3})
    >>>
    >>> wrapped_f = wrap(f, ingress)
    >>> assert wrapped_f(2, x=3, y=4) == '(w:=4) + (x:=6) * (y:=7) ** (z:=3) == 2062'


    The following is an example that involves several aspects of the `Ingress` class.

    >>> from i2 import Sig
    >>> def kwargs_trans(outer_kw):
    ...     return dict(
    ...         w=outer_kw['w'] * 2,
    ...         x=outer_kw['w'] * 3,
    ...         # need to pop you (inner func has no you argument)
    ...         y=outer_kw['x'] + outer_kw.pop('you'),
    ...         # Note that no z is mentioned: This means we're just leaving it alone
    ...     )
    >>>
    >>> ingress = Ingress(
    ...     inner_sig=signature(f),
    ...     kwargs_trans=kwargs_trans,
    ...     outer_sig=Sig(f).ch_names(y='you')  # need to give the outer sig a you
    ...     # You could also express it this way (though you'd loose the annotations)
    ...     # outer_sig=lambda w, /, x, you=2, *, z=3: None
    ... )
    >>> assert ingress(2, x=3, you=4) == ((4,), {'x': 6, 'y': 7, 'z': 3})
    >>>
    >>> wrapped_f = wrap(f, ingress)
    >>> assert wrapped_f(2, x=3, you=4) == '(w:=4) + (x:=6) * (y:=7) ** (z:=3) == 2062'

    """

    def __init__(self, inner_sig, kwargs_trans=None, outer_sig=None):
        """Init of an Ingress instance.

        :param inner_sig: Signature of the inner function the ingress is for.
            The function itself can be given and the signature will be extracted.
        :param kwargs_trans: A dict-to-dict transformation of the outer kwargs to
            the kwargs that should be input to the inner function.
        :param outer_sig: The outer signature. The signature the ingress function
            (there for the wrapped function) will have. Also serves to convert input
            (args, kwargs) to the kwargs that will be given to kwargs_trans.

        When making an Ingress function directly, one must take care that
        `inner_sig`, `kwargs_trans` and `outer_sig` are aligned.

        Namely, 'kwargs_trans' must be able to handle outputs of
        `outer_sig.kwargs_from_args_and_kwargs` and itself output kwargs that
        can be handled by `inner_sig.args_and_kwargs_from_kwargs`.

        """
        self.inner_sig = Sig(inner_sig)

        # kwargs_trans should be callable and have one required arg: a dict
        # if it's None, we'll just make it be the identity function
        if kwargs_trans is None:
            kwargs_trans = identity
        self.kwargs_trans = kwargs_trans

        # default to inner_sig = outer_sig
        if outer_sig is None:
            outer_sig = inner_sig
        self.outer_sig = Sig(outer_sig)

        self.outer_sig(self)

    def __call__(self, *ingress_args, **ingress_kwargs):
        # Get the all-keywords version of the arguments (args,kwargs->kwargs)
        func_kwargs = self.outer_sig.kwargs_from_args_and_kwargs(
            ingress_args, ingress_kwargs, apply_defaults=True
        )

        func_kwargs = dict(
            func_kwargs,  # by default, keep the func_kwargs, but
            **self.kwargs_trans(func_kwargs),  # change those that kwargs_trans desires
        )

        # Return an (args,kwargs) pair the respects the inner function's
        # argument kind restrictions.
        return self.inner_sig.args_and_kwargs_from_kwargs(
            func_kwargs, apply_defaults=True
        )

    @classmethod
    def name_map(cls, wrapped, **new_names):
        """Change argument names"""
        return cls(
            wrapped,
            partial(Pipe(items_with_mapped_keys, dict), key_mapper=new_names),
            Sig(wrapped).ch_names(**new_names),
        )

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


def _handle_ingress_class_inputs(
    inner_sig, kwargs_trans, outer_sig, *, _allow_reordering=False
):
    inner_sig = Sig(inner_sig)

    # kwargs_trans should be callable and have one required arg: a dict
    # if it's None, we'll just make it be the identity function
    if kwargs_trans is None:
        kwargs_trans = identity

    # default to inner_sig = outer_sig
    if outer_sig is None:
        outer_sig = inner_sig  # if nothing specified, want same outer and inner sigs
    elif isinstance(outer_sig, Mapping):
        changes_for_name = outer_sig  # it's a dict of modifications of the inner sig
        outer_sig = inner_sig.modified(
            _allow_reordering=_allow_reordering, **changes_for_name
        )
    else:
        outer_sig = Sig(outer_sig)

    return inner_sig, kwargs_trans, outer_sig


class InnerMapIngress:
    """A class to help build ingresses systematically by mapping the inner signature.

    *Systematically*, i.e. "according to a fixed plan/system" is what it's about
    here. As we'll see below, if you need to write a particular adapter for a
    specific

    :param inner_sig: The signature of the wrapped function.
    :param _allow_reordering: Whether we want to allow reordering of variables
    :param in_to_out_sig_changes: The `inner_name=dict_of_changes_for_that_name`
    pairs, the `dict_of_changes_for_that_name` is a `dict` with keys being valid
    `inspect.Parameter`

    Consider the following function that has a position only, a keyword only,
    two arguments with annotations, and three with a default.

    >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
    ...     return w + x * y ** z

    Say we wanted a version of this function

    - that didn't have the argument kind restrinctions (all POSITION_OR_KEYWORD),

    - where the annotation of ``x`` was changed ``int`` and the default removed

    - where `y` was named `you` instead, and has an annotation (`int`).

    - where the default of ``z`` was ``10`` instead of ``3``, and doesn't have an
    annotation.

    In order to get a version of this function we wanted (more lenient kinds,
    with some annotations and a default change), we can use the ingress function:

    >>> def directly_defined_ingress(w, x: int, you: int=2, z = 10):
    ...     return (w,), dict(x=x, y=you, z=z)


    When we need to wrap a specific function in a specific way, defining an
    ingress
    function  this way is usually the simplest way.
    But in some cases we need to build the ingress function using some predefined
    rule/protocol to make applying rule/protocol systematic.

    For those cases, ``InnerMapIngress`` comes in handy.

    With `InnerMapIngress` we'd build our ingress function like this:

    >>> from inspect import Parameter, signature
    >>> PK = Parameter.POSITIONAL_OR_KEYWORD
    >>> empty = Parameter.empty
    >>> ingress = InnerMapIngress(
    ...     f,
    ...     # change kind to PK:
    ...     w=dict(kind=PK),
    ...     # change annotation of x from float to int and remove default
    ...     x=dict(annotation=int, default=empty),
    ...     # rename y to you and add annotation int:
    ...     y=dict(name='you', annotation=int),
    ...     # change kind to PK, default to 10, and remove annotation:
    ...     z=dict(kind=PK, default=10, annotation=empty),
    ... )

    Note:

    - Only the changes we wish to make to the parameters are mentioned.
    You could also define the parameters explicitly by simply listing all three
    of the dimensions (kind, annotation, and default)

    - Three? But a `Parameter` object has four; what about the name?
    Indeed, you can use name as well, more on that later.

    - Note that in order to specify that you want no default, or no annotation,
    you cannot use `None` since `None` is both a valid default and a valid
    annotation; So instead you need to use `Parameter.empty` (conveniently
    assigned
    to a constant named `empty` in the `wrapping` module.

    Now see that all arguments are `POSITIONAL_OR_KEYWORD`, `x` and `y` are
    `int`,
    and default of `z` is 10:

    >>> assert (
    ...     str(signature(ingress))
    ...     == str(signature(directly_defined_ingress))
    ...     == '(w, x: int, you: int = 2, z=10)'
    ... )

    Additionally, `ingress` function does it's job of dispatching the right args
    and kwargs to the target function:

    >>> assert (
    ...     ingress(0,1,2,3)
    ...     == directly_defined_ingress(0,1,2,3)
    ...     == ((0,), {'x': 1, 'y': 2, 'z': 3})
    ... )


    """

    def __init__(
        self,
        inner_sig,
        kwargs_trans=None,
        *,
        _allow_reordering=False,
        **changes_for_name,
    ):
        self.inner_sig = Sig(inner_sig)

        self.outer_sig = self.inner_sig.modified(
            _allow_reordering=_allow_reordering, **changes_for_name
        )

        # kwargs_trans should be callable and have one required arg: a dict
        # if it's None, we'll just make it be the identity function
        if kwargs_trans is None:
            kwargs_trans = identity
        self.kwargs_trans = kwargs_trans

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
        func_kwargs = dict(
            func_kwargs,  # by default, keep the func_kwargs, but
            **self.kwargs_trans(func_kwargs),  # change those that kwargs_trans desires
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

        Say we wanted to get a version of the function:

        >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
        ...     return w + x * y ** z

        That was equivalent to (note the kind, default and annotation differences):

        >>> def g(w, x=1, y=2, z=10):
        ...     return w + x * y ** z


        >>> ingress = InnerMapIngress.from_signature(
        ... f, outer_sig=lambda w, x=1, y=2, z=10: None
        ... )
        >>> Sig(ingress)
        <Sig (w, x=1, y=2, z=10)>
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

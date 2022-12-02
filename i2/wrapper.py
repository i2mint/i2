"""A wrapper object and tools to work with it

How the ``Wrap`` class works:

.. code-block::

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


How the ``Ingress`` class (ingress templated function maker) works:

.. code-block::

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

"""

from functools import wraps, partial
from inspect import Parameter, signature
from typing import (
    Mapping,
    Callable,
    Any,
    Optional,
    Union,
    Iterable,
    Sequence,
    NewType,
    Dict,
    Tuple,
)

from types import MethodType
from inspect import Parameter
from dataclasses import make_dataclass, dataclass

from i2.signatures import (
    Sig,
    name_of_obj,
    KO,
    PK,
    VK,
    parameter_to_dict,
    call_forgivingly,
    _call_forgivingly,
)
from i2.errors import OverwritesNotAllowed
from i2.multi_object import Pipe
from i2.deco import double_up_as_factory

# ---------------------------------------------------------------------------------------
# Wrap

empty = Parameter.empty
OuterKwargs = dict
InnerKwargs = dict
KwargsTrans = Callable[[OuterKwargs], InnerKwargs]


def identity(x):
    """Transparent function, returning what's been input"""
    return x


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


class MakeFromFunc:
    """Used to indicate that an object should be made as a function of an input func"""

    def __init__(self, func_to_obj):
        self.func_to_obj = func_to_obj

    def __call__(self, func):
        return self.func_to_obj(func)


# TODO: Continue factoring out Wrap and Wrapx code
class _Wrap:
    """To be used as the base of actual Wrap objects."""

    _init_args = ()
    _init_kwargs = ()

    def __init__(self, func, *args, **kwargs):
        self._init_args = (func, *args)
        self._init_kwargs = kwargs
        # wraps(func)(self) is there to copy over to self anything that func may
        # have had. It should be before anything else so it doesn't overwrite stuff
        # that we may add to self in init (like .func for example!)
        wraps(func)(self)  # TODO: should we really copy everything by default?
        if name := kwargs.get('name', None) is not None:
            self.__name__ = name
        self.func = func  # Note: overwrites self.func that wraps MAY have inserted
        self.__wrapped__ = func
        # TODO: Pros and cons analysis of pointing __wrapped__ to func. partial uses
        #  .func, but wraps looks for __wrapped__

    def __call__(self, *args, **kwargs):
        """Just forward the call to the wrapped function."""
        return self.func(*args, **kwargs)

    def __reduce__(self):
        """reduce is meant to control how things are pickled/unpickled"""
        return type(self), self._init_args, dict(self._init_kwargs)

    def __get__(self, instance, owner):
        """This method allows things to work well when we use Wrap object as method"""
        if instance is None:
            return self
        return MethodType(self, instance)

    def __repr__(self):
        # TODO: Replace i2.Wrap with dynamic (Wrap or Wrapx)
        name = getattr(self, '__name__', None) or 'Wrap'
        return f'<i2.Wrap {name}{signature(self)}>'

    # TODO: Don't know exactly what I'm doing below. Review with someone!
    def __set_name__(self, owner, name):
        """So that name of function is passed on to method when assigning to attribute
        That is, doing ``method = Wrap(func)`` in a class definition"""
        # TODO: Look into sanity of mutating the name and other ways to achieve same
        self.__name__ = name

    def __set__(self, instance, value):
        instance.__dict__[self.__name__] = value

    # To get help(instance.method) to work!
    # TODO: Does this have undesirable side effects?
    @property
    def __code__(self):
        return self.func.__code__


def _defaults_and_kwdefaults_of_func(func: Callable):
    try:
        return func.__defaults__, func.__kwdefaults__
    except AttributeError:
        # if python can't do it on it's own, you may have a special kind of callable
        # here, so try doing it through Sig, which is more flexible
        sig = Sig(func)
        return sig._defaults_, sig._kwdefaults_


class Wrap(_Wrap):
    """A function wrapper with interface modifiers.

    :param func: The wrapped function
    :param ingress: The incoming data transformer. It determines the argument properties
        (name, kind, default and annotation) as well as the actual input of the
        wrapped function.
    :param egress: The outgoing data transformer. It also takes precedence over the
        wrapped function to determine the return annotation of the ``Wrap`` instance
    :param name: Name to give the wrapper (will use wrapped func name by default)
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
    >>> wrapped_func = wrap(func, ingress=ingress)
    >>> assert wrapped_func(2, "world! ", "Hi") == "Hi world! Hi world! Hi world! "
    >>>
    >>> wrapped_func = wrap(func, egress=len)
    >>> assert wrapped_func(2, "co") == 4 == len("coco") == len(func(2, "co"))
    >>>
    >>> wrapped_func = wrap(func, ingress=ingress, egress=len)
    >>> assert (
    ...     wrapped_func(2, "world! ", "Hi")
    ...     == 30
    ...     == len("Hi world! Hi world! Hi world! ")
    ... )

    An ``ingress`` function links the interface of the wrapper to the interface of the
    wrapped func; therefore it's definition often depends on information of both,
    and for that reason, we provide the ability to specify the ingress not only
    explicitly (as in the examples above), but through a factory -- a function that
    will be called on ``func`` to produce the ingress that should be used to wrap it.

    .. seealso::

        ``wrap`` function.

    """

    def __init__(self, func, ingress=None, egress=None, *, name=None):
        super().__init__(func, ingress, egress, name=name)
        ingress_sig = Sig(func)

        if ingress is None:
            self.ingress = transparent_ingress
            self.__defaults__, self.__kwdefaults__ = _defaults_and_kwdefaults_of_func(
                func
            )
        else:

            if isinstance(ingress, MakeFromFunc):
                self.ingress = ingress  # the ingress function is
                func_to_ingress = ingress  # it's not the ingress function itself
                # ... but an ingress factory: Should make the ingress in function of func
                self.ingress = func_to_ingress(func)
            else:
                assert callable(ingress), f'Should be callable: {ingress}'
                self.ingress = ingress

            ingress_sig = Sig(self.ingress)
            self.__defaults__, self.__kwdefaults__ = _defaults_and_kwdefaults_of_func(
                self.ingress
            )

        return_annotation = empty

        if egress is None:
            self.egress = transparent_egress
        else:
            self.egress = egress
            egress_return_annotation = Sig(egress).return_annotation
            if egress_return_annotation is not Parameter.empty:
                return_annotation = egress_return_annotation

        self.__signature__ = Sig(ingress_sig, return_annotation=return_annotation)

    def __call__(self, *ingress_args, **ingress_kwargs):
        func_args, func_kwargs = self.ingress(*ingress_args, **ingress_kwargs)
        return self.egress(self.func(*func_args, **func_kwargs))


@double_up_as_factory
def wrap(
    func=None, *, ingress=None, egress=None, caller=None, name=None, dflt_wrap=Wrap
):
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

    >>> h = wrap(f, ingress=ingress, egress=egress)

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

    Note that ``wrap`` can also be used as a decorator "factory", for instance to
    wrap functions at definition time, and if we change ``caller`` it will automatically
    use ``Wrapx`` instead of ``Wrap`` to wrap the function.

    >>> def iterize(func, args, kwargs):
    ...     first_arg_val = next(iter(kwargs.values()))
    ...     return list(map(func, first_arg_val))
    >>>
    >>> @wrap(caller=iterize)
    ... def func(x, y=2):
    ...     return x + y
    ...
    >>> isinstance(func, Wrapx)
    True
    >>> func([1, 2, 3, 4])
    [3, 4, 5, 6]

    For more examples, see also the

    .. seealso::

        ``Wrap`` class.
        ``Wrapx`` class.

    """
    if _should_use_wrapx(func, ingress, egress, caller):
        return Wrapx(func, ingress, egress, caller=caller, name=name)
    else:
        return dflt_wrap(func, ingress, egress, name=name)


# TODO: Add conditions on egress to route to Wrapx when complex egress
def _should_use_wrapx(func, ingress, egress, caller):
    if caller is not None:
        return True
    else:
        return False


def append_empty_args(func):
    """To use to transform an ingress function that only returns kwargs to one that
    returns the normal form of ingress functions: ((), kwargs)"""

    @wraps(func)
    def _func(*args, **kwargs):
        return (), func(*args, **kwargs)

    return _func


class Ingress:
    """The Ingress class offers a template for creating ingress classes.

    Note that when writing a decorator with ``i2.wrapper``, you're usually better off
    writing an ingress function for the purpose. As a result, your code will usually
    be less complex, easier to read, and more efficient than using the Ingress class.

    So why use the ``Ingress`` class at all? For one, because it'll take care of some
    common mechanics for you, so once you understand how to use it, you'll probably
    create a correct wrapper faster.

    Further, if you're writing a general wrapping tool (e.g. your own currying machine,
    some rule-based input casting function, etc.) then you'll find that using
    Ingres will usually with on the complexity, readability and/or efficiency front.

    Consider the following function:

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

    >>> from i2.wrapper import Ingress, wrap
    >>> from inspect import signature
    >>>
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

    - ``1-to-many`` (e.g. the outer ``w`` is used to compute the inner ``w`` and ``x``)

    - ``many-to-1`` (e.g. the outer ``x`` and ``y`` are used to compute inner ``y``)

    .. code-block::

          w   x   y   z
         / \   \ /    |
        w   x   y     z


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
    >>> wrapped_f = wrap(f, ingress=ingress)
    >>> assert wrapped_f(2, x=3, y=4) == '(w:=4) + (x:=6) * (y:=7) ** (z:=3) == 2062'


    The following is an example that involves several aspects of the ``Ingress`` class.

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
    ...     # You could also express it this way (though you'd lose the annotations)
    ...     # outer_sig=lambda w, /, x, you=2, *, z=3: None
    ... )
    >>> assert ingress(2, x=3, you=4) == ((4,), {'x': 6, 'y': 7, 'z': 3})
    >>>
    >>> wrapped_f = wrap(f, ingress=ingress)
    >>> assert wrapped_f(2, x=3, you=4) == '(w:=4) + (x:=6) * (y:=7) ** (z:=3) == 2062'

    A convenience method allows to do the same with the ingress instance itself:

    >>> wrapped_f = ingress.wrap(f)
    >>> assert wrapped_f(2, x=3, you=4) == '(w:=4) + (x:=6) * (y:=7) ** (z:=3) == 2062'
    """

    def __init__(
        self,
        inner_sig,
        kwargs_trans: Optional[KwargsTrans] = None,
        outer_sig=None,
        *,
        allow_excess=True,
        apply_defaults=True,
        allow_partial=False,
    ):
        """Init of an Ingress instance.

        :param inner_sig: Signature of the inner function the ingress is for.
            The function itself can be given and the signature will be extracted.
        :param kwargs_trans: A dict-to-dict transformation of the outer kwargs to
            the kwargs that should be input to the inner function.
            That is ``kwargs_trans`` is ``outer_kwargs -> inner_kwargs``.
            Note that though both outer and inner signatures could have those annoying
            position-only kinds, you don't have to think of that.
            The parameter kind restrictions are taken care of automatically.
        :param outer_sig: The outer signature. The ingress function's signature will
            have (and therefore, the wrapped function's signature too.
            Also serves to convert input ``(args, kwargs)`` to the ``kwargs``
            to the kwargs that will be given to ``kwargs_trans``.
        :param allow_excess: Whether to allow the inner kwargs to have some excess
            variables in them. This enables more flexibility in the outer signature,
            but may want to set to ``False`` to be more explicit.
        :param apply_defaults: Whether to apply the defaults of the outer signature.
            Default is ``True`` but in some rare cases, you may not want to apply them.

        When making an Ingress function directly, one must take care that
        ``inner_sig``, ``kwargs_trans`` and ``outer_sig`` are aligned.

        Namely, ``kwargs_trans`` must be able to handle outputs of
        ``outer_sig.kwargs_from_args_and_kwargs`` and itself output kwargs that
        can be handled by ``inner_sig.args_and_kwargs_from_kwargs``.

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
        self.apply_defaults = apply_defaults
        self.allow_excess = allow_excess
        self.allow_partial = allow_partial

    def __call__(self, *ingress_args, **ingress_kwargs):
        # Get the all-keywords version of the arguments (args,kwargs->kwargs)
        func_kwargs = self.outer_sig.kwargs_from_args_and_kwargs(
            ingress_args, ingress_kwargs, apply_defaults=self.apply_defaults
        )

        func_kwargs = dict(
            func_kwargs,  # by default, keep the func_kwargs, but
            **self.kwargs_trans(func_kwargs),  # change those that kwargs_trans desires
        )

        # Return an (args, kwargs) pair that respects the inner function's
        # argument kind restrictions.
        # TODO: Reflect on pros/cons of allow_excess=True
        return self.inner_sig.args_and_kwargs_from_kwargs(
            func_kwargs,
            apply_defaults=True,
            allow_excess=self.allow_excess,
            allow_partial=self.allow_partial,
        )

    def __repr__(self):
        return f'Ingress signature: {signature(self)}'

    def wrap(self, func: Callable, egress=None, *, name=None) -> Wrap:
        """Convenience method to wrap a function with the instance ingress.
        ``ingress.wrap(func,...)`` equivalent to ``Wrap(func, ingress, ...)``
        """
        return Wrap(func, ingress=self, egress=egress, name=name)

    @classmethod
    def name_map(cls, wrapped, **old_to_new_name):
        """Change argument names.

        >>> def f(w, /, x: float, y=2, *, z: int = 3):
        ...     return f"(w:={w}) + (x:={x}) * (y:={y}) ** (z:={z}) == {w + x * y ** z}"
        >>> ingress = Ingress.name_map(f, w='DoubleYou', z='Zee')
        >>> ingress
        Ingress signature: (DoubleYou, /, x: float, y=2, *, Zee: int = 3)
        >>> wrapped_f = ingress.wrap(f)
        >>> wrapped_f(1, 2, y=3, Zee=4)
        '(w:=1) + (x:=2) * (y:=3) ** (z:=4) == 163'

        """
        new_to_old_name = {v: k for k, v in old_to_new_name.items()}
        assert len(new_to_old_name) == len(
            old_to_new_name
        ), f'Inversion is not possible since {old_to_new_name=} has duplicate values.'
        return cls(
            wrapped,
            partial(Pipe(items_with_mapped_keys, dict), key_mapper=new_to_old_name),
            Sig(wrapped).ch_names(**old_to_new_name),
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
    """Transform dict keys. More precisely yield (new_k,v) pairs from a key mapper dict.

    :param d: src dict
    :param key_mapper: {old_name: new_name, ...} mapping
    :return: generator of (new_name, value) pairs

    Often used in conjunction with dict:

    >>> dict(items_with_mapped_keys(
    ...     {'a': 1, 'b': 2, 'c': 3, 'd': 4},
    ...     {'a': 'Ay', 'd': 'Dee'})
    ... )
    {'Ay': 1, 'b': 2, 'c': 3, 'Dee': 4}

    """
    for k, v in d.items():
        # key_mapper.get(k, k) will give the new key name if present,
        # else will use the old
        yield key_mapper.get(k, k), v


def invert_map(d: dict):
    new_d = {v: k for k, v in d.items()}
    if len(new_d) == len(d):
        return new_d
    else:
        raise ValueError(f'There are duplicate keys so I can invert map: {d}')


def parameters_to_dict(parameters):
    return {name: parameter_to_dict(param) for name, param in parameters.items()}


def _handle_ingress_class_inputs(
    inner_sig,
    kwargs_trans: Optional[KwargsTrans],
    outer_sig,
    *,
    _allow_reordering=False,
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


# TODO: See what this adds over ``Ingress`` class. Consider merging or reusing.
class InnerMapIngress:
    """A class to help build ingresses systematically by mapping the inner signature.

    *Systematically*, i.e. "according to a fixed plan/system" is what it's about here.
    As we'll see below, if you need to write a particular adapter for a specific case,
    you probably should do by writing an actual ingress function directly.
    In cases where you might want to apply a same logic to wrap many functions,
    you may want to fix that wrapping logic: ``InnerMapIngress`` provides one
    way to do this.

    :param inner_sig: The signature of the wrapped function.
    :param kwargs_trans: A dict-to-dict transformation of the outer kwargs to
        the kwargs that should be input to the inner function.
        That is ``kwargs_trans`` is ``outer_kwargs -> inner_kwargs``.
        Note that though both outer and inner signatures could have those annoying
        position-only kinds, you don't have to think of that.
        The parameter kind restrictions are taken care of automatically.
    :param _allow_reordering: Whether we want to allow reordering of variables
    :param in_to_out_sig_changes: The ``inner_name=dict_of_changes_for_that_name``
    pairs, the ``dict_of_changes_for_that_name`` is a ``dict`` with keys being valid
    ``inspect.Parameter``

    Consider the following function that has a position only, a keyword only,
    two arguments with annotations, and three with a default.

    >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
    ...     return w + x * y ** z

    Say we wanted a version of this function

    - that didn't have the argument kind restrinctions (all POSITION_OR_KEYWORD),

    - where the annotation of ``x`` was changed ``int`` and the default removed

    - where ``y`` was named ``you`` instead, and has an annotation (``int``).

    - where the default of ``z`` was ``10`` instead of ``3``, and doesn't have an
        annotation.

    In order to get a version of this function we wanted (more lenient kinds,
    with some annotations and a default change), we can use the ingress function:

    >>> def directly_defined_ingress(w, x: int, you: int=2, z = 10):
    ...     return (w,), dict(x=x, y=you, z=z)


    When we need to wrap a specific function in a specific way, defining an
    ingress function  this way is usually the simplest way.
    But in some cases we need to build the ingress function using some predefined
    rule/protocol to make applying the rule/protocol systematic.

    For those cases, ``InnerMapIngress`` comes in handy.

    With ``InnerMapIngress`` we'd build our ingress function like this:

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

    - Three? But a ``Parameter`` object has four; what about the name?
        Indeed, you can use name as well, more on that later.

    - Note that in order to specify that you want no default, or no annotation,
        you cannot use ``None`` since ``None`` is both a valid default and a valid
        annotation; So instead you need to use ``Parameter.empty`` (conveniently
        assigned to a constant named ``empty`` in the ``wrapping`` module.

    Now see that all arguments are ``POSITIONAL_OR_KEYWORD``, ``x`` and ``y`` are
    ``int``, and default of ``z`` is 10:

    >>> assert (
    ...     str(signature(ingress))
    ...     == str(signature(directly_defined_ingress))
    ...     == '(w, x: int, you: int = 2, z=10)'
    ... )

    Additionally, ``ingress`` function does it's job of dispatching the right args
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
        kwargs_trans: Optional[KwargsTrans] = None,
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

        # TODO: See Ingress: Has allow_partial etc.
        # Return an (args, kwargs) pair the respects the inner function's
        # argument kind restrictions.
        return self.inner_sig.args_and_kwargs_from_kwargs(func_kwargs)

    # TODO: Use/test annotations of outer_sig
    @classmethod
    def from_signature(cls, inner_sig, outer_sig, _allow_reordering=False):
        """
        A convienience ingress constructor to specify wrappings that affect arguments
        independently.

        :param inner_sig: The signature of wrapped, inner function (or the inner
        function itself)
        :param outer_sig: The desired outer signature. Can also use a function (will
        only take it's signature though).
        :param _allow_reordering: Whether to allow ``outer_sig`` to reorder arguments.
        :return: An ingress that will allow one to use a function having the
        ``inner_sig`` signature to

        Say we wanted to get a version of the function:

        >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
        ...     return w + x * y ** z

        That was equivalent to (note the kind, default and annotation differences):

        >>> def g(w, x=1, y: float = 2.0, z=10):
        ...     return w + x * y ** z

        >>> h = wrap(f, ingress=InnerMapIngress.from_signature(f, g))
        >>> Sig(h)
        <Sig (w, x=1, y: float = 2.0, z=10)>

        Note we could have used ``...from_signature(Sig(f), Sig(g))`` as well,
        since the method doesn't use the actual functions, just their signatures.

        So we've seen that ``h`` takes on the signature (kind, defaults,
        and annotations) of ``g``.
        Let's see now that ``h`` actually computes, uses the defaults of ``g`` and
        can doesn't have the position only restriction on ``w``.

        >>> assert h(0) == g(0) == 1024 == 0 + 1 * 2 ** 10
        >>> assert h(1,2) == g(1,2) == 2049 == 1 + 2 * 2 ** 10
        >>> assert h(1,2,3,4) == g(1,2,3,4) == 1 + 2 * 3 ** 4
        >>>
        >>> assert h(w=1,x=2,y=3,z=4) == g(1,2,3,4) == 1 + 2 * 3 ** 4  # w keyword arg!

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


# TODO: Replace with i2.wrapper instead
# TODO: Make sure VARIADICs are handled properly, or error raised if present


def apply_func_on_cond(func, cond, k, v):
    if cond(k, v):
        return func(v)
    else:
        return v


def modify_dict_on_cond(d, cond, func):
    return {k: apply_func_on_cond(func, cond, k, v) for k, v in d.items()}


def convert_to_PK(kinds):
    return {name: PK for name in kinds}


def convert_VK_to_KO(kinds):
    cond = lambda k, v: v == VK
    func = lambda v: KO

    return modify_dict_on_cond(kinds, cond, func)


def nice_kinds(func, kinds_modifier=convert_to_PK):
    """Wraps the func, changing the argument kinds according to kinds_modifier.
    The default behaviour is to change all kinds to POSITIONAL_OR_KEYWORD kinds.
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

    See also: ``i2.signatures.all_pk_signature``

    """
    from i2 import Sig, call_somewhat_forgivingly

    sig = Sig(func)
    kinds_modif = kinds_modifier(sig.kinds)

    sig = sig.ch_kinds(**kinds_modif)

    # sig = sig.ch_kinds(**{name: Sig.POSITIONAL_OR_KEYWORD for name in sig.names})

    @wraps(func)
    def _func(*args, **kwargs):
        return call_somewhat_forgivingly(func, args, kwargs, enforce_sig=sig)

    _func.__signature__ = sig
    return _func


def wrap_from_sig(func, new_sig):
    from i2 import call_somewhat_forgivingly

    @wraps(func)
    def _func(*args, **kwargs):
        return call_somewhat_forgivingly(func, args, kwargs, enforce_sig=new_sig)

    _func.__signature__ = new_sig

    return _func


# ---------------------------------------------------------------------------------------
# wrap tools


@double_up_as_factory
def ch_names(func=None, **old_to_new_name):
    """Change the argument names of a function.

    >>> def f(w, /, x: float, y=2, *, z: int = 3):
    ...     return f"(w:={w}) + (x:={x}) * (y:={y}) ** (z:={z}) == {w + x * y ** z}"
    >>> wrapped_f = ch_names(f, w='DoubleYou', z='Zee')
    >>> wrapped_f
    <i2.Wrap f(DoubleYou, /, x: float, y=2, *, Zee: int = 3)>
    >>> wrapped_f(1, 2, y=3, Zee=4)
    '(w:=1) + (x:=2) * (y:=3) ** (z:=4) == 163'

    Can also be used as a factory:
    >>> @ch_names(a='alpha', g='gamma')
    ... def foo(a, b, g=1):
    ...     return a + b * g
    >>> foo(alpha=1, b=2, gamma=3)
    7
    """
    return Ingress.name_map(func, **old_to_new_name).wrap(func)


map_names = ch_names  # back-compatibility alias

from i2.deco import _resolve_inclusion


def include_exclude_ingress_factory(func, include=None, exclude=None):
    """A pattern underlying any ingress that takes a subset of parameters (possibly
    reordering them).

    For example: Keep only required arguments, or reorder params to be able to
    partialize #3 (without having to partialize #1 and #2)

    Note: A more general version would allow include and exclude to be expressed as
    functions that apply to one or several properties of the params
    (name, kind, default, annotation).
    """
    sig = Sig(func)
    include = _resolve_inclusion(include, exclude, sig.names)
    return Ingress(inner_sig=sig, outer_sig=sig[include])


@double_up_as_factory
def include_exclude(func=None, *, include=None, exclude=None):
    """Reorder and/or remove parameters.

    >>> def foo(a, b, c='C', d='D'):
    ...     print(f"{a=},{b=},{c=},{d=}")
    >>> bar = include_exclude(foo, include='b a', exclude='c d')

    The signature of ``bar`` has only ``b`` and ``a``, in that order:

    >>> from inspect import signature
    >>> str(signature(bar))
    '(b, a)'

    But the function still works and does the same thing:

    >>> bar('B', 'A')
    a='A',b='B',c='C',d='D'

    """
    return wrap(func, ingress=include_exclude_ingress_factory(func, include, exclude))


# TODO: Not working completely with allow_removal_of_non_defaulted_params=True
#  See https://github.com/i2mint/i2/issues/44
@double_up_as_factory
def rm_params(
    func=None, *, params_to_remove=(), allow_removal_of_non_defaulted_params=False
):
    """Get a function with some parameters removed.

    >>> from inspect import signature
    >>> def func(x, y=1, z=2):
    ...     return x + y * z
    >>>
    >>> f = rm_params(func, params_to_remove='z')
    >>> assert f(3) == func(3) == 5
    >>> assert f(3, 4) == func(3, 4) == 11
    >>> str(signature(f))
    '(x, y=1)'
    >>>
    >>> f = rm_params(func, params_to_remove='y z')
    >>> assert f(3) == func(3) == 5
    >>> str(signature(f))
    '(x)'

    But ``rm_params`` won't let you remove params that don't have defaults.

    >>> f = rm_params(func, params_to_remove='x z')
    Traceback (most recent call last):
    ...
    AssertionError: Some of the params you want to remove don't have defaults: {'x'}

    """
    if isinstance(params_to_remove, str):
        params_to_remove = params_to_remove.split()
    sig = Sig(func)
    params_to_remove_that_do_not_have_defaults = set(params_to_remove) & set(
        sig.without_defaults.names
    )
    if not allow_removal_of_non_defaulted_params:
        assert not params_to_remove_that_do_not_have_defaults, (
            f"Some of the params you want to remove don't have defaults: "
            f'{params_to_remove_that_do_not_have_defaults}'
        )

    return wrap(
        func, ingress=include_exclude_ingress_factory(func, exclude=params_to_remove)
    )


#     new_sig = sig - params_to_remove
#     return new_sig(func)


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


# ---------------------------------------------------------------------------------------
# Utils to help define value conversions in ingress functions


def convert_dict_values(to_convert: dict, key_to_conversion_function: dict):
    for k, v in to_convert.items():
        if k in key_to_conversion_function:
            conversion_func = key_to_conversion_function[k]
            yield k, conversion_func(v)  # converted kv pair
        else:
            yield k, v  # unconverted kv pair


# TODO: Test for performance and ask about readability
def _alt_convert_dict_values(to_convert: dict, key_to_conversion_function: dict):
    for k, v in to_convert.items():
        conversion_func = key_to_conversion_function.get(k, lambda x: x)
        yield k, conversion_func(v)


def params_used_in_funcs(funcs):
    return {name for func in funcs for name in Sig(func).names}


def required_params_used_in_funcs(funcs):
    return {name for func in funcs for name in Sig(func).required_names}


# TODO: Should we add some explicit/validation/strict options?
def kwargs_trans(
    kwargs: dict = None, /, _recursive=False, _inplace=False, **key_and_val_func
):
    """Transform a kwargs dict or build a transformer.

    :param kwargs: The dict containing the input kwargs that we will transform
    :param _recursive: Whether the transformations listed in ``key_and_val_func`` should
        be applied "recursively". When set to ``False`` (default), each transformation
        function applies to the original ``kwargs``, not the one that was transformed,
        so far.
    :param _inplace: If set to ``False`` will make a shallow copy of the ``kwargs``
        before transforming it (only relevant if ``_recursive=True``
    :param key_and_val_func: The ``key=val_func`` pairs that indicate that a
        ``val_func`` should be applied to the ``kwargs``, maching the argument names of
        the ``val_func`` to the keys of ``kwargs`` and extracting the values found
        therein to use for the corresponding inputs of that ``val_func``.
    :return: The transformed kwargs.

    >>> d = dict(a=1, b=2, c=3)
    >>> kwargs_trans(
    ...     d,
    ...     a=lambda a: a * 10,
    ...     b=lambda a, b: a + b
    ... )
    {'a': 10, 'b': 3, 'c': 3}

    See that ``d`` is unchanged here (transformation is not in place).

    >>> d
    {'a': 1, 'b': 2, 'c': 3}

    Typically you'll use ``kwargs_trans`` as a factory:

    >>> trans = kwargs_trans(a=lambda a: a * 10, b=lambda a, b: a + b)
    >>> trans(d)
    {'a': 10, 'b': 3, 'c': 3}

    Here we'll demo what the ``_recursive`` and ``_inplace`` arguments do.

    >>> from functools import partial
    >>> re_kwargs_trans = partial(kwargs_trans, _recursive=True, _inplace=True)
    >>> d = dict(a=1, b=2, c=3)
    >>>
    >>> re_kwargs_trans(
    ...     d,
    ...     a=lambda a: a * 10,
    ...     b=lambda a, b: a + b
    ...     # since _recursive=True, the a that is used is the new a = 10, not a = 1:
    ... )
    {'a': 10, 'b': 12, 'c': 3}

    Since ``_inplace=True``, ``d`` itself has changed:

    >>> d
    {'a': 10, 'b': 12, 'c': 3}

    Sometimes you'll pipe several transformers together:

    >>> from i2 import Pipe
    >>> trans = Pipe(
    ...     re_kwargs_trans(a=lambda a: a / 10),
    ...     re_kwargs_trans(c=lambda a,b,c: a * b * c),
    ...     # and then compute a new value of a using a and c:
    ...     re_kwargs_trans(a=lambda a, c: c - 1),
    ... )
    >>> trans(d)
    {'a': 35.0, 'b': 12, 'c': 36.0}
    """
    if kwargs is None:
        return partial(
            kwargs_trans, _recursive=_recursive, _inplace=_inplace, **key_and_val_func
        )
    if not _recursive:
        new_kwargs = dict()
        for key, val_func in key_and_val_func.items():
            new_kwargs[key] = _call_forgivingly(val_func, (), kwargs)
        return dict(kwargs, **new_kwargs)
    else:
        if _inplace is False:
            kwargs = kwargs.copy()
        for key, val_func in key_and_val_func.items():
            kwargs[key] = _call_forgivingly(val_func, (), kwargs)
        return kwargs


# ---------------------------------------------------------------------------------------

empty = Parameter.empty


def camelize(s):
    """
    >>> camelize('camel_case')
    'CamelCase'
    """
    return ''.join(ele.title() for ele in s.split('_'))


def kwargs_trans_to_extract_args_from_attrs(
    outer_kwargs: dict, attr_names=(), obj_param='self'
):
    self = outer_kwargs.pop(obj_param)
    arguments_extracted_from_obj = {name: getattr(self, name) for name in attr_names}
    # The kwargs we need are the union of the extracted arguments with the remaining outer_kwargs
    return dict(arguments_extracted_from_obj, **outer_kwargs)


# TODO: kind lost here, only 3.10 offers dataclasses with some control over kind:
#   See: https://stackoverflow.com/questions/49908182/how-to-make-keyword-only-fields-with-dataclasses
def param_to_dataclass_field_tuple(param: Parameter):
    return param.name, param.annotation, param.default


MethodFunc = Callable


def func_to_method_func(
    func,
    instance_params=(),
    *,
    method_name=None,
    method_params=None,
    instance_arg_name='self',
) -> MethodFunc:
    """Get a 'method function' from a 'normal function'. Also known as "methodize".

    That is, get a function that gives the same outputs as the 'normal function',
    except that some of the arguments are sourced from the attributes of the first
    argument.

    The intended use case is when you want to inject one or several methods in a class
    or instance, sourcing some of the arguments of the underlying function from a
    common pool: The attributes of the instance.

    Consider the following function involving four parameters: ``a, b, c`` and ``d``.

    >>> def func(a, b: int, c=2, *, d='bar'):
    ...     return f"{d}: {(a + b) * c}"
    >>> func(1, 2, c=3, d='hello')
    'hello: 9'

    If we wanted to make an equivalent "method function" that would source it's ``a``
    and it's ``c`` from the first argument's (in practice this first argument will be
    and instance of the class the method will be bound to), we can do so like so:

    >>> method_func = func_to_method_func(func, 'a c')
    >>> from inspect import signature
    >>> str(signature(method_func))
    "(self, b: int, *, d='bar')"

    Note that the first argument is ``self`` (default name for an "instance"),
    that ``a`` and ``c`` are not there, but that the two remaining parameters,
    ``b`` and ``d`` are present, in the same order, and with the same annotations and
    parameter kind (the ``d`` is still keyword-only).

    Now let's make a dummy object that has attributes ``a`` and a ``c``, and use it to
    call ``method_func``:

    >>> from collections import namedtuple
    >>> instance = namedtuple('FakeInstance', 'a c')(1, 3)
    >>> method_func(instance, 2, d='hello')
    'hello: 9'

    Which is:

    >>> assert method_func(instance, 2, d='hello') == func(1, 2, c=3, d='hello')

    Often, though, what you'll want is to include this method function directly in
    a class, as you're making that class "normally". That works too:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Klass:
    ...     a : int = 1
    ...     c : int = 3
    ...     method_func = func_to_method_func(func, 'a c')
    >>> instance = Klass(1, 3)
    >>> instance.method_func(2, d='hello')
    'hello: 9'

    What if your function has argument names that don't correspond to the names you
    have, or want, as attributes of the class? Or even, you have several functions that
    share an argument name that need to be bound to a different attribute?

    For that, just use ``map_names`` to wrap the function, giving it the names that
    you need to give it to have the effect you want (the binding of those arguments
    to attributes of the instance):

    >>> from i2.wrapper import ch_names
    >>> def func(x, y: int, z=2, *, d='bar'):
    ...     return f"{d}: {(x + y) * z}"
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Klass:
    ...     a : int = 1
    ...     c : int = 3
    ...     method_func = func_to_method_func(ch_names(func, x='a', z='c'), 'a c')
    >>> instance = Klass(1, 3)
    >>> instance.method_func(2, d='hello')
    'hello: 9'

    """
    # get a signature object for func
    sig = Sig(func)
    # if method_name not give, use the function's name
    method_name = method_name or sig.name
    # if params expressed as string, split it into a list of parameter (names)
    if isinstance(instance_params, str):
        instance_params = instance_params.split()
    if isinstance(method_params, str):
        method_params = method_params.split()
    # if method_params is not given, take those parameters that aren't in instance_params
    method_params = method_params or tuple(
        name for name in sig.names if name not in instance_params
    )
    # the Sig object of the method: instance name followed with method_params
    method_sig = instance_arg_name + Sig(func)[method_params]
    # make the ingress function that will map method_sig's interface to sig's.
    ingress = Ingress(
        inner_sig=sig,
        # inside, foo will be doing the work, so need to map to its signature
        kwargs_trans=partial(
            # this is how to transform outer (args, kwargs) to inner ones
            kwargs_trans_to_extract_args_from_attrs,
            attr_names=instance_params,
            obj_param=instance_arg_name,
        ),
        outer_sig=method_sig,  # this is the signature we want at the interface
    )
    # wrap the function, name it and return it
    method_func = wrap(func, ingress=ingress)
    method_func.__name__ = method_name
    return method_func


# TODO: See bind_funcs_object_attrs_old and get rid of it if bind_funcs_object_attrs
#  is better and has all old has.
# TODO: Make instances picklable (class is already, but not instances!)!!
def bind_funcs_object_attrs(
    funcs, init_params=(), *, cls=None, module=None, **extra_attrs
):
    """Transform one or several functions into a class that contains them as methods
    sourcing specific arguments from the instance's attributes.

    >>> from inspect import signature
    >>> from dataclasses import dataclass
    >>>
    >>> def foo(a, b, c=2, *, d='bar'):
    ...     return f"{d}: {(a + b) * c}"
    >>> foo(1, 2)
    'bar: 6'
    >>> Klass = bind_funcs_object_attrs(foo, init_params='a c')
    >>> Klass.__name__
    'Foo'
    >>> instance = Klass(a=1, c=3)
    >>> assert instance.foo(2, d='hello') == 'hello: 9' == foo(
    ...     a=1, b=2, c=3, d='hello')
    >>> str(signature(Klass))
    '(a, c=2) -> None'
    >>>
    >>> instance = Klass(a=1, c=3)
    >>> str(instance)
    'Foo(a=1, c=3)'
    >>> str(signature(instance.foo))
    "(b, *, d='bar')"
    >>> instance.foo(2, d='hello')
    'hello: 9'
    >>> instance.foo(10, d='goodbye')
    'goodbye: 33'

    >>> def foo(a, b, c):
    ...     return a + b * c
    ...
    >>> def bar(d, e):
    ...     return f"{d=}, {e=}"
    ...
    >>> @dataclass
    ... class K:
    ...     a: int
    ...     e: int
    ...
    >>> C = bind_funcs_object_attrs([foo, bar], 'a e', cls=K)
    >>> str(signature(C))
    '(a: int, e: int) -> None'
    >>> c = C(1,2)
    >>> assert str(signature(c.foo)) == '(b, c)'
    >>> c.foo(3,4)
    13
    >>> assert str(signature(c.bar)) == '(d)'
    >>> c.bar(5)
    'd=5, e=2'
    """

    if isinstance(init_params, str):
        init_params = init_params.split()

    dflt_cls_name = 'FuncsUnion'
    if callable(funcs) and not isinstance(funcs, Iterable):
        single_func = funcs
        dflt_cls_name = camelize(getattr(single_func, '__name__', dflt_cls_name))
        funcs = [single_func]

    # If cls not given, make one from the funcs and init_params
    if not isinstance(cls, type):
        # if the class is not given, we need to make one
        # ... for this we make a name
        if isinstance(cls, str):
            cls_name = cls
        else:
            cls_name = dflt_cls_name
        # ... and an actual class
        cls = _mk_base_class_for_funcs(cls_name, init_params, funcs)

    for func in funcs:
        method_func = func_to_method_func(func, init_params)
        setattr(cls, method_func.__name__, method_func)

    if module:
        cls.__module__ = module
    for attr_name, attr_val in extra_attrs.items():
        setattr(cls, attr_name, attr_val)

    return cls


class PickleHelperMixin:
    def __reduce__(self):
        return self.__name__


def _mk_base_class_for_funcs(cls_name, init_params, funcs):
    """
    Make a class with given init_params.

    :param cls_name: The name the class should have
    :param init_params: list of strings used to determine what arg names are in the init
    :param funcs: list of functions used to add defaults and annotations to the init args
    :return:
    """
    # Make the signature for the init
    class_init_sig = Sig(init_params)
    if funcs:
        for func in funcs:
            init_params_in_func_sig = list(
                filter(None, map(Sig(func).get, init_params))
            )
            class_init_sig = class_init_sig.merge_with_sig(
                init_params_in_func_sig,
                default_conflict_method='fill_defaults_and_annotations',
            )

    dataclass_fields = list(map(param_to_dataclass_field_tuple, class_init_sig.params))
    cls = make_dataclass(cls_name, dataclass_fields, bases=(PickleHelperMixin,))

    return cls


def bind_funcs_object_attrs_old(
    funcs, init_params=(), *, cls=None,
):
    """Transform one or several functions into a class that contains them as methods
    sourcing specific arguments from the instance's attributes.
    >>> from inspect import signature
    >>> from dataclasses import dataclass
    >>>
    >>> def foo(a, b, c=2, *, d='bar'):
    ...     return f"{d}: {(a + b) * c}"
    >>> foo(1, 2)
    'bar: 6'
    >>> Klass = bind_funcs_object_attrs_old(foo, init_params='a c')
    >>> Klass.__name__
    'Foo'
    >>> instance = Klass(a=1, c=3)
    >>> assert instance.foo(2, d='hello') == 'hello: 9' == foo(
    ...     a=1, b=2, c=3, d='hello')
    >>> str(signature(Klass))
    '(a, c=2) -> None'
    >>>
    >>> instance = Klass(a=1, c=3)
    >>> str(instance)
    'Foo(a=1, c=3)'
    >>> str(signature(instance.foo))
    "(b, *, d='bar')"
    >>> instance.foo(2, d='hello')
    'hello: 9'
    >>> instance.foo(10, d='goodbye')
    'goodbye: 33'
    >>> def foo(a, b, c):
    ...     return a + b * c
    ...
    >>> def bar(d, e):
    ...     return f"{d=}, {e=}"
    ...
    >>> @dataclass
    ... class K:
    ...     a: int
    ...     e: int
    ...
    >>> C = bind_funcs_object_attrs([foo, bar], 'a e', cls=K)
    >>> str(signature(C))
    '(a: int, e: int) -> None'
    >>> c = C(1,2)
    >>> assert str(signature(c.foo)) == '(b, c)'
    >>> c.foo(3,4)
    13
    >>> assert str(signature(c.bar)) == '(d)'
    >>> c.bar(5)
    'd=5, e=2'
    """

    if isinstance(init_params, str):
        init_params = init_params.split()

    dflt_cls_name = 'FuncsUnion'
    if callable(funcs) and not isinstance(funcs, Iterable):
        single_func = funcs
        dflt_cls_name = camelize(getattr(single_func, '__name__', dflt_cls_name))
        funcs = [single_func]

    if not isinstance(cls, type):
        # if the class is not given, we need to make one
        if isinstance(cls, str):
            cls_name = cls
        else:
            cls_name = dflt_cls_name

        # init_parameter_objects = Sig(func)[init_params].params
        # Make the signature for the init
        class_init_sig = Sig()
        for func in funcs:
            class_init_sig = class_init_sig.merge_with_sig(func)[init_params]

        dataclass_fields = list(
            map(param_to_dataclass_field_tuple, class_init_sig.params)
        )
        cls = make_dataclass(cls_name, dataclass_fields)

    for func in funcs:
        method_func = func_to_method_func(func, init_params)
        setattr(cls, method_func.__name__, method_func)
    return cls


def _items_filt(d: dict, keys):
    for k, v in d.items():
        if k in keys:
            yield k, v


def _mk_sig_from_params_and_funcs(params, funcs):
    def gen():
        for param in params:
            pass


## An attempt to redo a func_to_method_func because I forgot func_to_method_func existed
## Has some different ideas, so keeping around until I decide it's time to let go.
# NoSuchKey = type('NoSuchKey', (), {})
# _instance_extractor: KwargsTrans
#
# # TODO: Add more (possibly optional) bind validation to fail early.
# def _instance_extractor(
#     outer_kwargs, bind: IdentifierMapping = (), instance_param: Identifier = 'self'
# ):
#     """
#
#     :param outer_kwargs: The input/outer keyword arguments
#     :param bind: The inner->outer param mapping that defines what we want to extract
#     :param instance_param: The name of the outer_kwargs key that contains the 'instance'
#         from which we'll extract the bound
#     :return:
#     """
#     """A KwargsTrans that extracts need arguments from one of the 'instance'
#     outer_kwargs values.
#
#
#     """
#     # Compute the inverse {outer:inner,...} of {inner:outer,...} bind
#     inv_bind = invert_map(bind)
#     outer_kwargs = outer_kwargs.copy()
#     instance = outer_kwargs.pop(instance_param)  # TODO: Better error handing
#
#     def gen():
#         for outer_param, outer_val in outer_kwargs.items():
#             if inner_param := inv_bind.get(outer_param, NoSuchKey) is not NoSuchKey:
#                 # if outer_param was bound, the bound inner_param should be tied to
#                 # the instance's attribute
#                 yield inner_param, getattr(instance, outer_val)
#             else:
#                 # take the arg name and val as is
#                 yield outer_param, outer_val
#
#     return dict(gen())

#
# def methodize(func, bind: Bind = ()):
#     bind = identifier_mapping(bind)
#     ingress = Ingress(
#         outer_sig=Sig(func),
#         kwargs_trans=partial(_instance_extractor, bind=bind),
#         inner_sig=Sig('self') + Sig(func) - Sig(list(bind)),  # TODO: solve type or lint
#     )
#     return wrap(func, ingress=ingress)


# ---------------------------------------------------------------------------------------
# Extended Wrapper class


class WrapperValidationError(ValueError):
    """Raised when wrapper some construction params are not valid"""


class EgressValidationError(WrapperValidationError):
    """Raised when a egress is not valid"""


class IngressValidationError(WrapperValidationError):
    """Raised when a ingress is not valid"""


class CallerValidationError(WrapperValidationError):
    """Raised when a caller is not valid"""


def _default_ingress(*args, **kwargs):
    return args, kwargs


def _default_egress(output, **egress_params):
    return output


def _default_caller(func, args, kwargs):
    return func(*args, **kwargs)


_keyword_kinds = {Sig.KEYWORD_ONLY, Sig.VAR_KEYWORD}


def _all_kinds_are_keyword_only_or_variadic_keyword(sig):
    return all(kind in _keyword_kinds for kind in list(sig.kinds.values())[3:])


# TODO: Factor out more common parts with Wrap and reuse (possibly through _Wrap)
class Wrapx(_Wrap):
    def __init__(self, func, ingress=None, egress=None, *, caller=None, name=None):
        """An extended wrapping object that allows more complex wrapping mechanisms.

        :param func: The wrapped function
        :param ingress: The incoming data transformer. It determines the argument properties
            (name, kind, default and annotation) as well as the actual input of the
            wrapped function.
        :param egress: The outgoing data transformer. It also takes precedence over the
            wrapped function to determine the return annotation of the ``Wrap`` instance
        :param caller: A caller defines what it means to call the ``func`` on the
            arguments it is given. It should be of the form
            ``caller(func, args, kwargs, *, ...extra_keyword_only_params)``.
            By default, the caller will simply return ``func(*args, **kwargs)``.
        :param name: Name to give the wrapper (will use wrapped func name by default)

        :return: A callable instance wrapping ``func``

        >>> from inspect import signature
        >>>
        >>> def func(x, y):
        ...     return x + y
        ...
        >>> def save_on_output_egress(v, *, k, s):
        ...     s[k] = v
        ...     return v
        ...
        >>> save_on_output = Wrapx(func, egress=save_on_output_egress)
        >>> # TODO: should be `(x, y, *, k, s)` --> Need to work on the merge for this.
        >>> str(signature(save_on_output))
        '(x, y, k, s)'
        >>>
        >>> store = dict()
        >>> save_on_output(1, 2, k='save_here', s=store)
        3
        >>> assert save_on_output(1, 2, k='save_here', s=store) == 3 == func(1, 2)
        >>> store  # see what's in the store now!
        {'save_here': 3}

        A caller is meant to control the way the function is called.
        It is given the ``func`` and the ``func_args`` and ``func_kwargs``
        (whatever the ingress function gives it, if present) and possibly additional
        params and will return... well, what ever you tell it to.

        This can be used, for example, to call the function in a subprocess,
        or on a remote system, differ computation (command pattern, for example, using
        ``functools.partial``, or do what ever needs to have a view both on the function
        and its inputs.

        Here, we will wrap the function so it will apply to an iterable of inputs
        (of the first argument), returning a list of results

        >>> def func(x, y=2):
        ...     return x + y
        ...
        >>> def iterize(func, args, kwargs):
        ...     first_arg_val = next(iter(kwargs.values()))
        ...     return list(map(func, first_arg_val))
        ...
        >>> iterized_func = Wrapx(func, caller=iterize)
        >>> iterized_func([1, 2, 3, 4])
        [3, 4, 5, 6]

        Let's do the same as above, but allow other variables (here ``y``) to be input as
        well. This takes a bit more work...

        >>> from functools import partial
        >>> def _iterize_first_arg(func, args, kwargs):
        ...     first_arg_name = next(iter(kwargs))
        ...     remaining_kwargs = {
        ...         k: v for k, v in kwargs.items() if k != first_arg_name
        ...     }
        ...     return list(
        ...         map(partial(func, **remaining_kwargs), kwargs[first_arg_name])
        ...     )

        Let's demo a different way of using Wrapx: Making a wrapper to apply at
        function definition time

        >>> iterize_first_arg = partial(Wrapx, caller=_iterize_first_arg)
        >>> @iterize_first_arg
        ... def func(x, y):
        ...     return x + y
        >>>
        >>> func([1, 2, 3, 4], 10)
        [11, 12, 13, 14]


        """
        super().__init__(func, ingress, egress, caller=caller, name=name)
        self.ingress, self.egress, self.caller, self.sig = _process_wrapx_params(
            func, ingress, egress, caller
        )

        self.__signature__ = self.sig
        self.__wrapped__ = func
        # TODO: Pros and cons analysis of pointing __wrapped__ to func. partial uses
        #  .func, but wraps looks for __wrapped__

    def __call__(self, *args, **kwargs):
        try:
            _kwargs = self.sig.kwargs_from_args_and_kwargs(args, kwargs)
            # TODO: Consider call_forgivingly(self.ingress, *args, **kwargs)
            #  because call_forgivingly(self.ingress, **_kwargs) doesn't allow
            #  same ingress functions to be used for Wrap and Wrapx
            func_args, func_kwargs = call_forgivingly(self.ingress, **_kwargs)
            inner_output = call_forgivingly(
                self.caller, self.func, func_args, func_kwargs
            )
            return call_forgivingly(self.egress, inner_output, **_kwargs)
        except Exception as e:
            # Try running again, but with more careful validation, to try to give
            # more specific error messages
            # We don't do this in the first run so that we don't incur the validation
            # overhead on every call.
            _kwargs = self.sig.kwargs_from_args_and_kwargs(args, kwargs)
            func_args, func_kwargs = _validate_ingress_output(
                call_forgivingly(self.ingress, **_kwargs)
            )
            inner_output = call_forgivingly(self.func, *func_args, **func_kwargs)
            return call_forgivingly(self.egress, inner_output, **_kwargs)


def _validate_ingress_output(ingress_output):
    if (
        not isinstance(ingress_output, tuple)
        or not len(ingress_output) == 2
        or not isinstance(ingress_output[0], tuple)
        or not isinstance(ingress_output[1], dict)
    ):
        raise IngressValidationError(
            f'An ingress function should return a (tuple, dict) pair. '
            f'This ingress function returned: {ingress_output}'
        )
    return ingress_output


def _process_wrapx_params(func, ingress, egress, caller):
    func_sig = Sig(func)

    ingress, ingress_sig = _init_ingress(func_sig, ingress)
    egress, egress_sig = _init_egress(func_sig, egress)
    caller, caller_sig = _init_caller(caller)

    sig = _mk_composite_sig(ingress_sig, egress_sig, caller_sig)
    return ingress, egress, caller, sig


def _mk_composite_sig(ingress_sig, egress_sig, caller_sig):
    egress_sig_minus_first_arg = egress_sig - egress_sig.names[0]
    caller_sig_minus_three_first_args = caller_sig - caller_sig.names[:3]
    sig = Sig(
        ingress_sig + egress_sig_minus_first_arg + caller_sig_minus_three_first_args,
        return_annotation=egress_sig.return_annotation,
    )
    return sig


def _init_caller(caller):
    if caller is None:
        caller = _default_caller
        caller_sig = Sig('func args kwargs')  # sig with three inputs
    else:
        caller_sig = Sig(caller)
        if len(caller_sig) < 3:
            raise CallerValidationError(
                f'A caller must have at least three arguments: '
                f'{caller} signature was {caller_sig}'
            )
        if not _all_kinds_are_keyword_only_or_variadic_keyword(caller_sig):
            raise CallerValidationError(
                f'A caller must have at least three arguments'
                f'{caller} signature was {caller_sig}'
            )
    return caller, caller_sig


def _init_egress(func_sig, egress):
    if egress is None:
        egress = _default_egress
        # signature with a single 'output' arg and func_sig's return_annotation
        egress_sig = Sig('output', return_annotation=func_sig.return_annotation)
        return_annotation = func_sig.return_annotation
    else:
        egress_sig = Sig(
            egress,
            # if egress has no return_annotation, use the func_sig's one.
            # TODO: Is this really correct/safe? What if egress changes the type?
            return_annotation=Sig(egress).return_annotation
            or func_sig.return_annotation,
        )

    return egress, egress_sig


def _init_ingress(func_sig, ingress):
    if ingress is None:
        ingress = _default_ingress
        ingress_sig = func_sig
    else:
        ingress_sig = Sig(ingress)
    return ingress, ingress_sig


# ---------------------------------------------------------------------------------------
# partialx


def partialx(
    func, *args, __name__=None, _rm_partialize=False, _allow_reordering=False, **kwargs
):
    """
    Extends the functionality of builtin ``functools.partial`` with the ability to

    - set ``__name__ ``

    - remove partialized arguments from signature

    - reorder params (so that defaults are at the end)

    >>> def f(a, b=2, c=3):
    ...     return a + b * c
    >>> curried_f = partialx(f, c=10, _rm_partialize=True)
    >>> curried_f.__name__
    'f'
    >>> from inspect import signature
    >>> str(signature(curried_f))
    '(a, b=2)'

    >>> def f(a, b, c=3):
    ...     return a + b * c

    Note that ``a`` gets a default, but ``b`` does not, yet is after ``a``.
    This is allowed because these parameters all became KEYWORD_ONLY.

    >>> g = partialx(f, a=1)
    >>> str(Sig(g))
    '(*, a=1, b, c=3)'

    If you wanted to reorder the parameters to have all defaulted kinds be at the end,
    as usual, you can do so using ``_allow_reordering=True``

    >>> g = partialx(f, a=1, _allow_reordering=True)
    >>> str(Sig(g))
    '(*, b, a=1, c=3)'

    """
    f = partial(func, *args, **kwargs)
    if _rm_partialize:
        sig = Sig(func)
        partialized = list(
            sig.kwargs_from_args_and_kwargs(args, kwargs, allow_partial=True)
        )
        sig = sig - partialized
        f = sig(partial(f, *args, **kwargs))
    if _allow_reordering:
        # TODO: Instead of Sig(f).defaults, move only params that need to move
        # TODO: + Change signature so that the number of params that become keyword-only
        #   is minimize.
        f = move_params_to_the_end(f, Sig(f).defaults)
    f.__name__ = __name__ or name_of_obj(func)
    return f


def move_params_to_the_end(
    func: Callable, names_to_move: Union[Callable, Iterable[str]]
):
    """
    Choose args from func, according to choice_args_func and move them
    to the right

    >>> from functools import partial
    >>> from i2 import Sig
    >>> def foo(a, b, c):
    ...     return a + b + c
    >>> g = partial(foo, b=4)  # fixing a, which is before b
    >>> h = move_params_to_the_end(g, Sig(g).defaults)
    >>> assert str(Sig(g)) == '(a, *, b=4, c)'
    >>> assert str(Sig(h)) == '(a, *, c, b=4)'

    """
    if callable(names_to_move):
        names_to_move = names_to_move(func)
    assert isinstance(names_to_move, Iterable), (
        f'names_to_move must be an iterable of names '
        f'or a callable producing one from a function. Was {names_to_move}'
    )

    names = Sig(func).names
    reordered = move_names_to_the_end(names, names_to_move)
    wrapped_func = include_exclude(func, include=reordered)
    return wrapped_func


def move_names_to_the_end(names, names_to_move_to_the_end):
    """
    Remove the items of ``names_to_move_to_the_end`` from ``names``
    and append to the right of names

    >>> names = ['a','c','d','e']
    >>> names_to_move_to_the_end = ['c','e']
    >>> move_names_to_the_end(names, names_to_move_to_the_end)
    ['a', 'd', 'c', 'e']
    >>> names_to_move_to_the_end = 'c e'
    >>> move_names_to_the_end(names, names_to_move_to_the_end)
    ['a', 'd', 'c', 'e']

    """
    if isinstance(names_to_move_to_the_end, str):
        names_to_move_to_the_end = names_to_move_to_the_end.split()
    else:
        names_to_move_to_the_end = list(names_to_move_to_the_end)
    removed = [x for x in names if x not in names_to_move_to_the_end]
    return list(removed) + names_to_move_to_the_end


# --------------------------------------------------------------------------------------
# smart defaults

# IDEA: Could make SmartDefault have an out and __call__, working like a meshed.FuncNode
@dataclass
class SmartDefault:
    func_computing_default: Callable
    original_default: Any = empty

    def __repr__(self):
        func = self.func_computing_default
        func_name = getattr(func, '__name__', str(func))
        if self.original_default is empty:
            return f'SmartDefault({func_name})'
        else:
            dflt = self.original_default
            if isinstance(dflt, str):
                return f"SmartDefault({func_name}, '{dflt}')"
            else:
                return f'SmartDefault({func_name}, {dflt})'


# PATTERN: Yet another "meshed dict completion"
# IDEA: Could make SmartDefault have an out and __call__, working like a meshed.FuncNode
def complete_dict_applying_functions(
    d: dict, /, _only_if_name_missing=True, _allow_overwrites=False, **func_for_name
):
    """Complete dict ``d`` by applying function to variables in ``d``, sequentially.

    That is, doing ``d[name] = func(**d)`` for all ``name, func in d.items()``.

    Set ``_allow_overwrites=True`` to allow overwrites.

    Set ``_only_if_name_missing=False`` to apply all functions of ``func_for_name``
    regardless if the ``name`` already exists in ``d`` or not.

    >>> func_for_name = dict(
    ...     b=lambda a: a * 10, c=lambda a, b: a + b, d=lambda c: c * 2
    ... )
    >>> complete_dict_applying_functions(dict(a=1), **func_for_name)
    {'a': 1, 'b': 10, 'c': 11, 'd': 22}

    Notice that when ``b`` is present in input ``dict``, it's value is conserved.
    That is, the ``b`` of ``func_for_name`` isn't applied to compute it.

    >>> complete_dict_applying_functions(dict(a=1, b=2), **func_for_name)
    {'a': 1, 'b': 2, 'c': 3, 'd': 6}

    If you specify ``_only_if_name_missing=False``,
    ``complete_dict_applying_functions`` will try to compute everything
    ``func_for_name`` tells it too, regardless if the input dictionary contains the
    key or not, resulting in an error:

    >>> complete_dict_applying_functions(
    ...     dict(a=1, b=2), **func_for_name, _only_if_name_missing=False
    ... )
    Traceback (most recent call last):
      ...
    i2.errors.OverwritesNotAllowed: You're not allowed to overwrite to the values of b

    If you want, on the other hand, to allow overwrites, you can do so specifying
    ``_allow_overwrites=True``:

    >>> complete_dict_applying_functions(
    ...     dict(a=1, b=2), **func_for_name,
    ...     _only_if_name_missing=False, _allow_overwrites=True
    ... )
    {'a': 1, 'b': 10, 'c': 11, 'd': 22}

    """

    if _only_if_name_missing:
        func_for_name = {
            name: func for name, func in func_for_name.items() if name not in d
        }
    elif not _allow_overwrites and not func_for_name.keys().isdisjoint(d):
        raise OverwritesNotAllowed.for_keys(func_for_name.keys() & d)

    for name, func in func_for_name.items():
        # TODO: Can optimize using under-the-hood of call_forgivingly
        #  (extracting from _kwargs directly)
        # complete the keyword arguments with defaults computed from existing arguments
        d[name] = call_forgivingly(func, **d)

    return d


def _compute_new_sigs(func, /, **smart_defaults):
    original_sig = Sig(func)
    return original_sig.ch_defaults(
        **{
            name: SmartDefault(
                func_computing_default=func,
                original_default=original_sig.defaults.get(name, empty),
            )
            for name, func in smart_defaults.items()
        }
    )


@double_up_as_factory
def add_smart_defaults(
    func=None, *, _only_if_name_missing=True, _allow_overwrites=False, **smart_defaults,
):
    """Add smart defaults to function.

    Smart defaults compute defaults of inputs based on the other given inputs.

    :param func: The function to wrap.
    :param smart_defaults: ``input_arg=function_to_compute_it_from_other_args`` where
        the function's argument names must match the argument names of ``func``, the
        wrapped function.
    :return:

    >>> def xyz_sum(x, y, z='c'):  return x + y + z
    >>> def times_two(x):  return x * 2
    >>> def just_z():  return 'Z'
    >>> f = add_smart_defaults(xyz_sum, y=times_two, z=just_z)
    >>>
    >>> f('a', 'b', 'c')
    'abc'
    >>> f('a', 'b')
    'abZ'
    >>> f('a')
    'aaaZ'
    >>> f
    <i2.Wrap xyz_sum(x, y=SmartDefault(times_two), z=SmartDefault(just_z, 'c'))>

    """
    names_not_in_func_arguments = smart_defaults.keys() - Sig(func).names
    assert (
        not names_not_in_func_arguments
    ), f"These weren't argument names of the {func} function: {names_not_in_func_arguments}"
    kwargs_trans = partial(
        complete_dict_applying_functions,
        _only_if_name_missing=_only_if_name_missing,
        _allow_overwrites=_allow_overwrites,
        **smart_defaults,
    )
    smart_defaults_ingress = Ingress(
        inner_sig=Sig(func),
        kwargs_trans=kwargs_trans,
        outer_sig=_compute_new_sigs(func, **smart_defaults),
        apply_defaults=False,
    )

    return smart_defaults_ingress.wrap(func)

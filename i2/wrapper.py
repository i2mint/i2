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
from typing import Mapping, Callable, Optional
from types import MethodType

from i2.signatures import Sig
from i2.multi_object import Pipe

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

    An ingress function links the interface of the wrapper to the interface of the
    wrapped func; therefore it's definition often depends on information of both,
    and for that reason, we provide the ability to specify the ingress not only
    explicitly (as in the examples above), but through a factory -- a function that
    will be called on `func` to produce the ingress that should be used to wrap it.



    .. seealso::

        ``wrap`` function.

    """

    def __init__(self, func, ingress=None, egress=None):
        self.func = func
        wraps(func)(self)  # TODO: should we really copy everything by default?

        # remember the actual value of ingress and egress (for reduce to reproduce)
        self._ingress = ingress
        self._egress = egress

        ingress_sig = Sig(func)

        if ingress is None:
            self.ingress = transparent_ingress
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

        return_annotation = empty

        if egress is None:
            self.egress = transparent_egress
        else:
            self.egress = egress
            egress_return_annotation = Sig(egress).return_annotation
            if egress_return_annotation is not Parameter.empty:
                return_annotation = egress_return_annotation

        self.__signature__ = Sig(ingress_sig, return_annotation=return_annotation)
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

    - ``1-to-many`` (e.g. the outer 'w' is used to compute the inner ``w`` and ``x``)

    - ``many-to-1 (e.g. the outer ``x`` and ``y`` are used to compute inner ``y``)

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
    >>> wrapped_f = wrap(f, ingress)
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
    >>> wrapped_f = wrap(f, ingress)
    >>> assert wrapped_f(2, x=3, you=4) == '(w:=4) + (x:=6) * (y:=7) ** (z:=3) == 2062'

    """

    def __init__(
        self, inner_sig, kwargs_trans: Optional[KwargsTrans] = None, outer_sig=None
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
        :param outer_sig: The outer signature. The signature the ingress function
            (there for the wrapped function) will have. Also serves to convert input
            (args, kwargs) to the kwargs that will be given to kwargs_trans.

        When making an Ingress function directly, one must take care that
        ``inner_sig``, ``kwargs_trans`` and ``outer_sig`` are aligned.

        Namely, 'kwargs_trans' must be able to handle outputs of
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

    def __call__(self, *ingress_args, **ingress_kwargs):
        # Get the all-keywords version of the arguments (args,kwargs->kwargs)
        func_kwargs = self.outer_sig.kwargs_from_args_and_kwargs(
            ingress_args, ingress_kwargs, apply_defaults=True
        )

        func_kwargs = dict(
            func_kwargs,  # by default, keep the func_kwargs, but
            **self.kwargs_trans(func_kwargs),  # change those that kwargs_trans desires
        )

        # Return an (args,kwargs) pair that respects the inner function's
        # argument kind restrictions.
        # Note: Originally was with (default) allow_excess=False. Changed to True to
        #       allow more flexibility in outer sig. But is this sane? Worth it?
        # TODO: Reflect on pros/cons of allow_excess=True
        return self.inner_sig.args_and_kwargs_from_kwargs(
            func_kwargs, apply_defaults=True, allow_excess=True
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

        # Return an (args, kwargs) pair the respects the inner function's
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


def include_exclude_ingress_factory(func, include=None, exclude=None):
    """A pattern underlying any ingress that takes a subset of parameters (possibly
    reordering them).
    For example: Keep only required arguments, or reorder params to be able to
    partialize #3 (without having to partialize #1 and #2)
    Note: A more general version would allow include and exclude to be expressed as
    functions
    that apply to one or several properties of the params (name, kind, default,
    annotation).
    """
    sig = Sig(func)
    exclude = exclude or set()
    include = [x for x in (include or sig.names) if x not in exclude]

    return Ingress(inner_sig=sig, outer_sig=sig[include])


def remove_params_ingress_factory(func, params_to_remove):
    """Get a version of the function without some specific params"""
    if isinstance(params_to_remove, str):
        params_to_remove = params_to_remove.split()
    return include_exclude_ingress_factory(func, exclude=params_to_remove)


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


# TODO: Test for performance and ask about readability
def _alt_convert_dict_values(to_convert: dict, key_to_conversion_function: dict):
    for k, v in to_convert.items():
        conversion_func = key_to_conversion_function.get(k, lambda x: x)
        yield k, conversion_func(v)


from inspect import Parameter
from dataclasses import make_dataclass

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
    t = (param.name, param.annotation, param.default)
    if t[2] is empty:
        t = t[:2]
    if t[1] is empty:
        if len(t) == 2:
            t = t[0]
        else:
            t = (t[0], 'typing.Any', t[2])
    return t


MethodFunc = Callable


def func_to_method_func(
    func,
    instance_params=(),
    *,
    method_name=None,
    method_params=None,
    instance_arg_name='self',
) -> MethodFunc:
    """Get a 'method function' from a 'normal function'.

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

    >>> class Klass:
    ...     a = 1
    ...     c = 3
    >>> instance = Klass()
    >>> method_func(instance, 2, d='hello')
    'hello: 9'

    Which is:

    >>> assert method_func(instance, 2, d='hello') == func(1, 2, c=3, d='hello')

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
    method_func = wrap(func, ingress)
    method_func.__name__ = method_name
    return method_func


from typing import Iterable


def make_funcs_binding_class(
    funcs, init_params=(), cls_name=None,
):
    """Transform one or several functions into a class that contains them as methods
    sourcing specific arguments from the instance's attributes.

    >>> from inspect import signature
    >>> def foo(a, b, c=2, *, d='bar'):
    ...     return f"{d}: {(a + b) * c}"
    >>> foo(1, 2)
    'bar: 6'
    >>> Klass = make_funcs_binding_class(foo, init_params='a c')
    >>> Klass.__name__
    'Foo'
    >>> instance = Klass(a=1, c=3)
    >>> assert instance.foo(2, d='hello') == 'hello: 9' == foo(
    ...     a=1, b=2, c=3, d='hello')
    >>> str(signature(Klass))
    "(a: 'typing.Any', c: 'typing.Any' = 2) -> None"
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
    """

    dflt_cls_name = 'FuncsUnion'
    if callable(funcs) and not isinstance(funcs, Iterable):
        single_func = funcs
        dflt_cls_name = camelize(getattr(single_func, '__name__', dflt_cls_name))
        funcs = [single_func]

    cls_name = cls_name or dflt_cls_name
    if isinstance(init_params, str):
        init_params = init_params.split()

    # init_parameter_objects = Sig(func)[init_params].params
    class_init_sig = Sig()
    for func in funcs:
        class_init_sig = class_init_sig.merge_with_sig(func)[init_params]
    dataclass_fields = list(map(param_to_dataclass_field_tuple, class_init_sig.params))
    Klass = make_dataclass(cls_name, dataclass_fields)

    for func in funcs:
        method_func = func_to_method_func(func, init_params)
        setattr(Klass, method_func.__name__, method_func)
    return Klass

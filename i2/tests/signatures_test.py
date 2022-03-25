"""Test signatures module


# Notes to the reader

Both in the code and in the docs, we'll use short hands for parameter (argument) kind.
    PK = Parameter.POSITIONAL_OR_KEYWORD
    VP = Parameter.VAR_POSITIONAL
    VK = Parameter.VAR_KEYWORD
    PO = Parameter.POSITIONAL_ONLY
    KO = Parameter.KEYWORD_ONLY

"""
import pytest
from functools import reduce
from typing import Any

from i2.signatures import *
from i2.signatures import normalized_func

from i2.tests.util import sig_to_inputs, trace_call


def test_sig_wrap_edge_cases():
    """Tests some edge cases involving simultaneous changes of defaults and kinds.

    Current Sig.wrap design allows you to change defaults, and this will have the
    effect of changing the ``__defaults__`` and ``__kwdefaults__`` of the wrapped
    function.

    But, when we change the default of a parameter and change it's kind at the same
    time, bad things happen (see https://github.com/i2mint/i2/issues/16).

    So `Sig.wrap` makes a few checks and raises an error if it's thinks it's not
    safe to do the wrapping.

    Some of these facts may change with new designs. When this is the case,
    this test should be invalidated."""

    # Non edge-case tests:

    def foo(x, y, z=0):
        return x + y * z

    assert foo.__defaults__ == (0,)
    assert foo(1, 2) == 1

    @Sig(lambda x, y, z=3: None)
    def foo(x, y, z=0):
        return x + y * z

    assert foo(1, 2) == 7
    #    works because Sig also changed __defaults__:
    assert foo.__defaults__ == (3,)

    @Sig(lambda x, y, *, z=3: None)
    def foo(x, y, *, z=0):
        return x + y * z

    assert foo(1, 2) == 7
    #    works because Sig also changed __defaults__ and __kwdefaults__:
    assert foo.__defaults__ == ()
    assert foo.__kwdefaults__ == {'z': 3}

    # The following (where we go from a (same kind) param not having a default,
    # to having one, also work

    @Sig(lambda x, y, z=3: None)
    def foo(x, y, z):
        return x + y * z

    assert foo(1, 2) == 7
    assert foo(1, 2, z=10) == 21

    @Sig(lambda x, y, *, z=3: None)
    def foo(x, y, *, z):
        return x + y * z

    assert foo(1, 2) == 7
    assert foo(1, 2, z=10) == 21


def test_tuple_the_args():
    from i2.signatures import tuple_the_args

    def func(a, *args, bar):
        return trace_call(func, locals())

    assert func(1, 2, 3, bar=4) == 'func(a=1, args=(2, 3), bar=4)'

    wfunc = tuple_the_args(func)

    # here, not that (1) args is specified as one iterable ([2, 3] instead of 2,
    # 3) and (2) the function name is the same as the wrapped (func)
    assert wfunc(1, [2, 3], bar=4) == 'func(a=1, args=(2, 3), bar=4)'

    # See the func itself hasn't changed
    assert func(1, 2, 3, bar=4) == 'func(a=1, args=(2, 3), bar=4)'

    assert str(Sig(func)) == '(a, *args, bar)'
    # See that args is now a PK kind with a default of (). Also, bar became KO.
    assert str(Sig(wfunc)) == '(a, args=(), *, bar)'

    # -----------------------------------------------------------------------------------
    # Let's see what happens when we give bar a default value

    def func2(a, *args, bar=10):
        return trace_call(func2, locals())

    wfunc = tuple_the_args(func2)
    assert wfunc(1, [2, 3]) == 'func2(a=1, args=(2, 3), bar=10)'
    assert wfunc(1, [2, 3], bar=4) == 'func2(a=1, args=(2, 3), bar=4)'

    # On the other hand, specifying bar as a positional won't work.
    # The reason is: args was a variadic, so everything after it should be KO or VK
    # The tuple_the_args doesn't change those signatures.
    #
    with pytest.raises(FuncCallNotMatchingSignature) as e_info:
        wfunc(1, [2, 3], 4)
        assert e_info.value == (
            'There should be only keyword arguments after the Variadic args. '
            'Function was called with (positional=(1, [2, 3], 4), keywords={})'
        )

    # pytest.raises()


@pytest.mark.xfail
def test_normalize_func_simply(function_normalizer=normalized_func):
    # -----------------------------------------------------------------------------------
    def p0113(po1, /, pk1, pk2, *, ko1):
        return f'{po1=}, {pk1=}, {pk2=}, {ko1=}'

    func = p0113
    po1, pk1, pk2, ko1 = 1, 2, 3, 4

    norm_func = function_normalizer(func)

    func_output = func(po1, pk1, pk2=pk2, ko1=ko1)

    norm_func_output = norm_func(po1, pk1, pk2, ko1)

    assert norm_func_output == func_output
    norm_func_output = norm_func(po1, pk1, pk2, ko1=ko1)
    assert norm_func_output == func_output
    norm_func_output = norm_func(po1, pk1, pk2=pk2, ko1=ko1)
    assert norm_func_output == func_output
    norm_func_output = norm_func(po1, pk1=pk1, pk2=pk2, ko1=ko1)
    assert norm_func_output == func_output
    norm_func_output = norm_func(po1=po1, pk1=pk1, pk2=pk2, ko1=ko1)
    assert norm_func_output == func_output

    # -----------------------------------------------------------------------------------
    def p1234(pka, *vpa, koa, **vka):
        return f'{pka=}, {vpa=}, {koa=}, {vka=}'

    pka, vpa, koa, vka = 1, (2, 3), 4, {'a': 'b', 'c': 'd'}

    func = p1234
    norm_func = function_normalizer(func)

    func_output = func(pka, *vpa, koa, **vka)
    norm_func_output = norm_func(pka, vpa, koa, vka)

    assert norm_func_output == func_output


# -----------------------------------------------------------------------------------


def p1234(pka, *vpa, koa, **vka):
    return f'{pka=}, {vpa=}, {koa=}, {vka=}'


@pytest.mark.xfail
def test_normalize_func_combinatorially(function_normalizer=normalized_func):
    # -----------------------------------------------------------------------------------
    def p0113(po1, /, pk1, pk2, *, ko1):
        return f'{po1=}, {pk1=}, {pk2=}, {ko1=}'

    func = p0113
    po1, pk1, pk2, ko1 = 1, 2, 3, 4

    poa = [po1]
    ppka, kpka = [pk1], {'pk2': pk2}
    vpa = []  # no VP argument
    koa = {'ko1': ko1}
    vka = {}  # no VK argument

    norm_func = function_normalizer(func)

    func_output = func(*poa, *ppka, *vpa, **kpka, **koa, **vka)
    norm_func_output = norm_func(po1, pk1, pk2, ko1)
    assert norm_func_output == func_output
    norm_func_output = norm_func(po1, pk1, pk2, ko1=ko1)
    assert norm_func_output == func_output
    norm_func_output = norm_func(po1, pk1, pk2=pk2, ko1=ko1)
    assert norm_func_output == func_output
    norm_func_output = norm_func(po1, pk1=pk1, pk2=pk2, ko1=ko1)
    assert norm_func_output == func_output
    norm_func_output = norm_func(po1=po1, pk1=pk1, pk2=pk2, ko1=ko1)
    assert norm_func_output == func_output

    # -----------------------------------------------------------------------------------
    def p1234(pka, *vpa, koa, **vka):
        return f'{pka=}, {vpa=}, {koa=}, {vka=}'

    pka, vpa, koa, vka = 1, (2, 3), 4, {'a': 'b', 'c': 'd'}

    func = p1234
    norm_func = function_normalizer(func)

    func_output = func(pka, *vpa, koa, **vka)
    norm_func_output = norm_func(pka, vpa, koa, vka)

    assert norm_func_output == func_output

    # -----------------------------------------------------------------------------------


# TODO: It seems in some cases, the better choice would be to oblige the user to deal
#  with return annotation explicitly


def mk_sig(
    obj: Union[Signature, Callable, Mapping, None] = None,
    return_annotations=empty,
    **annotations,
):
    """Convenience function to make a signature or inject annotations to an existing one.

    >>> s = mk_sig(lambda a, b, c=1, d='bar': ..., b=int, d=str)
    >>> s
    <Signature (a, b: int, c=1, d: str = 'bar')>
    >>> # showing that sig can take a signature input, and overwrite an existing annotation:
    >>> mk_sig(s, a=list, b=float)  # note the b=float
    <Signature (a: list, b: float, c=1, d: str = 'bar')>
    >>> mk_sig()
    <Signature ()>

    Trying to annotate an argument that doesn't exist will lead to an AssertionError:

    >>> mk_sig(lambda a, b=2, c=3: ..., d=int)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: These argument names weren't found in the signature: {'d'}
    """
    if obj is None:
        return Signature()
    if callable(obj):
        obj = Signature.from_callable(obj)  # get a signature object from a callable
    if isinstance(obj, Signature):
        obj = obj.parameters  # get the parameters attribute from a signature
    params = dict(obj)  # get a writable copy of parameters
    if not annotations:
        return Signature(params.values(), return_annotation=return_annotations)
    else:
        assert set(annotations) <= set(
            params
        ), f"These argument names weren't found in the signature: {set(annotations) - set(params)}"
        for name, annotation in annotations.items():
            p = params[name]
            params[name] = Parameter(
                name=name, kind=p.kind, default=p.default, annotation=annotation,
            )
        return Signature(params.values(), return_annotation=return_annotations)


def mk_signature(parameters, *, return_annotation=empty, __validate_parameters__=True):
    """Make an inspect.Signature object with less boilerplate verbosity.
    Args:
        signature: A list of parameter specifications. This could be an inspect.Parameter object or anything that
            the mk_param function can resolve into an inspect.Parameter object.
        return_annotation: Passed on to inspect.Signature.
        __validate_parameters__: Passed on to inspect.Signature.

    Returns:
        An inspect.Signature object

    # >>> mk_signature(['a', 'b', 'c'])
    # <Signature (a, b, c)>
    # >>> mk_signature(['a', ('b', None), ('c', 42, int)])  # specifying defaults and annotations
    # <Signature (a, b=None, c: int = 42)>
    # >>> import inspect
    # >>> mk_signature(['a', ('b', inspect._empty, int)])  # specifying an annotation without a default
    # <Signature (a, b: int)>
    # >>> mk_signature(['a', 'b', 'c'], return_annotation=str)  # specifying return annotation
    # <Signature (a, b, c) -> str>
    # >>>
    # >>> # But you can always specify parameters the "long" way
    # >>> mk_signature([inspect.Parameter(name='kws', kind=inspect.Parameter.VAR_KEYWORD)], return_annotation=str)
    # <Signature (**kws) -> str>
    # >>>
    # >>> # Note that mk_signature is an inverse of signature_to_dict:
    # >>> def foo(a, b: int=0, c=None) -> int: ...
    # >>> sig_foo = signature(foo)
    # >>> assert mk_signature(**signature_to_dict(sig_foo)) == sig_foo

    """
    return Sig(parameters, return_annotation=return_annotation)


# PATTERN: tree crud pattern
def signature_to_dict(sig: Signature):
    # warn("Use Sig instead", DeprecationWarning)
    # return Sig(sig).to_simple_signature()
    return {
        'parameters': sig.parameters,
        'return_annotation': sig.return_annotation,
    }


def _merge_sig_dicts(sig1_dict, sig2_dict):
    """Merge two signature dicts. A in dict.update(sig1_dict, **sig2_dict),
    but specialized for signature dicts.
    If sig1_dict and sig2_dict both define a parameter or return annotation,
    sig2_dict decides on what the output is.
    """
    return {
        'parameters': dict(sig1_dict['parameters'], **sig2_dict['parameters']),
        'return_annotation': sig2_dict['return_annotation']
        or sig1_dict['return_annotation'],
    }


def _merge_signatures(sig1, sig2):
    """Get the merged signatures of two signatures (sig2 is the final decider of conflics)

    >>> def foo(a='a', b: int=0, c=None) -> int: ...
    >>> def bar(b: float=0.0, d: str='hi') -> float: ...
    >>> foo_sig = signature(foo)
    >>> bar_sig = signature(bar)
    >>> foo_sig
    <Signature (a='a', b: int = 0, c=None) -> int>
    >>> bar_sig
    <Signature (b: float = 0.0, d: str = 'hi') -> float>
    >>> _merge_signatures(foo_sig, bar_sig)
    <Signature (a='a', b: float = 0.0, c=None, d: str = 'hi') -> float>
    >>> _merge_signatures(bar_sig, foo_sig)
    <Signature (b: int = 0, d: str = 'hi', a='a', c=None) -> int>
    """
    # sig1_dict = Sig(sig1).to_simple_signature()
    # sig1_dict = signature_to_dict(sig1)
    # # remove variadic kinds from sig1
    # sig1_dict['parameters'] = {k: v for k, v in sig1_dict['parameters'].items() if v.kind not in var_param_kinds}
    # return Sig(**_merge_sig_dicts(sig1_dict, Sig(sig2).to_simple_dict()))
    sig1_dict = signature_to_dict(sig1)
    # remove variadic kinds from sig1
    sig1_dict['parameters'] = {
        k: v
        for k, v in sig1_dict['parameters'].items()
        if v.kind not in var_param_kinds
    }
    kws = _merge_sig_dicts(sig1_dict, signature_to_dict(sig2))
    kws['obj'] = kws.pop('parameters')
    return Sig(**kws).to_simple_signature()


def _merge_signatures_of_funcs(func1, func2):
    """Get the merged signatures of two functions (func2 is the final decider of conflics)

    >>> def foo(a='a', b: int=0, c=None) -> int: ...
    >>> def bar(b: float=0.0, d: str='hi') -> float: ...
    >>> _merge_signatures_of_funcs(foo, bar)
    <Signature (a='a', b: float = 0.0, c=None, d: str = 'hi') -> float>
    >>> _merge_signatures_of_funcs(bar, foo)
    <Signature (b: int = 0, d: str = 'hi', a='a', c=None) -> int>
    """
    return _merge_signatures(signature(func1), signature(func2))


def _merged_signatures_of_func_list(funcs, return_annotation: Any = empty):
    """

    >>> def foo(a='a', b: int=0, c=None) -> int: ...
    >>> def bar(b: float=0.0, d: str='hi') -> float: ...
    >>> def hello(x: str='hi', y=1) -> str: ...
    >>>
    >>> # Test how the order of the functions affect the order of the parameters
    >>> _merged_signatures_of_func_list([foo, bar, hello])
    <Signature (a='a', b: float = 0.0, c=None, d: str = 'hi', x: str = 'hi', y=1)>
    >>> _merged_signatures_of_func_list([hello, foo, bar])
    <Signature (x: str = 'hi', y=1, a='a', b: float = 0.0, c=None, d: str = 'hi')>
    >>> _merged_signatures_of_func_list([foo, bar, hello])
    <Signature (a='a', b: float = 0.0, c=None, d: str = 'hi', x: str = 'hi', y=1)>
    >>>
    >>> # Test the return_annotation argument
    >>> _merged_signatures_of_func_list([foo, bar], list)  # specifying that the return type is a list
    <Signature (a='a', b: float = 0.0, c=None, d: str = 'hi') -> list>
    >>> _merged_signatures_of_func_list([foo, bar], foo)  # specifying that the return type is a list
    <Signature (a='a', b: float = 0.0, c=None, d: str = 'hi') -> int>
    >>> _merged_signatures_of_func_list([foo, bar], bar)  # specifying that the return type is a list
    <Signature (a='a', b: float = 0.0, c=None, d: str = 'hi') -> float>
    """

    s = reduce(_merge_signatures, map(signature, funcs))
    # s = Sig.from_objs(*funcs).to_simple_signature()

    if (
        return_annotation in funcs
    ):  # then you want the return annotation of a specific func of funcs
        return_annotation = signature(return_annotation).return_annotation

    return s.replace(return_annotation=return_annotation)


# TODO: will we need more options for the priority argument? Like position?
def update_signature_with_signatures_from_funcs(*funcs, priority: str = 'last'):
    """Make a decorator that will merge the signatures of given funcs to the signature of the wrapped func.
    By default, the funcs signatures will be placed last, but can be given priority by asking priority = 'first'

    >>> def foo(a='a', b: int=0, c=None) -> int: ...
    >>> def bar(b: float=0.0, d: str='hi') -> float: ...
    >>> def something(y=(1, 2)): ...
    >>> def another(y=10): ...
    >>> @update_signature_with_signatures_from_funcs(foo, bar)
    ... def hello(x: str='hi', y=1) -> str:
    ...     pass
    >>> signature(hello)
    <Signature (x: str = 'hi', y=1, a='a', b: float = 0.0, c=None, d: str = 'hi')>
    >>>
    >>> # Try a different order and priority == 'first'. Notice the b arg type and default!
    >>> add_foobar_to_signature_first = update_signature_with_signatures_from_funcs(
    ...     bar, foo, priority='first'
    ... )
    >>> bar_foo_something = add_foobar_to_signature_first(something)
    >>> signature(bar_foo_something)
    <Signature (b: int = 0, d: str = 'hi', a='a', c=None, y=(1, 2))>
    >>> # See how you can reuse the decorator several times
    >>> bar_foo_another = add_foobar_to_signature_first(another)
    >>> signature(bar_foo_another)
    <Signature (b: int = 0, d: str = 'hi', a='a', c=None, y=10)>
    """
    if not isinstance(priority, str):
        raise TypeError('priority should be a string')

    if priority == 'last':

        def transform_signature(func):
            # func.__signature__ = Sig.from_objs(func, *funcs).to_simple_signature()
            func.__signature__ = _merged_signatures_of_func_list([func] + list(funcs))
            return func

    elif priority == 'first':

        def transform_signature(func):
            # func.__signature__ = Sig.from_objs(*funcs, func).to_simple_signature()
            func.__signature__ = _merged_signatures_of_func_list(list(funcs) + [func])
            return func

    else:
        raise ValueError("priority should be 'last' or 'first'")

    return transform_signature


@pytest.mark.parametrize(
    'sig_spec',
    [
        ('(po, /)'),
        ('(po=0, /)'),
        ('(pk)'),
        ('(pk=0)'),
        ('(*, ko)'),
        ('(*, ko=0)'),
        ('(po, /, pk, *, ko)'),
        ('(po=0, /, pk=0, *, ko=0)'),
        ('(*args)'),
        ('(**kwargs)'),
        ('(*args, **kwargs)'),
        ('(po, /, pk, *args, ko)'),
        ('(po=0, /, pk=0, *args, ko=0)'),
        ('(po, /, pk, *, ko, **kwargs)'),
        ('(po=0, /, pk=0, *, ko=0, **kwargs)'),
        ('(po, /, pk, *args, ko, **kwargs)'),
        ('(po=0, /, pk=0, *args, ko=0, **kwargs)'),
        ('(po1, po2, /)'),
        ('(pk1, pk2)'),
        ('(*, ko1, ko2)'),
        ('(po1, po2, /, pk1, pk2, *, ko1, ko2)'),
        ('(po1, po2, /, pk1, pk2, *args, ko1, ko2, **kwargs)'),
        ('(po1=0, po2=0, /, pk1=0, pk2=0, *args, ko1=0, ko2=0, **kwargs)'),
    ],
)
def test_call_forgivingly(sig_spec):
    sig = Sig(sig_spec)

    @sig
    def foo(*args, **kwargs):
        return args, kwargs

    def validate_call_forgivingly(*args, **kwargs):
        expected_output_kwargs = (
            kwargs
            if VK in sig.kinds.values()
            else {k: v for k, v in kwargs.items() if k in sig}
        )
        pk_in_kwargs_count = sum(
            kind == PK
            for param, kind in sig.kinds.items()
            if param in expected_output_kwargs
        )
        expected_output_args_count = (
            len(args)
            if VP in sig.kinds.values()
            else sum(kind <= PK for kind in sig.kinds.values()) - pk_in_kwargs_count
        )
        expected_output_args = args[:expected_output_args_count]
        expected_output = (expected_output_args, expected_output_kwargs)
        output = call_forgivingly(foo, *args, **kwargs)
        # print()
        # print(args, kwargs)
        # print(expected_output)
        # print(output)
        assert output == expected_output

    for args, kwargs in sig_to_inputs(sig, ignore_variadics=True):
        kwargs = dict(kwargs, **{'some': 'extra', 'added': 'kwargs'})
        po_pk_count = sum(kind <= PK for kind in sig.kinds.values())
        if len(args) == po_pk_count:
            args = args + ('some', 'extra', 'args')

        validate_call_forgivingly(*args, **kwargs)

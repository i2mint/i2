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
from i2.signatures import _empty
from i2.signatures import *
from i2.signatures import normalized_func

from typing import Iterable, Union, Callable, List, Any
from inspect import Parameter, Signature
from functools import singledispatch


# ------------------------------------------------------------------------------------
# Test utils

ParameterAble = Union[int, Parameter, str]
ParamsAble_ = Union[ParamsAble, str, List[int]]


def _to_params(params: ParamsAble_):
    """

    >>> str(Sig(_to_params([0, 0, 1, 1, 1, 2, 3, 4])))
    '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'

    >>> str(Sig(_to_params("00111234")))
    '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'
    """
    if isinstance(params, (Callable, Signature)):
        yield from Sig(params).params
    else:
        for i, spec in enumerate(params):
            if isinstance(spec, int):
                kind = spec
                yield Parameter(f'a{kind}{i}', kind=kind)
            elif isinstance(spec, Parameter):
                param = spec
                yield param
            elif isinstance(spec, str) and spec.isnumeric():
                kind = int(spec)
                yield Parameter(f'a{kind}{i}', kind=kind)
            else:
                try:
                    yield ensure_param(spec)
                except Exception:
                    raise TypeError(
                        f"Don't know how to handle this type of obj: {spec}"
                    )


def _params_to_arg_name_and_val(params: ParamsAble_):
    """

    >>> assert dict(_params_to_arg_name_and_val(_to_params("00111234"))) == {
    ...     "a00": 0,
    ...     "a01": 1,
    ...     "a12": 2,
    ...     "a13": 3,
    ...     "a14": 4,
    ...     "a25": (5, -5),
    ...     "a36": 6,
    ...     "a47": {"a47": 7, "a47_": -7},
    ... }
    """
    params = _to_params(params)
    for i, param in enumerate(params):
        if param.kind == Parameter.VAR_POSITIONAL:
            val = (i, -i)
        elif param.kind == Parameter.VAR_KEYWORD:
            val = {param.name: i, param.name + '_': -i}
        else:
            val = i
        yield (param.name, val)


assert dict(_params_to_arg_name_and_val(_to_params('00111234'))) == {
    'a00': 0,
    'a01': 1,
    'a12': 2,
    'a13': 3,
    'a14': 4,
    'a25': (5, -5),
    'a36': 6,
    'a47': {'a47': 7, 'a47_': -7},
}


def inject_defaults(params: ParamsAble_, defaults: dict):
    """Yields params with defaults ({argname: default_val,...}) edited.

    >>> assert (
    ...     str(
    ...         Sig(
    ...             inject_defaults(
    ...                 _to_params("00111234"), defaults={"a14": 40, "a36": 60}
    ...             )
    ...         )
    ...     )
    ...     == "(a00, a01, /, a12, a13, a14=40, *a25, a36=60, **a47)"
    ... )
    """
    for param in _to_params(params):
        if param.name in defaults:
            yield param.replace(default=defaults[param.name])
        else:
            yield param


assert (
    str(Sig(inject_defaults(_to_params('00111234'), defaults={'a14': 40, 'a36': 60})))
    == '(a00, a01, /, a12, a13, a14=40, *a25, a36=60, **a47)'
)


def _str_of_call_args(_call_kwargs: dict):
    return ', '.join(f'{k}={v}' for k, v in _call_kwargs.items())


def mk_func_from_params(
    params: ParamsAble = '00111234',
    *,
    defaults=None,
    name=None,
    callback: Callable[[dict], Any] = _str_of_call_args,
):
    """Make a function (that actually returns something based on args) from params.

    :param params: params (arguments) of the function (can be expressed in many ways!)
    :param defaults: Optional {argname: default,...} dict to inject defaults
    :param name: Optional name to give the function
    :param callback: The function defining what the function actually does.
        Must be a function taking a single dict input encapsulating the all arguments.
        The default will return a string representation of this dict.
    :return: A function with the specified params, returning a string of it's (call) args

    There's many ways you can express the `params` input.
    Any of the ways understood by the `signatures.ensure_params` function, for one;
    plus a few more.

    One nice way to express the params is through an actual function.
    Note that the code of the function isn't even looked out.
    Only it's signature is taken into consideration.
    The returned function will have the same signature.
    Instead, the callback function will be acalled on the infered _call_kwargs
    dict of {argname: argval} pairs.
    The default callaback is a string exhibiting these argname/argval pairs.

    >>> f = mk_func_from_params(lambda x, /, y, *, z: None)
    >>> print(f"{f.__name__}{Sig(f)}")
    f(x, /, y, *, z)
    >>> f(1, 2, z=3)
    'x=1, y=2, z=3'
    >>> f(1, y=2, z=3)
    'x=1, y=2, z=3'
    >>> f = mk_func_from_params(lambda x, /, y=42, *, z='ZZZ': None)
    >>> print(f"{f.__name__}{Sig(f)}")
    f(x, /, y=42, *, z='ZZZ')
    >>> f(3.14)
    'x=3.14, y=42, z=ZZZ'

    If you're not interested in having that level of control, but are just
    interested in the number and kinds of the arguments, you can specify only that;
    a sequence of kinds.
    These must be a non-decreasing sequence of integers between
    0 and 4 inclusive. These integers represent kinds of parameters.
    See https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
    to see what each integer value means.
    You can also specify this integer sequence as a single string, as shown below.

    >>> f = mk_func_from_params(params="00111234")
    >>> str(Sig(f))
    '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'
    >>> f(0, 1, 2, 3, 4, 5, -5, a36=6, a47={"a47": 7, "a47_": -7})
    "a00=0, a01=1, a12=2, a13=3, a14=4, a25=(5, -5), a36=6, a47={'a47': {'a47': 7, 'a47_': -7}}"
    >>> f(0, 1, 2, a13=3, a14=4, a36=6)
    'a00=0, a01=1, a12=2, a13=3, a14=4, a25=(), a36=6, a47={}'

    What just happened?
    Well, `params="00111234"` was transformed to `params=[0, 0, 1, 1, 1, 2, 3, 4]`,
    which was transformed to a list of the same size, using

    Now, if you really want full control over those params, you can specify them
    completely using the `inspect.Parameter` class.
    You can also decide what level of control you want, and mix and match all kinds of
    specifications, as below.

    >>> from inspect import Parameter
    >>> f = mk_func_from_params([
    ...     0,
    ...     'blah',
    ...     Parameter(name='hello',
    ...               kind=Parameter.POSITIONAL_OR_KEYWORD,
    ...               default='world')
    ... ])
    >>> print(f"{f.__name__}{Sig(f)}")
    f(a00, /, blah, hello='world')
    >>> assert f(11, 22) == 'a00=11, blah=22, hello=world'

    """
    params = _to_params(params)
    params = inject_defaults(params, defaults=defaults or {})
    sig = Sig(params)

    @sig
    def arg_str_func(*args, **kwargs):
        _call_kwargs = sig.kwargs_from_args_and_kwargs(
            args, kwargs, apply_defaults=True
        )
        return callback(_call_kwargs)

    arg_str_func.__name__ = name or 'f' + ''.join(str(p.kind) for p in params)

    return arg_str_func


def mk_func_inputs_for_params(params: ParamsAble_, param_to_input):
    pass


#
# @mk_func_from_params.register
# def mk_func_from_params(params: Iterable[int], defaults=None, name=None):
#     """
#
#     :param kinds:
#     :param defaults:
#     :return:
#
#     Make a sequence of kinds (must be a non-decreasing sequence of integers between
#     0 and 4 inclusive. These integers represent kinds of parameters.
#     See https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
#     to see what each integer value means.
#
#     >>> kinds = list(map(int, "00111234"))
#     >>> kinds
#     [0, 0, 1, 1, 1, 2, 3, 4]
#
#     Note: `kinds_to_arg_str_func` also works directly with strings such as "00111234".
#
#     Now tell `kinds_to_arg_str_func` to make a function with those kinds.
#
#     >>> f = kinds_to_arg_str_func(kinds)
#     >>> str(Sig(f))
#     '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'
#     >>> f(0, 1, 2, 3, 4, 5, -5, a36=6, a47={"a47": 7, "a47_": -7})
#     "a00=0, a01=1, a12=2, a13=3, a14=4, a25=(5, -5), a36=6, a47={'a47': {'a47': 7, 'a47_': -7}}"
#     >>> f(0, 1, 2, a13=3, a14=4, a36=6)
#     'a00=0, a01=1, a12=2, a13=3, a14=4, a25=(), a36=6, a47={}'
#     """
#     kinds = params
#     params = inject_defaults(_to_params(kinds), defaults=defaults or {})
#     name = name or "f" + "".join(map(str, kinds))
#
#     return mk_func_from_params(params, defaults, name)

#
#
# @mk_func_from_params.register
# def _(kinds: str):
#     """
#     >>> f = kinds_to_arg_str_func("00111234")
#     >>> str(Sig(f))
#     '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'
#     >>> f(0, 1, 2, 3, 4, 5, -5, a36=6, a47={"a47": 7, "a47_": -7})
#     "a00=0, a01=1, a12=2, a13=3, a14=4, a25=(5, -5), a36=6, a47={'a47': {'a47': 7, 'a47_': -7}}"
#     >>> f(0, 1, 2, a13=3, a14=4, a36=6)
#     'a00=0, a01=1, a12=2, a13=3, a14=4, a25=(), a36=6, a47={}'
#
#     """
#     return mk_func_from_params(_kinds_str_to_int_list(kinds))
#
#
# f = mk_func_from_params("00111234")
# assert str(Sig(f)) == "(a00, a01, /, a12, a13, a14, *a25, a36, **a47)"
# assert (
#     f(0, 1, 2, 3, 4, 5, -5, a36=6, a47={"a47": 7, "a47_": -7})
#     == "a00=0, a01=1, a12=2, a13=3, a14=4, a25=(5, -5), a36=6, a47={'a47': {'a47': 7, 'a47_': -7}}"
# )


empty = _empty

mappingproxy = type(Signature().parameters)


def trace_call(func, local_vars, name=None):
    if name is None:
        name = func.__name__
    return (
        f'{name}('
        + ', '.join(f'{argname}={local_vars[argname]}' for argname in Sig(func).names)
        + ')'
    )


# class KeywordArg(dict):
#     """Just to mark a dict as a keyword argument"""
#
#
# def _separate_pk_arguments_into_positional_and_keyword(pka):
#     args = []
#     kwargs = {}
#     pka_iter = iter(pka)
#     for a in pka_iter:
#         if not isinstance(a, KeywordArg):
#             args.append(a)
#         else:
#             kwargs.update(dict(a))
#     for a in pka_iter:
#         kwargs.update(dict(a))
#
#     return args, kwargs

# ------------------------------------------------------------------------------------
# And finally... the tests


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

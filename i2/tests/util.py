"""Utils for testing"""

from inspect import Parameter, Signature
from typing import List, Any, Union, Callable

from i2.signatures import _empty
from i2.signatures import ParamsAble, Sig, ensure_param

ParameterAble = Union[int, Parameter, str]
ParamsAble_ = Union[ParamsAble, str, List[int]]


def generate_params(params: ParamsAble_):
    """Generate inspect.Parameter instances quickly.

    Example: Generate params solely from a list of their kinds

    >>> str(Sig(generate_params([0, 0, 1, 1, 1, 2, 3, 4])))
    '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'

    >>> str(Sig(generate_params("00111234")))
    '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'
    """
    if isinstance(params, (Callable, Signature)):
        # generate params from a callable or signature
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


def params_to_arg_name_and_val(params: ParamsAble_):
    """Generate a `{argname: argval, ...}` dictionary from an iterable of params.

    >>> assert dict(params_to_arg_name_and_val(generate_params("00111234"))) == {
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
    params = generate_params(params)
    for i, param in enumerate(params):
        if param.kind == Parameter.VAR_POSITIONAL:
            val = (i, -i)
        elif param.kind == Parameter.VAR_KEYWORD:
            val = {param.name: i, param.name + '_': -i}
        else:
            val = i
        yield (param.name, val)


def inject_defaults(params: ParamsAble_, defaults: dict):
    """Yields params with defaults ({argname: default_val,...}) edited.

    >>> assert (
    ...     str(
    ...         Sig(
    ...             inject_defaults(
    ...                 generate_params("00111234"), defaults={"a14": 40, "a36": 60}
    ...             )
    ...         )
    ...     )
    ...     == "(a00, a01, /, a12, a13, a14=40, *a25, a36=60, **a47)"
    ... )
    """
    for param in generate_params(params):
        if param.name in defaults:
            yield param.replace(default=defaults[param.name])
        else:
            yield param


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
    params = generate_params(params)
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
#     params = inject_defaults(generate_params(kinds), defaults=defaults or {})
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

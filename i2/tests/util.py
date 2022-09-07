"""Utils for testing"""

from inspect import Parameter, Signature
from itertools import product
from typing import (
    List,
    Any,
    Union,
    Callable,
    Iterator,
    Tuple,
    Optional,
    Iterable,
    Mapping,
)
from inspect import signature

from i2.signatures import KO, PK, PO, VP, VK, var_param_kinds
from i2.signatures import _empty
from i2.signatures import ParamsAble, Sig, ensure_param, SignatureAble

ParameterAble = Union[int, Parameter, str]
ParamsAble_ = Union[ParamsAble, str, List[int]]


def _is_valid_arg_for_sig(x):
    return (
        isinstance(x, (Callable, Signature))
        or isinstance(x, str)
        and x.startswith('(')
        and x.endswith(')')
    )


def generate_params(params: ParamsAble_):
    """Generate inspect.Parameter instances quickly.

    Example: Generate params solely from a list of their kinds

    >>> str(Sig(generate_params([0, 0, 1, 1, 1, 2, 3, 4])))
    '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'

    >>> str(Sig(generate_params("00111234")))
    '(a00, a01, /, a12, a13, a14, *a25, a36, **a47)'
    """
    if _is_valid_arg_for_sig(params):
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


def _params_to_name(params):
    return 'f' + ''.join(str(int(p.kind)) for p in params)


def mk_func_from_params(
    params: ParamsAble = '00111234',
    *,
    defaults=None,
    name=None,
    callback: Callable[[dict], Any] = _str_of_call_args,
):
    """Make a function (that actually returns something based on args) from params.

    See Also: ``sig_to_func``

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
    "a00=0, a01=1, a12=2, a13=3, a14=4, a25=(5, -5), a36=6, a47={'a47': 7, 'a47_': -7}"
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

    arg_str_func.__name__ = name or _params_to_name(params)

    return arg_str_func


def _sig_to_str_of_call_args_code_str(sig: Sig):
    return (
        'return ' + 'f"' + _str_of_call_args({p: f'{{{p}}}' for p in sig.names}) + '"'
    )


def _is_simple_expression(code_lines):
    if len(code_lines) == 1:
        line = code_lines[0].strip()
        if not (
            line in {'pass', '...'}
            or line.startswith('return')
            or (line.startswith('"') and line.endswith("'"))
        ):
            return True
    else:
        return False


def sig_to_func(
    sig: ParamsAble = '00111234',
    code_lines: Union[
        str, Iterable, Callable[[Sig], str]
    ] = _sig_to_str_of_call_args_code_str,
    *,
    name: Optional[str] = None,
    globals: Optional[dict] = None,
    locals: Optional[Mapping] = None,
):
    """
    Make a function from a signature

    See Also: ``mk_func_from_params``

    More information: https://github.com/i2mint/i2/issues/34

    :param sig: Signature (or something that can be made into a signature) the func
    should have
    :param name: Name the function should have
    :param code_lines: The code lines

    The ``globals`` must be a dictionary and ``locals`` can be any mapping, defaulting
    to the current globals and locals. If only ``globals`` is given, ``locals``
    defaults to it.

    >>> sig = Sig('(a, /, b=2)')
    >>> f = sig_to_func(sig)
    >>> str(Sig(f))
    '(a, /, b=2)'
    >>> f(2, 3)
    'a=2, b=3'
    >>> f.__name__
    'f01'

    See how the function was given a name automatically? This name was created by
    appending the kind number
    (see https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind)
    to 'f'.

    Let's demo how we can

    - use kind numbers (see `inspect` module) to specify the signature

    - give an explicit name to the function

    - specify ``locals()`` so that ``sig_to_func`` can insert the new function there
    (which makes it picklable, for instances)

    >>> _ = sig_to_func('012', name='foo', locals=locals())
    >>> # and now `foo` is in local name space, and has signature:
    >>> str(Sig(foo))
    '(a00, /, a11, *a22)'
    >>> foo(1,2,3,4,5)
    'a00=1, a11=2, a22=(3, 4, 5)'

    In the examples above, the function body was was created from the input ``sig``
    through the ``_sig_to_str_of_call_args_code_str`` function which outputs a string
    formed from the input argument names and values.

    But you can specify your own function. This function should take a ``Sig`` object
    and return the string or lines (iterable of strings) of the function's body.
    You can also specify a string or lines directly.

    >>> g = sig_to_func('(x, y=2)', 'pass')
    >>> assert g(2, y=3) is None
    >>> h = sig_to_func('(x, y=2)', 'return x * y')
    >>> h(3)
    6
    >>> h(x=10, y=3)
    30

    You can even omit the return instruction if, as in ``lambda`` functions, the body is
    a simple expression (not `pass`, `...`, or starting something with quotes).

    >>> h = sig_to_func('(x, y=2)', 'x / y')
    >>> h(10, 5)
    2.0

    """
    sig = Sig(generate_params(sig))
    name = name or _params_to_name(sig.params)
    if callable(code_lines):
        code_lines = code_lines(sig)  # call the function on sig to get lines
    if isinstance(code_lines, str):
        code_lines = code_lines.split('\n')
    if _is_simple_expression(code_lines):
        # If code_lines has only one line and it seems it's an expression, prepend return
        code_lines = [f'return {code_lines[0]}']
    code_string = '\n\t'.join(code_lines)
    func_def_string = f'def {name}{sig}:\n\t{code_string}'
    _locals = locals or {}
    exec(func_def_string, globals, _locals)
    return _locals[name]


def _all_prefixes(x: Iterable):
    """
    >>> list(_all_prefixes([1,2,3]))
    [(), (1,), (1, 2), (1, 2, 3)]
    >>> list(map(dict, _all_prefixes({'a': 1, 'b': 2}.items())))
    [{}, {'a': 1}, {'a': 1, 'b': 2}]
    """
    x = tuple(x)
    for i in range(len(x) + 1):
        yield x[:i]


def _args_kwargs_combinations(args, kwargs):
    """

    >>> assert list(_args_kwargs_combinations((1,2), {'a': 3, 'b': 4})) == [
    ...     ((), {}),
    ...     ((), {'a': 3}),
    ...     ((), {'a': 3, 'b': 4}),
    ...     ((1,), {}),
    ...     ((1,), {'a': 3}),
    ...     ((1,), {'a': 3, 'b': 4}),
    ...     ((1, 2), {}),
    ...     ((1, 2), {'a': 3}),
    ...     ((1, 2), {'a': 3, 'b': 4})
    ... ]

    """
    for args_prefix, kwargs_items_prefix in product(
        _all_prefixes(args), _all_prefixes(kwargs.items())
    ):
        yield args_prefix, dict(kwargs_items_prefix)


def variadic_type(sig, variadics):
    var_kinds = [sig.kinds[param] for param in variadics]
    if not variadics:
        return 'no_var'
    if VP in var_kinds and VK not in var_kinds:
        return 'vp_only'
    if VK in var_kinds and VP not in var_kinds:
        return 'vk_only'
    else:
        return 'vp_vk'


def create_variadic_source(sig, variadics, dflt_source):
    var_type = variadic_type(sig, variadics)
    if var_type == 'no_var':
        result = ((), {})
    elif var_type == 'vp_only':
        result = (dflt_source[0], {})
    elif var_type == 'vk_only':
        result = ((), dflt_source[1])
    elif var_type == 'vp_vk':
        result = dflt_source
    return result


# TODO: Get rid of ignore_variadics once using code is refactored
def sig_to_inputs(
    sig: SignatureAble,
    argument_vals: Optional[Iterable] = None,
    *,
    variadics_source: Tuple[tuple, dict] = (
        ('args1', 'args2'),
        {'kwargs1': 'kwargs1_val'},
    ),
) -> Iterator[Tuple[tuple, dict]]:
    """Generate all kind-valid (arg, kwargs) input combinations for a function with a
    given signature ``sig``, with argument values taken from the ``argument_vals``

    :param sig: A signature or anything that ``i2.Sig`` can use to create one (e.g.
        function, string, list of dicts etc.)
    :param argument_vals: An interable from which the argument values will be drawn.
        Defaults to ``list(range(n_args))``.
    :param variadics_source: The ``(tuple, dict)`` pair that will be used to extract
        variadics' inputs.
    :return: A generator of ``(args: tuple, kwargs: dict)`` pairs

    >>> assert list(
    ...     sig_to_inputs(lambda a, b, /, c, d, *, e, f: None)
    ... ) == [
    ...     ((0, 1), {'c': 2, 'd': 3, 'e': 4, 'f': 5}),
    ...     ((0, 1, 2), {'d': 3, 'e': 4, 'f': 5}),
    ...     ((0, 1, 2, 3), {'e': 4, 'f': 5})
    ... ]

    >>> list(sig_to_inputs(Sig('(a, *args, b, **kwargs)')))  # doctest: +NORMALIZE_WHITESPACE
    [((), {'a': 0, 'b': 1}),
     ((), {'a': 0, 'b': 1, 'kwargs1': 'kwargs1_val'}),
     (('args1',), {'a': 0, 'b': 1}),
     (('args1',), {'a': 0, 'b': 1, 'kwargs1': 'kwargs1_val'}),
     (('args1', 'args2'), {'a': 0, 'b': 1}),
     (('args1', 'args2'), {'a': 0, 'b': 1, 'kwargs1': 'kwargs1_val'}),
     ((0,), {'b': 1}),
     ((0,), {'b': 1, 'kwargs1': 'kwargs1_val'}),
     ((0, 'args1'), {'b': 1}),
     ((0, 'args1'), {'b': 1, 'kwargs1': 'kwargs1_val'}),
     ((0, 'args1', 'args2'), {'b': 1}),
     ((0, 'args1', 'args2'), {'b': 1, 'kwargs1': 'kwargs1_val'})]

    Tip: To ignore variadics all together, you can specify ``variadics_source=((), {})``.
    """
    sig = Sig(sig)
    init_sig = sig
    already_yielded = []
    variadics = [param for param, kind in sig.kinds.items() if kind in var_param_kinds]
    if variadics:
        for v in variadics:
            sig -= v
        variadics_source = create_variadic_source(
            init_sig, variadics, dflt_source=variadics_source
        )
        var_args, var_kwargs = variadics_source
        for args, kwargs in sig_to_inputs(sig, argument_vals):
            for _args, _kwargs in _args_kwargs_combinations(var_args, var_kwargs):
                yield args + _args, dict(kwargs, **_kwargs)
    else:
        for sub_sig in _get_sub_sigs_from_default_values(sig):
            po, pk, ko = _get_non_variadic_kind_counts(sub_sig)
            for args, kwargs_vals in _sig_to_inputs(
                po, pk, ko, argument_vals=argument_vals
            ):
                input_ = (
                    tuple(args),
                    {k: v for k, v in zip(sub_sig.names[len(args) :], kwargs_vals)},
                )
                if input_ not in already_yielded:
                    yield input_
                    already_yielded.append(input_)


def _sig_to_inputs(po=0, pk=0, ko=0, argument_vals: Optional[Iterable] = None):
    """

    >>> list(_sig_to_inputs(2,2,2))
    [([0, 1], [2, 3, 4, 5]), ([0, 1, 2], [3, 4, 5]), ([0, 1, 2, 3], [4, 5])]

    :param po: Number of POSITION_ONLY args in signature.
    :param pk: Number of POSITION_OR_KEYWORD args in signature.
    :param ko: Number of KEYWORD_ONLY args in signature.
    :param argument_vals: An interable from which the argument values will be drawn.
        Defaults to ``list(range(n_args))``.
    :return: A generator of ``(vals_for_args: tuple, vals_for_kwargs)`` pairs
    """
    if argument_vals is None:
        argument_vals = list(range(po + pk + ko))
    else:
        argument_vals = list(argument_vals)
    for n_args_from_pk in range(pk + 1):
        yield argument_vals[: (po + n_args_from_pk)], argument_vals[
            (po + n_args_from_pk) :
        ]


def _get_sub_sigs_from_default_values(sig: Sig) -> Iterator[Sig]:
    """Generate all the signatures compatible with the given signature
    by ignoring the arguments that have default values.
    >>> sig = Sig('(a=0, /, b=0, *args, c=0, **kwargs)')
    >>> assert [str(s) for s in _get_sub_sigs_from_default_values(sig)] == [
    ...     '(*args, **kwargs)',
    ...     '(a=0, /, *args, **kwargs)',
    ...     '(*args, b=0, **kwargs)',
    ...     '(a=0, /, b=0, *args, **kwargs)',
    ...     '(*args, c=0, **kwargs)',
    ...     '(a=0, /, *args, c=0, **kwargs)',
    ...     '(*args, b=0, c=0, **kwargs)',
    ...     '(a=0, /, b=0, *args, c=0, **kwargs)'
    ... ]
    """

    def internal_get_sub_sigs(sig):
        kos = [n for n in sig.names_of_kind[KO] if n in sig.defaults]
        for ko in reversed(kos):
            _sig = sig - ko
            yield from internal_get_sub_sigs(_sig)

        pks = [n for n in sig.names_of_kind[PK] if n in sig.defaults]
        for i, pk in reversed(list(enumerate(pks))):
            _sig = sig - pk
            pks_to_transform_to_ko = pks[i + 1 :]
            params = [
                p.replace(kind=KO) if p.name in pks_to_transform_to_ko else p
                for p in _sig.params
            ]
            params.sort(key=lambda p: p.kind)
            _sig = Sig(params)
            yield from internal_get_sub_sigs(_sig)

        pos = [n for n in sig.names_of_kind[PO] if n in sig.defaults]
        if pos:
            _sig = sig - pos[-1]
            params = [p.replace(kind=KO) if p.kind == PK else p for p in _sig.params]
            params.sort(key=lambda p: p.kind)
            _sig = Sig(params)
            yield from internal_get_sub_sigs(_sig)

        if sig not in already_yielded:
            yield sig
            already_yielded.add(sig)

    already_yielded = set()
    yield from internal_get_sub_sigs(sig)


def _get_non_variadic_kind_counts(sig: Sig):
    po = pk = ko = 0
    for kind in sig.kinds.values():
        po += kind == sig.POSITIONAL_ONLY
        pk += kind == sig.POSITIONAL_OR_KEYWORD
        ko += kind == sig.KEYWORD_ONLY
    return po, pk, ko


def mk_func_inputs_for_params(params: ParamsAble_, param_to_input):
    pass


# ---------------------------------------------------------------------------------------
# Tools to analyze compatibility between signature and function call

from i2 import Pipe
import re

_signature_msg_patterns = [
    'keyword arguments$',
    'invalid keyword argument',
    'expected at most',
    'keyword argument',
    'got some positional\-only arguments passed as keyword arguments',
    'no signature found',
]

_signature_msg_regex = re.compile('|'.join(map('({})'.format, _signature_msg_patterns)))
is_signature_msg = Pipe(_signature_msg_regex.search, bool)


def _is_signature_error(
    error_obj,
    signature_error_types=(TypeError, ValueError),
    is_signature_msg=is_signature_msg,
):
    if isinstance(error_obj, signature_error_types):
        error_msg = str(error_obj)
        return is_signature_msg(error_msg)
    return False


def call_and_return_error(func, /, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return None
    except BaseException as error_obj:
        return error_obj


def on_error_return_none(func, /, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except BaseException as error_obj:
        return None


call_raises_signature_error = Pipe(call_and_return_error, _is_signature_error)

call_raises_signature_error.__doc__ = '''
>>> call_raises_signature_error(lambda x, /, y: None, 1, y=2)
False
>>> call_raises_signature_error(lambda x, /, y: None, x=1, y=2)
True
'''

# Yes, I too see that this can be made into yet another Pipe!
def function_is_compatible_with_signature(func, sig):
    """
    Runs through all combinations of positional and keyword arguments,

    >>> function_is_compatible_with_signature(hasattr, Sig(lambda obj, name: ...))
    False
    >>> function_is_compatible_with_signature(hasattr, Sig(lambda obj, name, /: ...))
    True
    """

    def _call_raises_sig_error():
        for args, kwargs in sig_to_inputs(sig):
            yield call_raises_signature_error(func, *args, **kwargs)

    return not any(_call_raises_sig_error())


def builtin_objects():
    for name in dir(__builtins__):
        yield getattr(__builtins__, name)


def builtin_signatureless_callables():
    """
    A generator of builtin callables that don't have signatures.
    """
    for obj in builtin_objects():
        if callable(obj) and call_raises_signature_error(signature, obj):
            yield obj


# ---------------------------------------------------------------------------------------

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

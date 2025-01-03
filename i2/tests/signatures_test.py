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

from i2 import Sig
from i2.signatures import *
from i2.signatures import (
    normalized_func,
    sigs_for_sigless_builtin_name,
    _robust_signature_of_callable,
)

from i2.tests.util import (
    call_and_return_error,
    sig_to_inputs,
    trace_call,
    function_is_compatible_with_signature,
)


def test_class_attribute_signatures():
    class Klass:
        def leave(self): ...

        @property
        def no(self): ...

        @cached_property
        def stone(self): ...

        unturned = partial(lambda self: self)

    assert str(Sig(Klass.leave)) == "(self)"
    assert str(Sig(Klass.no)) == "(self)"
    assert str(Sig(Klass.stone)) == "(self)"
    assert str(Sig(Klass.unturned)) == "(self)"

    instance = Klass()

    assert str(Sig(instance.leave)) == "()"
    assert str(Sig(instance.no)) == "()"
    assert str(Sig(instance.stone)) == "()"
    assert str(Sig(instance.unturned)) == "(self)"


def test_add_optional_keywords():
    @Sig.add_optional_keywords({"c": 2, "d": 3}, {"c": int})
    def foo(a, *, b=1, **kwargs):
        return f"{a=}, {b=}, {kwargs=}"

    assert str(Sig(foo)) == "(a, *, c: int = 2, d=3, b=1, **kwargs)"
    assert foo(0, d=10) == "a=0, b=1, kwargs={'d': 10}"

    # Testing when kwarg_annotations is used with keyword argument
    @Sig.add_optional_keywords({"c": 2, "d": 3}, kwarg_annotations={"c": int})
    def foo(a, *, b=1, **kwargs):
        pass

    assert str(Sig(foo)) == "(a, *, c: int = 2, d=3, b=1, **kwargs)"

    # Testing when both kwarg_and_defaults and kwarg_annotations are used
    # with keyword argument
    @Sig.add_optional_keywords(
        kwarg_and_defaults={"c": 2, "d": 3}, kwarg_annotations={"c": int}
    )
    def foo(a, *, b=1, **kwargs):
        pass

    assert str(Sig(foo)) == "(a, *, c: int = 2, d=3, b=1, **kwargs)"

    # Testing when add_optional_keywords is used as instance method
    def foo(a, *, b=1, **kwargs):
        return f"{a=}, {b=}, {kwargs=}"

    assert foo(0, d=10) == "a=0, b=1, kwargs={'d': 10}"

    sig = Sig(foo)
    new_sig = sig.add_optional_keywords({"c": 2, "d": 3}, {"c": int})

    assert str(new_sig) == ("(a, *, c: int = 2, d=3, b=1, **kwargs)")

    sig(foo)  # decorate foo, and see that it still works
    assert foo(0, d=10) == "a=0, b=1, kwargs={'d': 10}"
    f = sig(foo)  # decorate foo, returning a pointer to foo called f
    assert f(0, d=10) == "a=0, b=1, kwargs={'d': 10}"


def test_signature_equality_and_hashing():
    import pickle

    def foo(x, y, z=3):
        pass

    def bar(x, y, z=3):
        pass

    # different instances of (should be a) same signature
    ref_sig = Sig(foo)

    sigs = [
        ref_sig,
        Sig(foo),  # another instance of the same Sig(foo)
        Sig("(x, y, z=3)"),  # signature made explicitly from a string
        Sig("x") + Sig("(y, z=3)"),  # signature made from an add operation
        Sig(bar),  # different function, same signature
        pickle.loads(pickle.dumps(ref_sig)),  # un-pickled signature
    ]

    assert all(
        sig == ref_sig for sig in sigs
    ), "sigs should be equal from a == point of view"

    # Let's see that
    assert all(
        hash(sig) == hash(ref_sig) for sig in sigs
    ), "sigs should have the same hash"

    # ... and if that wasn't convincing, let's see how the sigs behave as dict keys:
    # If sigs were different, the following would add key-value pairs to the base dict.
    # But it doesn't. You always get the same one key (same signature) with different
    # values:
    t = {ref_sig: 0}
    assert t == {ref_sig: 0}
    t[sigs[2]] = 2
    assert t == {ref_sig: 2}  # same signature with new 2 value
    t[sigs[3]] = 3
    assert t == {ref_sig: 3}  # same signature with new 3 value

    # What if we just have a return annotation?
    def baz(x, y, z=3) -> None:
        pass

    assert Sig(baz) != ref_sig, "return annotation of baz should make it different"


def test_signature_of_partial():
    from functools import partial

    def foo(a, b, c=3) -> int:
        return a + b * c

    assert str(Sig(foo)) == "(a, b, c=3) -> int"
    assert str(Sig(partial(foo, 1))) == "(b, c=3) -> int"
    assert str(Sig(partial(foo, a=1, b=2))) == "(*, a=1, b=2, c=3) -> int"
    assert str(Sig(partial(foo, 1, 2))) == "(c=3) -> int"
    assert str(Sig(partial(foo, 1, b=2))) == "(*, b=2, c=3) -> int"


def test_some_edge_cases_of_sig():
    from operator import itemgetter, attrgetter, methodcaller

    assert Sig(itemgetter).names == ["key", "keys"]
    assert Sig(itemgetter(1)).names == ["iterable"]
    assert Sig(itemgetter(1, 2)).names == ["iterable"]
    assert Sig(attrgetter).names == ["key", "keys"]
    assert Sig(attrgetter("foo")).names == ["iterable"]
    assert Sig(attrgetter("foo", "bar")).names == ["iterable"]
    assert Sig(methodcaller).names == ["name", "args", "kwargs"]
    # assert Sig(methodcaller('foo')).names == []  # fix!!


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

    assert Sig(foo)._defaults_ == (3,)

    assert foo(1, 2) == 7
    #    works because Sig also changed __defaults__:
    assert foo.__defaults__ == (3,)

    @Sig(lambda x, y, *, z=3: None)
    def foo(x, y, *, z=0):
        return x + y * z

    assert foo(1, 2) == 7
    #    works because Sig also changed __defaults__ and __kwdefaults__:
    assert foo.__defaults__ == ()
    assert foo.__kwdefaults__ == {"z": 3}
    assert Sig(foo)._defaults_ == ()
    assert Sig(foo)._kwdefaults_ == {"z": 3}

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
    assert Sig(foo)._defaults_ == ()
    assert Sig(foo)._kwdefaults_ == {"z": 3}


def test_tuple_the_args():
    from i2.signatures import tuple_the_args

    def func(a, *args, bar):
        return trace_call(func, locals())

    assert func(1, 2, 3, bar=4) == "func(a=1, args=(2, 3), bar=4)"

    wfunc = tuple_the_args(func)

    # here, not that (1) args is specified as one iterable ([2, 3] instead of 2,
    # 3) and (2) the function name is the same as the wrapped (func)
    assert wfunc(1, [2, 3], bar=4) == "func(a=1, args=(2, 3), bar=4)"

    # See the func itself hasn't changed
    assert func(1, 2, 3, bar=4) == "func(a=1, args=(2, 3), bar=4)"

    assert str(Sig(func)) == "(a, *args, bar)"
    # See that args is now a PK kind with a default of (). Also, bar became KO.
    assert str(Sig(wfunc)) == "(a, args=(), *, bar)"

    # -----------------------------------------------------------------------------------
    # Let's see what happens when we give bar a default value

    def func2(a, *args, bar=10):
        return trace_call(func2, locals())

    wfunc = tuple_the_args(func2)
    assert wfunc(1, [2, 3]) == "func2(a=1, args=(2, 3), bar=10)"
    assert wfunc(1, [2, 3], bar=4) == "func2(a=1, args=(2, 3), bar=4)"

    # On the other hand, specifying bar as a positional won't work.
    # The reason is: args was a variadic, so everything after it should be KO or VK
    # The tuple_the_args doesn't change those signatures.
    #
    with pytest.raises(FuncCallNotMatchingSignature) as e_info:
        wfunc(1, [2, 3], 4)
        assert e_info.value == (
            "There should be only keyword arguments after the Variadic args. "
            "Function was called with (positional=(1, [2, 3], 4), keywords={})"
        )

    # pytest.raises()


@pytest.mark.xfail
def test_normalize_func_simply(function_normalizer=normalized_func):
    # -----------------------------------------------------------------------------------
    def p0113(po1, /, pk1, pk2, *, ko1):
        return f"{po1=}, {pk1=}, {pk2=}, {ko1=}"

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
        return f"{pka=}, {vpa=}, {koa=}, {vka=}"

    pka, vpa, koa, vka = 1, (2, 3), 4, {"a": "b", "c": "d"}

    func = p1234
    norm_func = function_normalizer(func)

    func_output = func(pka, *vpa, koa, **vka)
    norm_func_output = norm_func(pka, vpa, koa, vka)

    assert norm_func_output == func_output


# -----------------------------------------------------------------------------------


def p1234(pka, *vpa, koa, **vka):
    return f"{pka=}, {vpa=}, {koa=}, {vka=}"


@pytest.mark.xfail
def test_normalize_func_combinatorially(function_normalizer=normalized_func):
    # -----------------------------------------------------------------------------------
    def p0113(po1, /, pk1, pk2, *, ko1):
        return f"{po1=}, {pk1=}, {pk2=}, {ko1=}"

    func = p0113
    po1, pk1, pk2, ko1 = 1, 2, 3, 4

    poa = [po1]
    ppka, kpka = [pk1], {"pk2": pk2}
    vpa = []  # no VP argument
    koa = {"ko1": ko1}
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
        return f"{pka=}, {vpa=}, {koa=}, {vka=}"

    pka, vpa, koa, vka = 1, (2, 3), 4, {"a": "b", "c": "d"}

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
                name=name,
                kind=p.kind,
                default=p.default,
                annotation=annotation,
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
        "parameters": sig.parameters,
        "return_annotation": sig.return_annotation,
    }


def _merge_sig_dicts(sig1_dict, sig2_dict):
    """Merge two signature dicts. A in dict.update(sig1_dict, **sig2_dict),
    but specialized for signature dicts.
    If sig1_dict and sig2_dict both define a parameter or return annotation,
    sig2_dict decides on what the output is.
    """
    return {
        "parameters": dict(sig1_dict["parameters"], **sig2_dict["parameters"]),
        "return_annotation": sig2_dict["return_annotation"]
        or sig1_dict["return_annotation"],
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
    sig1_dict["parameters"] = {
        k: v
        for k, v in sig1_dict["parameters"].items()
        if v.kind not in var_param_kinds
    }
    kws = _merge_sig_dicts(sig1_dict, signature_to_dict(sig2))
    kws["obj"] = kws.pop("parameters")
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
def update_signature_with_signatures_from_funcs(*funcs, priority: str = "last"):
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
        raise TypeError("priority should be a string")

    if priority == "last":

        def transform_signature(func):
            # func.__signature__ = Sig.from_objs(func, *funcs).to_simple_signature()
            func.__signature__ = _merged_signatures_of_func_list([func] + list(funcs))
            return func

    elif priority == "first":

        def transform_signature(func):
            # func.__signature__ = Sig.from_objs(*funcs, func).to_simple_signature()
            func.__signature__ = _merged_signatures_of_func_list(list(funcs) + [func])
            return func

    else:
        raise ValueError("priority should be 'last' or 'first'")

    return transform_signature


@pytest.mark.parametrize(
    "sig_spec",
    [
        ("(po, /)"),
        ("(po=0, /)"),
        ("(pk)"),
        ("(pk=0)"),
        ("(*, ko)"),
        ("(*, ko=0)"),
        ("(po, /, pk, *, ko)"),
        ("(po=0, /, pk=0, *, ko=0)"),
        ("(*args)"),
        ("(**kwargs)"),
        ("(*args, **kwargs)"),
        ("(po, /, pk, *args, ko)"),
        ("(po=0, /, pk=0, *args, ko=0)"),
        ("(po, /, pk, *, ko, **kwargs)"),
        ("(po=0, /, pk=0, *, ko=0, **kwargs)"),
        ("(po, /, pk, *args, ko, **kwargs)"),
        ("(po=0, /, pk=0, *args, ko=0, **kwargs)"),
        ("(po1, po2, /)"),
        ("(pk1, pk2)"),
        ("(*, ko1, ko2)"),
        ("(po1, po2, /, pk1, pk2, *, ko1, ko2)"),
        ("(po1, po2, /, pk1, pk2, *args, ko1, ko2, **kwargs)"),
        ("(po1=0, po2=0, /, pk1=0, pk2=0, *args, ko1=0, ko2=0, **kwargs)"),
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
        print()
        print(args, kwargs)
        print(expected_output)
        print(output)
        assert output == expected_output

    for args, kwargs in sig_to_inputs(sig, variadics_source=((), {})):
        kwargs = dict(kwargs, **{"some": "extra", "added": "kwargs"})
        po_pk_count = sum(kind <= PK for kind in sig.kinds.values())
        if len(args) == po_pk_count:
            args = args + ("some", "extra", "args")

        validate_call_forgivingly(*args, **kwargs)


@pytest.mark.parametrize(
    "sig_spec1, sig_spec2",
    [
        ("()", "(a)"),
        ("()", "(a=0)"),
        ("(a, /, *, c)", "(a, /, b, *, c)"),
        ("(a, /, *, c)", "(a, /, b=0, *, c)"),
        ("(a, /, b)", "(a, /, b, *, c)"),
        ("(a, /, b)", "(a, /, b, *, c=0)"),
        ("(a, /, b, *, c)", "(*args)"),
        ("(a, /, b, *, c)", "(**kwargs)"),
        ("(a, /, b, *, c)", "(*args, **kwargs)"),
        ("(a, /, b, *, c)", "(a, /, b, *args, c, **kwargs)"),
        ("(a, /, b, *, c)", "(a, b, c)"),
        ("(a, /, b, *, c)", "(a, b, /, *, c)"),
        ("(a, /, b, *, c)", "(a, /, *, b, c)"),
        ("(a, /, b, *, c)", "(a, b, /, c)"),
        ("(a, /, b, *, c)", "(a, *, b, c)"),
        (
            "(a, /, b, *, c)",
            "(x, /, b, *, c)",
        ),
        (
            "(a, /, b, *, c)",
            "(a, /, x, *, c)",
        ),
        (
            "(a, /, b, *, c)",
            "(a, /, b, *, x)",
        ),
        ("(a, /, b, *, c)", "(a=0, b=0, c=0)"),
        ("(a=0, /, b=0, *, c=0)", "(a, b, c)"),  # TODO: See if this should pass
        ("(a, b, /, c, d, *, e, f)", "(b, a, /, d, c, *, f, e)"),
        (
            "(a, b, /, c, d, *, e, f)",
            "(a, c, /, b, e, *, d, f)",
        ),
        ("()", "(*args)"),
        ("()", "(**kwargs)"),
        ("()", "(*args, **kwargs)"),
        ("(*args)", "()"),
        ("(*args)", "(**kwargs)"),
        ("(*args)", "(*args, **kwargs)"),
        ("(**kwargs)", "()"),
        ("(**kwargs)", "(*args)"),
        ("(**kwargs)", "(*args, **kwargs)"),
        ("(*args, **kwargs)", "(*args, **kwargs)"),
    ],
)
def test_call_compatibility(sig_spec1, sig_spec2):
    sig1 = Sig(sig_spec1)
    sig2 = Sig(sig_spec2)
    is_compatible = sig1.is_call_compatible_with(sig2)

    exec_env = dict()
    f_def = f"def f{sig_spec2}: pass"
    exec(f_def, exec_env)
    foo = exec_env["f"]

    # @sig2
    # def foo(*args, **kwargs):
    #     pass

    pos1, pks1, vp1, kos1, vk1 = sig1.detail_names_by_kind()
    for args, kwargs in sig_to_inputs(sig1, variadics_source=((), {})):
        if vp1 is not None and len(args) == len(pos1) + len(pks1):
            args += ("extra_arg",)
        if vk1 is not None:
            kwargs["extra_kwarg_key"] = "extra_kwarg_value"
        try:
            foo(*args, **kwargs)
        except TypeError:
            if is_compatible:
                raise
            else:
                return
    assert (
        is_compatible
    ), f"sig1 is not compatible with sig2, when it should: {sig1} and {sig2}"


def test_bool():
    name = "bool"

    sig = Sig(sigs_for_sigless_builtin_name[name])

    assert function_is_compatible_with_signature(bool, sig)


def visualize_errors_for_function_call(func, sig):
    """
    Calls func on sig_to_inputs and prints error if any
    """

    print(f"============================================================{str(func)}")
    for args, kwargs in sig_to_inputs(sig):
        result = call_and_return_error(func, *args, **kwargs)
        if result is not None:
            print(f"{args=} , {kwargs=}")
            print(result)


def test_sigless_builtins():
    from operator import itemgetter, attrgetter, methodcaller

    mapping_methods = {
        "__eq__",
        "__ne__",
        "__iter__",
        "__getitem__",
        "__len__",
        "__contains__",
        "__setitem__",
        "__delitem__",
    }
    special_cases = {"breakpoint"} | mapping_methods

    for name in sigs_for_sigless_builtin_name:
        # removed breakpoint as it triggers a pdb session
        if name in special_cases:
            continue
        sig = Sig(sigs_for_sigless_builtin_name[name])
        assert function_is_compatible_with_signature(eval(name), sig)

    d = {"a": 1, "b": 2}
    for name in mapping_methods:
        method = getattr(d, name)
        assert function_is_compatible_with_signature(method, Sig(method))


@pytest.mark.parametrize(
    "sig_spec, args, kwargs, map_arguments_kwargs, expected_output",
    [
        # ------------------------------------------------------------------------------
        # PO
        # ------------------------------------------------------------------------------
        ("(a, /)", (1,), None, None, {"a": 1}),
        ("(a, /)", None, None, None, (TypeError, "missing a required argument: 'a'")),
        ("(a, /)", None, None, dict(allow_partial=True), {}),
        ("(a=0, /)", None, None, dict(apply_defaults=True), {"a": 0}),
        (
            "(a=0, /)",
            None,
            None,
            dict(allow_partial=True, apply_defaults=True),
            {"a": 0},
        ),
        ("(a, /)", (1, 2), None, None, (TypeError, "too many positional arguments")),
        ("(a, /)", (1, 2), None, dict(allow_excess=True), {"a": 1}),
        (
            "(a, /)",
            (1,),
            {"b": 2},
            None,
            (TypeError, "got an unexpected keyword argument 'b'"),
        ),
        ("(a, /)", (1,), {"b": 2}, dict(allow_excess=True), {"a": 1}),
        (
            "(a, /)",
            None,
            {"a": 1},
            None,
            (
                TypeError,
                "'a' parameter is positional only, but was passed as a keyword",
            ),
        ),
        ("(a, /)", None, {"a": 1}, dict(ignore_kind=True), {"a": 1}),
        (
            "(a, /)",
            None,
            {"a": 1, "b": 2},
            dict(allow_excess=True, ignore_kind=True),
            {"a": 1},
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PK
        # ------------------------------------------------------------------------------
        ("(a)", (1,), None, None, {"a": 1}),
        ("(a)", None, {"a": 1}, None, {"a": 1}),
        ("(a)", None, None, None, (TypeError, "missing a required argument: 'a'")),
        ("(a)", None, None, dict(allow_partial=True), {}),
        ("(a=0)", None, None, dict(apply_defaults=True), {"a": 0}),
        ("(a=0)", None, None, dict(apply_defaults=True, allow_partial=True), {"a": 0}),
        ("(a)", (1, 2), None, None, (TypeError, "too many positional arguments")),
        ("(a)", (1, 2), None, dict(allow_excess=True), {"a": 1}),
        (
            "(a)",
            (1,),
            {"b": 2},
            None,
            (TypeError, "got an unexpected keyword argument 'b'"),
        ),
        ("(a)", (1,), {"b": 2}, dict(allow_excess=True), {"a": 1}),
        (
            "(a)",
            None,
            {"a": 1, "b": 2},
            None,
            (TypeError, "got an unexpected keyword argument 'b'"),
        ),
        ("(a)", None, {"a": 1, "b": 2}, dict(allow_excess=True), {"a": 1}),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # VP
        # ------------------------------------------------------------------------------
        ("(*args)", (1, 2), None, None, {"args": (1, 2)}),
        ("(*args)", None, None, None, {}),
        ("(*args)", None, None, dict(allow_partial=True), {}),
        ("(*args)", None, None, dict(apply_defaults=True), {"args": ()}),
        (
            "(*args)",
            None,
            None,
            dict(allow_partial=True, apply_defaults=True),
            {"args": ()},
        ),
        ("(*args)", (1, 2), None, dict(allow_excess=True), {"args": (1, 2)}),
        (
            "(*args)",
            (1, 2),
            {"a": 1, "b": 2},
            None,
            (TypeError, "got an unexpected keyword argument 'a'"),
        ),
        (
            "(*args)",
            (1, 2),
            {"a": 1, "b": 2},
            dict(allow_excess=True),
            {"args": (1, 2)},
        ),
        ("(*args)", ((1, 2),), None, None, {"args": ((1, 2),)}),
        ("(*args)", ((1, 2),), None, dict(ignore_kind=True), {"args": (1, 2)}),
        (
            "(*args)",
            None,
            {"args": (1, 2)},
            None,
            (TypeError, "got an unexpected keyword argument 'args'"),
        ),
        ("(*args)", None, {"args": (1, 2)}, dict(ignore_kind=True), {"args": (1, 2)}),
        (
            "(*args)",
            ((1, 2), 3),
            {"a": 4, "b": 5},
            dict(allow_excess=True, ignore_kind=True),
            {"args": (1, 2)},
        ),
        (
            "(*args)",
            (1, 2),
            {"args": (1, 2)},
            dict(allow_excess=True, ignore_kind=True),
            (TypeError, "multiple values for argument 'args'"),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # KO
        # def foo(*, a): ...
        # ------------------------------------------------------------------------------
        ("(*, a)", None, {"a": 1}, None, {"a": 1}),
        ("(*, a)", None, None, None, (TypeError, "missing a required argument: 'a'")),
        ("(*, a)", None, None, dict(allow_partial=True), {}),
        ("(*, a=0)", None, None, dict(apply_defaults=True), {"a": 0}),
        (
            "(*, a=0)",
            None,
            None,
            dict(allow_partial=True, apply_defaults=True),
            {"a": 0},
        ),
        ("(*, a)", (2,), {"a": 1}, None, (TypeError, "too many positional arguments")),
        ("(*, a)", (2,), {"a": 1}, dict(allow_excess=True), {"a": 1}),
        (
            "(*, a)",
            None,
            {"a": 1, "b": 2},
            None,
            (TypeError, "got an unexpected keyword argument 'b'"),
        ),
        ("(*, a)", None, {"a": 1, "b": 2}, dict(allow_excess=True), {"a": 1}),
        ("(*, a)", (1,), None, None, (TypeError, "too many positional arguments")),
        ("(*, a)", (1,), None, dict(ignore_kind=True), {"a": 1}),
        ("(*, a)", (1, 2), None, dict(allow_excess=True, ignore_kind=True), {"a": 1}),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # VK
        # def foo(**kwargs): ...
        # ------------------------------------------------------------------------------
        ("(**kwargs)", None, {"a": 1, "b": 2}, None, {"kwargs": {"a": 1, "b": 2}}),
        ("(**kwargs)", None, None, None, {}),
        ("(**kwargs)", None, None, dict(allow_partial=True), {}),
        ("(**kwargs)", None, None, dict(apply_defaults=True), {"kwargs": {}}),
        (
            "(**kwargs)",
            None,
            None,
            dict(allow_partial=True, apply_defaults=True),
            {"kwargs": {}},
        ),
        (
            "(**kwargs)",
            None,
            {"a": 1, "b": 2},
            dict(allow_excess=True),
            {"kwargs": {"a": 1, "b": 2}},
        ),
        (
            "(**kwargs)",
            (1, 2),
            {"a": 1, "b": 2},
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(**kwargs)",
            (1, 2),
            {"a": 1, "b": 2},
            dict(allow_excess=True),
            {"kwargs": {"a": 1, "b": 2}},
        ),
        (
            "(**kwargs)",
            None,
            {"kwargs": {"a": 1, "b": 2}},
            None,
            {"kwargs": {"kwargs": {"a": 1, "b": 2}}},
        ),
        (
            "(**kwargs)",
            None,
            {"kwargs": {"a": 1, "b": 2}},
            dict(ignore_kind=True),
            {"kwargs": {"a": 1, "b": 2}},
        ),
        (
            "(**kwargs)",
            ({"a": 1, "b": 2},),
            None,
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(**kwargs)",
            ({"a": 1, "b": 2},),
            None,
            dict(ignore_kind=True),
            {"kwargs": {"a": 1, "b": 2}},
        ),
        (
            "(**kwargs)",
            ({"a": 1, "b": 2},),
            {"kwargs": {"a": 1, "b": 2}},
            dict(allow_excess=True, ignore_kind=True),
            (TypeError, "multiple values for argument 'kwargs'"),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + KO
        # def foo(a, /, b, *, c): ...
        # ------------------------------------------------------------------------------
        ("(a, /, b, *, c)", (1, 2), {"c": 3}, None, {"a": 1, "b": 2, "c": 3}),
        ("(a, /, b, *, c)", (1,), {"b": 2, "c": 3}, None, {"a": 1, "b": 2, "c": 3}),
        (
            "(a, /, b, *, c)",
            (1, 2),
            {"b": 2, "c": 3},
            None,
            (TypeError, "multiple values for argument 'b'"),
        ),
        (
            "(a, /, b, *, c)",
            None,
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        (
            "(a, /, b, *, c)",
            (1,),
            None,
            None,
            (TypeError, "missing a required argument: 'b'"),
        ),
        (
            "(a, /, b, *, c)",
            (1, 2),
            None,
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        (
            "(a, /, b, *, c)",
            (1,),
            {"b": 2},
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        ("(a, /, b, *, c)", None, None, dict(allow_partial=True), {}),
        ("(a, /, b, *, c)", (1,), None, dict(allow_partial=True), {"a": 1}),
        ("(a, /, b, *, c)", (1, 2), None, dict(allow_partial=True), {"a": 1, "b": 2}),
        ("(a, /, b, *, c)", (1,), {"b": 2}, dict(allow_partial=True), {"a": 1, "b": 2}),
        (
            "(a=0, /, b=0, *, c=0)",
            None,
            None,
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "c": 0},
        ),
        (
            "(a=0, /, b=0, *, c=0)",
            (1,),
            None,
            dict(apply_defaults=True),
            {"a": 1, "b": 0, "c": 0},
        ),
        (
            "(a=0, /, b=0, *, c=0)",
            (1, 2),
            None,
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "c": 0},
        ),
        (
            "(a=0, /, b=0, *, c=0)",
            (1,),
            {"b": 2},
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "c": 0},
        ),
        (
            "(a=0, /, b=0, *, c=0)",
            None,
            {"b": 2},
            dict(apply_defaults=True),
            {"a": 0, "b": 2, "c": 0},
        ),
        (
            "(a=0, /, b=0, *, c=0)",
            None,
            {"c": 3},
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "c": 3},
        ),
        (
            "(a, /, b, *, c)",
            (1, 2, 3),
            {"c": 3},
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(a, /, b, *, c)",
            (1, 2, 3),
            {"c": 3},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            "(a, /, b, *, c)",
            (1, 2),
            {"c": 3, "d": 4},
            None,
            (TypeError, "got an unexpected keyword argument 'd'"),
        ),
        (
            "(a, /, b, *, c)",
            (1, 2),
            {"c": 3, "d": 4},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            "(a, /, b, *, c)",
            (1, 2, 3),
            {"c": 3, "d": 4},
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(a, /, b, *, c)",
            (1, 2, 3),
            {"c": 3, "d": 4},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            "(a, /, b, *, c)",
            (1, 2),
            {"b": 2, "c": 3},
            dict(allow_excess=True),
            (TypeError, "multiple values for argument 'b'"),
        ),
        (
            "(a, /, b, *, c)",
            (1, 2, 3),
            None,
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(a, /, b, *, c)",
            (1, 2, 3),
            None,
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            "(a, /, b, *, c)",
            None,
            {"a": 1, "b": 2, "c": 3},
            None,
            (
                TypeError,
                "'a' parameter is positional only, but was passed as a keyword",
            ),
        ),
        (
            "(a, /, b, *, c)",
            None,
            {"a": 1, "b": 2, "c": 3},
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            "(a, /, b, *, c)",
            None,
            {"a": 1, "b": 2, "c": 3},
            dict(allow_excess=True, ignore_kind=True),
            {"a": 1, "b": 2, "c": 3},
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # VP + VK
        # def foo(*args, **kwargs): ...
        # ------------------------------------------------------------------------------
        (
            "(*args, **kwargs)",
            (1, 2, 3),
            {"a": 1, "b": 2, "c": 3},
            None,
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        ("(*args, **kwargs)", (1, 2, 3), None, None, {"args": (1, 2, 3)}),
        (
            "(*args, **kwargs)",
            None,
            {"a": 1, "b": 2, "c": 3},
            None,
            {"kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        ("(*args, **kwargs)", None, None, None, {}),
        ("(*args, **kwargs)", None, None, dict(allow_partial=True), {}),
        (
            "(*args, **kwargs)",
            None,
            None,
            dict(apply_defaults=True),
            {"args": (), "kwargs": {}},
        ),
        (
            "(*args, **kwargs)",
            None,
            None,
            dict(allow_partial=True, apply_defaults=True),
            {"args": (), "kwargs": {}},
        ),
        (
            "(*args, **kwargs)",
            (1, 2, 3),
            {"a": 1, "b": 2, "c": 3},
            dict(allow_excess=True),
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        (
            "(*args, **kwargs)",
            ((1, 2, 3), {"a": 1, "b": 2, "c": 3}),
            None,
            dict(ignore_kind=True),
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        (
            "(*args, **kwargs)",
            ((1, 2, 3),),
            {"kwargs": {"a": 1, "b": 2, "c": 3}},
            dict(ignore_kind=True),
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        (
            "(*args, **kwargs)",
            None,
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
            dict(ignore_kind=True),
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        (
            "(*args, **kwargs)",
            ((1, 2, 3), {"a": 1, "b": 2, "c": 3}),
            None,
            dict(allow_excess=True, ignore_kind=True),
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        (
            "(*args, **kwargs)",
            ((1, 2, 3),),
            {"kwargs": {"a": 1, "b": 2, "c": 3}},
            dict(allow_excess=True, ignore_kind=True),
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        (
            "(*args, **kwargs)",
            None,
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
            dict(allow_excess=True, ignore_kind=True),
            {"args": (1, 2, 3), "kwargs": {"a": 1, "b": 2, "c": 3}},
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + VP + KO
        # def foo(a, /, b, *args, c): ...
        # ------------------------------------------------------------------------------
        (
            "(a, /, b, *args, c)",
            (1, 2, 3, 4),
            {"c": 5},
            None,
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2, 3, 4),
            {"b": 2, "c": 5},
            None,
            (TypeError, "multiple values for argument 'b'"),
        ),
        (
            "(a, /, b, *args, c)",
            None,
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        (
            "(a, /, b, *args, c)",
            (1,),
            None,
            None,
            (TypeError, "missing a required argument: 'b'"),
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2),
            None,
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2, 3, 4),
            None,
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        (
            "(a, /, b, *args, c)",
            (1,),
            {"b": 2},
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        ("(a, /, b, *args, c)", None, None, dict(allow_partial=True), {}),
        ("(a, /, b, *args, c)", (1,), None, dict(allow_partial=True), {"a": 1}),
        (
            "(a, /, b, *args, c)",
            (1, 2),
            None,
            dict(allow_partial=True),
            {"a": 1, "b": 2},
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2, 3, 4),
            None,
            dict(allow_partial=True),
            {"a": 1, "b": 2, "args": (3, 4)},
        ),
        (
            "(a, /, b, *args, c)",
            (1,),
            {"b": 2},
            dict(allow_partial=True),
            {"a": 1, "b": 2},
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            None,
            None,
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "args": (), "c": 0},
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            (1,),
            None,
            dict(apply_defaults=True),
            {"a": 1, "b": 0, "args": (), "c": 0},
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            (1, 2),
            None,
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "args": (), "c": 0},
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            (1, 2, 3, 4),
            None,
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 0},
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            (1,),
            {"b": 2},
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "args": (), "c": 0},
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            None,
            {"b": 2},
            dict(apply_defaults=True),
            {"a": 0, "b": 2, "args": (), "c": 0},
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            None,
            {"c": 3},
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "args": (), "c": 3},
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2, 3, 4, 5),
            {"c": 5},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "args": (3, 4, 5), "c": 5},
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2, 3, 4),
            {"c": 5, "d": 6},
            None,
            (TypeError, "got an unexpected keyword argument 'd'"),
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2, 3, 4),
            {"c": 5, "d": 6},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
        ),
        (
            "(a, /, b, *args, c)",
            (1, 2, (3, 4), 5),
            None,
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
        ),
        (
            "(a, /, b, *args, c)",
            (),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
            None,
            (
                TypeError,
                "'a' parameter is positional only, but was passed as a keyword",
            ),
        ),
        (
            "(a, /, b, *args, c)",
            (),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
        ),
        (
            "(a, /, b, *args, c)",
            (),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "d": 6},
            dict(allow_excess=True, ignore_kind=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
        ),
        (
            "(a, /, b, *args, c)",
            (3, 4),
            {"a": 1, "b": 2, "c": 5},
            None,
            (TypeError, "multiple values for argument 'b'"),
        ),
        (
            "(a, /, b, *args, c)",
            (3, 4),
            {"a": 1, "b": 2, "c": 5},
            dict(ignore_kind=True),
            (TypeError, "multiple values for argument 'a'"),
        ),
        (
            "(a, /, b, *args, c)",
            (3, 4),
            {"a": 1, "b": 2, "c": 5},
            dict(allow_excess=True, ignore_kind=True),
            (TypeError, "multiple values for argument 'a'"),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + KO + VK
        # def foo(a, /, b, *, c, **kwargs): ...
        # ------------------------------------------------------------------------------
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            {"c": 3, "d": 4, "e": 5},
            None,
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            {"b": 2, "c": 3, "d": 4, "e": 5},
            None,
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            None,
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            None,
            None,
            (TypeError, "missing a required argument: 'b'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            None,
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            {"b": 2},
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            None,
            {"d": 4, "e": 5},
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            {"d": 4, "e": 5},
            None,
            (TypeError, "missing a required argument: 'b'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            {"d": 4, "e": 5},
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            {"b": 2, "d": 4, "e": 5},
            None,
            (TypeError, "missing a required argument: 'c'"),
        ),
        ("(a, /, b, *, c, **kwargs)", None, None, dict(allow_partial=True), {}),
        ("(a, /, b, *, c, **kwargs)", (1,), None, dict(allow_partial=True), {"a": 1}),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            None,
            dict(allow_partial=True),
            {"a": 1, "b": 2},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            {"b": 2},
            dict(allow_partial=True),
            {"a": 1, "b": 2},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            None,
            {"d": 4, "e": 5},
            dict(allow_partial=True),
            {"kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            {"d": 4, "e": 5},
            dict(allow_partial=True),
            {"a": 1, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            {"d": 4, "e": 5},
            dict(allow_partial=True),
            {"a": 1, "b": 2, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1,),
            {"b": 2, "d": 4, "e": 5},
            dict(allow_partial=True),
            {"a": 1, "b": 2, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            None,
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "c": 0, "kwargs": {}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            (1,),
            None,
            dict(apply_defaults=True),
            {"a": 1, "b": 0, "c": 0, "kwargs": {}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            (1, 2),
            None,
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "c": 0, "kwargs": {}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            (1,),
            {"b": 2},
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "c": 0, "kwargs": {}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            {"b": 2},
            dict(apply_defaults=True),
            {"a": 0, "b": 2, "c": 0, "kwargs": {}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            {"c": 3},
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "c": 3, "kwargs": {}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            {"d": 4, "e": 5},
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "c": 0, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            (1,),
            {"d": 4, "e": 5},
            dict(apply_defaults=True),
            {"a": 1, "b": 0, "c": 0, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            (1, 2),
            {"d": 4, "e": 5},
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "c": 0, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            (1,),
            {"b": 2, "d": 4, "e": 5},
            dict(apply_defaults=True),
            {"a": 1, "b": 2, "c": 0, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            {"b": 2, "d": 4, "e": 5},
            dict(apply_defaults=True),
            {"a": 0, "b": 2, "c": 0, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            {"c": 3, "d": 4, "e": 5},
            dict(apply_defaults=True),
            {"a": 0, "b": 0, "c": 3, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3),
            {"c": 3},
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3),
            {"c": 3},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            {"c": 3, "d": 4},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3),
            {"c": 3, "d": 4},
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3),
            {"c": 3, "d": 4},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            {"b": 2, "c": 3},
            dict(allow_excess=True),
            (TypeError, "multiple values for argument 'b'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2),
            {"b": 2, "c": 3, "d": 4, "e": 5},
            dict(allow_excess=True),
            (TypeError, "multiple values for argument 'b'"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3, {"d": 4, "e": 5}),
            None,
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3, {"d": 4, "e": 5}),
            None,
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            None,
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
            None,
            (
                TypeError,
                "'a' parameter is positional only, but was passed as a keyword",
            ),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            None,
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            None,
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}, "f": 6},
            dict(allow_excess=True, ignore_kind=True),
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3),
            {"kwargs": {"d": 4, "e": 5}},
            None,
            (TypeError, "too many positional arguments"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            (1, 2, 3),
            {"kwargs": {"d": 4, "e": 5}},
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + VP + KO + VK
        # def foo(a, /, b, *args, c, **kwargs): ...
        # ------------------------------------------------------------------------------
        (
            "(a, /, b, *args, c, **kwargs)",
            (1, 2, 3, 4),
            {"c": 5, "d": 6, "e": 7},
            None,
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            (1,),
            {"b": 2, "c": 5, "d": 6, "e": 7},
            None,
            {"a": 1, "b": 2, "c": 5, "kwargs": {"d": 6, "e": 7}},
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            (1, 2),
            {"c": 5},
            None,
            {"a": 1, "b": 2, "c": 5},
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            None,
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        ("(a, /, b, *args, c, **kwargs)", None, None, dict(allow_partial=True), {}),
        (
            "(a=0, /, b=0, *args, c=0, **kwargs)",
            None,
            None,
            dict(apply_defaults=True),
            {"a": 0, "args": (), "b": 0, "c": 0, "kwargs": {}},
        ),
        (
            "(a=0, /, b=0, *args, c=0, **kwargs)",
            None,
            None,
            dict(apply_defaults=True, allow_partial=True),
            {"a": 0, "args": (), "b": 0, "c": 0, "kwargs": {}},
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            (1, 2, 3, 4),
            {"c": 5, "d": 6, "e": 7},
            dict(allow_excess=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            None,
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
            None,
            (
                TypeError,
                "'a' parameter is positional only, but was passed as a keyword",
            ),
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            None,
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
            dict(ignore_kind=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            None,
            {
                "a": 1,
                "b": 2,
                "args": (3, 4),
                "c": 5,
                "kwargs": {"d": 6, "e": 7},
                "f": 8,
            },
            dict(allow_excess=True, ignore_kind=True),
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
        ),
        # ------------------------------------------------------------------------------
    ],
)
def test_map_arguments(sig_spec, args, kwargs, map_arguments_kwargs, expected_output):
    sig = Sig(sig_spec)
    map_arguments_kwargs = map_arguments_kwargs or {}
    call = lambda: sig.map_arguments(args, kwargs, **map_arguments_kwargs)
    _test_call(call, expected_output)


@pytest.mark.parametrize(
    "sig_spec, arguments, mk_args_and_kwargs_kw, expected_output",
    [
        # ------------------------------------------------------------------------------
        # PO
        # ------------------------------------------------------------------------------
        ("(a, /)", {"a": 1}, None, ((1,), {})),
        ("(a, /)", None, None, (TypeError, "missing a required argument: 'a'")),
        ("(a, /)", None, dict(allow_partial=True), ((), {})),
        ("(a=0, /)", None, dict(apply_defaults=True), ((0,), {})),
        ("(a=0, /)", None, dict(allow_partial=True, apply_defaults=True), ((0,), {})),
        ("(a=0, /)", None, dict(ignore_kind=True, apply_defaults=True), ((), {"a": 0})),
        (
            "(a, /)",
            {"a": 1, "b": 2},
            None,
            (TypeError, "Got unexpected keyword arguments: b"),
        ),
        ("(a, /)", {"a": 1, "b": 2}, dict(allow_excess=True), ((1,), {})),
        ("(a, /)", {"a": 1}, dict(ignore_kind=True), ((), {"a": 1})),
        (
            "(a, /)",
            {"a": 1, "b": 2},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"a": 1}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PK
        # ------------------------------------------------------------------------------
        ("(a)", {"a": 1}, None, ((), {"a": 1})),
        ("(a)", None, None, (TypeError, "missing a required argument: 'a'")),
        ("(a)", None, dict(allow_partial=True), ((), {})),
        ("(a=0)", None, dict(apply_defaults=True), ((), {"a": 0})),
        ("(a=0)", None, dict(allow_partial=True, apply_defaults=True), ((), {"a": 0})),
        ("(a=0)", None, dict(ignore_kind=True, apply_defaults=True), ((), {"a": 0})),
        (
            "(a)",
            {"a": 1, "b": 2},
            None,
            (TypeError, "Got unexpected keyword arguments: b"),
        ),
        ("(a)", {"a": 1, "b": 2}, dict(allow_excess=True), ((), {"a": 1})),
        ("(a)", {"a": 1}, dict(ignore_kind=True), ((), {"a": 1})),
        (
            "(a)",
            {"a": 1, "b": 2},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"a": 1}),
        ),
        ("(a)", {"a": 1}, dict(args_limit=None), ((1,), {})),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # VP
        # ------------------------------------------------------------------------------
        ("(*args)", {"args": (1, 2)}, None, ((1, 2), {})),
        ("(*args)", None, None, ((), {})),
        ("(*args)", None, dict(allow_partial=True), ((), {})),
        ("(*args)", None, dict(apply_defaults=True), ((), {})),
        ("(*args)", None, dict(allow_partial=True, apply_defaults=True), ((), {})),
        (
            "(*args)",
            None,
            dict(ignore_kind=True, apply_defaults=True),
            ((), {"args": ()}),
        ),
        (
            "(*args)",
            {"args": (1, 2), "a": 3},
            None,
            (TypeError, "Got unexpected keyword arguments: a"),
        ),
        ("(*args)", {"args": (1, 2), "a": 3}, dict(allow_excess=True), ((1, 2), {})),
        ("(*args)", {"args": (1, 2)}, dict(ignore_kind=True), ((), {"args": (1, 2)})),
        (
            "(*args)",
            {"args": (1, 2), "a": 3},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"args": (1, 2)}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # KO
        # def foo(*, a): ...
        # ------------------------------------------------------------------------------
        ("(*, a)", {"a": 1}, None, ((), {"a": 1})),
        ("(*, a)", None, None, (TypeError, "missing a required argument: 'a'")),
        ("(*, a)", None, dict(allow_partial=True), ((), {})),
        ("(*, a=0)", None, dict(apply_defaults=True), ((), {"a": 0})),
        (
            "(*, a=0)",
            None,
            dict(allow_partial=True, apply_defaults=True),
            ((), {"a": 0}),
        ),
        ("(*, a=0)", None, dict(ignore_kind=True, apply_defaults=True), ((), {"a": 0})),
        (
            "(*, a)",
            {"a": 1, "b": 2},
            None,
            (TypeError, "Got unexpected keyword arguments: b"),
        ),
        ("(*, a)", {"a": 1, "b": 2}, dict(allow_excess=True), ((), {"a": 1})),
        ("(*, a)", {"a": 1}, dict(ignore_kind=True), ((), {"a": 1})),
        (
            "(*, a)",
            {"a": 1, "b": 2},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"a": 1}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # VK
        # def foo(**kwargs): ...
        # ------------------------------------------------------------------------------
        ("(**kwargs)", {"kwargs": {"a": 1, "b": 2}}, None, ((), {"a": 1, "b": 2})),
        ("(**kwargs)", None, None, ((), {})),
        ("(**kwargs)", None, dict(allow_partial=True), ((), {})),
        ("(**kwargs)", None, dict(apply_defaults=True), ((), {})),
        ("(**kwargs)", None, dict(allow_partial=True, apply_defaults=True), ((), {})),
        (
            "(**kwargs)",
            None,
            dict(ignore_kind=True, apply_defaults=True),
            ((), {"kwargs": {}}),
        ),
        (
            "(**kwargs)",
            {"kwargs": {"a": 1, "b": 2}, "c": 3},
            None,
            (TypeError, "Got unexpected keyword arguments: c"),
        ),
        (
            "(**kwargs)",
            {"kwargs": {"a": 1, "b": 2}, "c": 3},
            dict(allow_excess=True),
            ((), {"a": 1, "b": 2}),
        ),
        (
            "(**kwargs)",
            {"kwargs": {"a": 1, "b": 2}},
            dict(ignore_kind=True),
            ((), {"kwargs": {"a": 1, "b": 2}}),
        ),
        (
            "(**kwargs)",
            {"kwargs": {"a": 1, "b": 2}, "c": 3},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"kwargs": {"a": 1, "b": 2}}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + KO
        # def foo(a, /, b, *, c): ...
        # ------------------------------------------------------------------------------
        ("(a, /, b, *, c)", {"a": 1, "b": 2, "c": 3}, None, ((1,), {"b": 2, "c": 3})),
        (
            "(a, /, b, *, c)",
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        ("(a, /, b, *, c)", None, dict(allow_partial=True), ((), {})),
        (
            "(a=0, /, b=0, *, c=0)",
            None,
            dict(apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *, c=0)",
            None,
            dict(allow_partial=True, apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *, c=0)",
            None,
            dict(ignore_kind=True, apply_defaults=True),
            ((), {"a": 0, "b": 0, "c": 0}),
        ),
        (
            "(a, /, b, *, c)",
            {"a": 1, "b": 2, "c": 3, "d": 4},
            None,
            (TypeError, "Got unexpected keyword arguments: d"),
        ),
        (
            "(a, /, b, *, c)",
            {"a": 1, "b": 2, "c": 3, "d": 4},
            dict(allow_excess=True),
            ((1,), {"b": 2, "c": 3}),
        ),
        (
            "(a, /, b, *, c)",
            {"a": 1, "b": 2, "c": 3},
            dict(ignore_kind=True),
            ((), {"a": 1, "b": 2, "c": 3}),
        ),
        (
            "(a, /, b, *, c)",
            {"a": 1, "b": 2, "c": 3, "d": 4},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"a": 1, "b": 2, "c": 3}),
        ),
        (
            "(a, /, b, *, c)",
            {"a": 1, "b": 2, "c": 3},
            dict(args_limit=None),
            ((1, 2), {"c": 3}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # VP + VK
        # def foo(*args, **kwargs): ...
        # ------------------------------------------------------------------------------
        (
            "(*args, **kwargs)",
            {"args": (1, 2), "kwargs": {"a": 1, "b": 2}},
            None,
            ((1, 2), {"a": 1, "b": 2}),
        ),
        ("(*args, **kwargs)", None, None, ((), {})),
        ("(*args, **kwargs)", None, dict(allow_partial=True), ((), {})),
        ("(*args, **kwargs)", None, dict(apply_defaults=True), ((), {})),
        (
            "(*args, **kwargs)",
            None,
            dict(allow_partial=True, apply_defaults=True),
            ((), {}),
        ),
        (
            "(*args, **kwargs)",
            None,
            dict(ignore_kind=True, apply_defaults=True),
            ((), {"args": (), "kwargs": {}}),
        ),
        (
            "(*args, **kwargs)",
            {"args": (1, 2), "kwargs": {"a": 1, "b": 2}, "c": 3},
            None,
            (TypeError, "Got unexpected keyword arguments: c"),
        ),
        (
            "(*args, **kwargs)",
            {"args": (1, 2), "kwargs": {"a": 1, "b": 2}, "c": 3},
            dict(allow_excess=True),
            ((1, 2), {"a": 1, "b": 2}),
        ),
        (
            "(*args, **kwargs)",
            {"args": (1, 2), "kwargs": {"a": 1, "b": 2}},
            dict(ignore_kind=True),
            ((), {"args": (1, 2), "kwargs": {"a": 1, "b": 2}}),
        ),
        (
            "(*args, **kwargs)",
            {"args": (1, 2), "kwargs": {"a": 1, "b": 2}, "c": 3},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"args": (1, 2), "kwargs": {"a": 1, "b": 2}}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + VP + KO
        # def foo(a, /, b, *args, c): ...
        # ------------------------------------------------------------------------------
        (
            "(a, /, b, *args, c)",
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
            None,
            ((1, 2, 3, 4), {"c": 5}),
        ),
        (
            "(a, /, b, *args, c)",
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        ("(a, /, b, *args, c)", None, dict(allow_partial=True), ((), {})),
        (
            "(a=0, /, b=0, *args, c=0)",
            None,
            dict(apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            None,
            dict(allow_partial=True, apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *args, c=0)",
            None,
            dict(ignore_kind=True, apply_defaults=True),
            ((), {"a": 0, "b": 0, "args": (), "c": 0}),
        ),
        (
            "(a, /, b, *args, c)",
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "d": 6},
            None,
            (TypeError, "Got unexpected keyword arguments: d"),
        ),
        (
            "(a, /, b, *args, c)",
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "d": 6},
            dict(allow_excess=True),
            ((1, 2, 3, 4), {"c": 5}),
        ),
        (
            "(a, /, b, *args, c)",
            {"a": 1, "b": 2, "args": (3, 4), "c": 5},
            dict(ignore_kind=True),
            ((), {"a": 1, "b": 2, "args": (3, 4), "c": 5}),
        ),
        (
            "(a, /, b, *args, c)",
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "d": 6},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"a": 1, "b": 2, "args": (3, 4), "c": 5}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + KO + VK
        # def foo(a, /, b, *, c, **kwargs): ...
        # ------------------------------------------------------------------------------
        (
            "(a, /, b, *, c, **kwargs)",
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
            None,
            ((1,), {"b": 2, "c": 3, "d": 4, "e": 5}),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        ("(a, /, b, *, c, **kwargs)", None, dict(allow_partial=True), ((), {})),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            dict(apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            dict(allow_partial=True, apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *, c=0, **kwargs)",
            None,
            dict(ignore_kind=True, apply_defaults=True),
            ((), {"a": 0, "b": 0, "c": 0, "kwargs": {}}),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}, "f": 6},
            None,
            (TypeError, "Got unexpected keyword arguments: f"),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}, "f": 6},
            dict(allow_excess=True),
            ((1,), {"b": 2, "c": 3, "d": 4, "e": 5}),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
            dict(ignore_kind=True),
            ((), {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}}),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}, "f": 6},
            dict(allow_excess=True, ignore_kind=True),
            ((), {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}}),
        ),
        (
            "(a, /, b, *, c, **kwargs)",
            {"a": 1, "b": 2, "c": 3, "kwargs": {"d": 4, "e": 5}},
            dict(args_limit=None),
            ((1, 2), {"c": 3, "d": 4, "e": 5}),
        ),
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # PO + PK + VP + KO + VK
        # def foo(a, /, b, *args, c, **kwargs): ...
        # ------------------------------------------------------------------------------
        (
            "(a, /, b, *args, c, **kwargs)",
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
            None,
            ((1, 2, 3, 4), {"c": 5, "d": 6, "e": 7}),
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            None,
            None,
            (TypeError, "missing a required argument: 'a'"),
        ),
        ("(a, /, b, *args, c, **kwargs)", None, dict(allow_partial=True), ((), {})),
        (
            "(a=0, /, b=0, *args, c=0, **kwargs)",
            None,
            dict(apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *args, c=0, **kwargs)",
            None,
            dict(allow_partial=True, apply_defaults=True),
            ((0,), {"b": 0, "c": 0}),
        ),
        (
            "(a=0, /, b=0, *args, c=0, **kwargs)",
            None,
            dict(ignore_kind=True, apply_defaults=True),
            ((), {"a": 0, "b": 0, "args": (), "c": 0, "kwargs": {}}),
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            {
                "a": 1,
                "b": 2,
                "args": (3, 4),
                "c": 5,
                "kwargs": {"d": 6, "e": 7},
                "f": 8,
            },
            None,
            (TypeError, "Got unexpected keyword arguments: f"),
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            {
                "a": 1,
                "b": 2,
                "args": (3, 4),
                "c": 5,
                "kwargs": {"d": 6, "e": 7},
                "f": 8,
            },
            dict(allow_excess=True),
            ((1, 2, 3, 4), {"c": 5, "d": 6, "e": 7}),
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}},
            dict(ignore_kind=True),
            ((), {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}}),
        ),
        (
            "(a, /, b, *args, c, **kwargs)",
            {
                "a": 1,
                "b": 2,
                "args": (3, 4),
                "c": 5,
                "kwargs": {"d": 6, "e": 7},
                "f": 8,
            },
            dict(allow_excess=True, ignore_kind=True),
            ((), {"a": 1, "b": 2, "args": (3, 4), "c": 5, "kwargs": {"d": 6, "e": 7}}),
        ),
        # ------------------------------------------------------------------------------
    ],
)
def test_mk_args_and_kwargs(
    sig_spec, arguments, mk_args_and_kwargs_kw, expected_output
):
    sig = Sig(sig_spec)
    mk_args_and_kwargs_kw = mk_args_and_kwargs_kw or {}
    call = lambda: sig.mk_args_and_kwargs(arguments, **mk_args_and_kwargs_kw)
    _test_call(call, expected_output)


def _test_call(call, expected_output):
    if (
        isinstance(expected_output, tuple)
        and isinstance(expected_output[0], type)
        and issubclass(expected_output[0], Exception)
    ):
        err, err_msg = expected_output
        with pytest.raises(err, match=err_msg):
            call()
    else:
        assert call() == expected_output

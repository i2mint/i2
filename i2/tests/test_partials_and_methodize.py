"""Tests for partialx(_rm_partialize), curried, methodize(default_from) and the
docstring parser fixes of i2mint/i2#111, #112, #113, #116 and #118."""

import asyncio
import inspect
from dataclasses import dataclass
from functools import partial

import pytest

from i2 import Sig, curried, methodize, partialx
from i2.doc_mint import detect_doc_style, docstring_to_params


def test_rm_partialize_keeps_a_partial_and_binds_positionals_once():
    def h(a, *args, **kw):
        return a, args, kw

    g = partialx(h, 1, 2, _rm_partialize=True)
    assert isinstance(g, partial) and g.func is h
    assert g(3) == (1, (2, 3), {})


def test_rm_partialize_makes_params_after_a_keyword_binding_keyword_only():
    def f(a, b, c=3):
        return a, b, c

    g = partialx(f, b=2, _rm_partialize=True)
    assert str(Sig(g)) == "(a, *, c=3)"
    assert g(1, c=5) == (1, 2, 5)


def test_rm_partialize_keeps_var_params_named_by_a_keyword():
    def h(a, *args, **kw):
        return a, args, kw

    g = partialx(h, kw=3, _rm_partialize=True)
    assert str(Sig(g)) == "(a, *args, **kw)"
    assert g(1) == (1, (), {"kw": 3})


def test_rm_partialize_keeps_coroutine_functions():
    async def af(a, b):
        return a + b

    g = partialx(af, b=2, _rm_partialize=True)
    assert inspect.iscoroutinefunction(g)
    assert asyncio.run(g(1)) == 3


def test_curried():
    def f(a, b, *, c):
        return a, b, c

    g = curried(f)
    assert g(1)(2)(c=3) == g(1, 2, c=3) == g(c=3)(1)(2) == (1, 2, 3)
    with pytest.raises(TypeError):
        g(1, d=4)


def test_curried_async_awaits_on_the_full_call_only():
    async def af(a, b):
        return a + b

    ga = curried(af)
    assert not inspect.iscoroutinefunction(ga)
    assert asyncio.run(ga(1)(2)) == 3


def test_methodize_default_from():
    def foo(x, y, z):
        return x + y * z

    @dataclass
    class K:
        x: int
        zoo: int = 2
        bar = methodize(foo, bind="x", default_from=[("z", "zoo")])

    k = K(1)
    assert str(Sig(k.bar)) == "(y, z=self.zoo)"
    assert (k.bar(3), k.bar(3, 10), k.bar(3, z=0)) == (7, 31, 1)


def test_methodize_default_from_json_dump_of_signature_defaults():
    """A default_from default is a placeholder: JSON consumers skip it, like NotSet."""
    import json

    from i2 import NotSet, is_placeholder_default

    def foo(x, y, z, w=NotSet, v=1):
        return x + y * z

    @dataclass
    class K:
        x: int
        zoo: int = 2
        bar = methodize(foo, bind="x", default_from=[("z", "zoo")])

    defaults = Sig(K(1).bar).defaults
    with pytest.raises(TypeError):
        json.dumps(defaults)  # what a naive consumer would meet
    concrete = {k: v for k, v in defaults.items() if not is_placeholder_default(v)}
    assert json.loads(json.dumps(concrete)) == {"v": 1}


def test_methodize_keeps_self_before_position_only_params():
    """i2mint/i2#118: ``self`` used to land after the ``/``."""

    def foo(a, /, b, *, c=3):
        return (a, b, c)

    @dataclass
    class K:
        a: int = 1
        b: int = 7
        bound = methodize(foo, bind="b")
        defaulted = methodize(foo, default_from="b")

    assert str(inspect.signature(K.bound)) == "(self, a, /, *, c=3)"
    assert str(inspect.signature(K.defaulted)) == "(self, a, /, b=self.b, *, c=3)"
    k = K()
    assert k.bound(5) == (5, 7, 3)
    assert (k.defaulted(5), k.defaulted(5, 0)) == ((5, 7, 3), (5, 0, 3))
    assert methodize(foo, bind="a")(k, 2) == (1, 2, 3)


def test_methodize_default_from_rejects_variadic_and_unknown_params():
    def foo(x, *args, **kwargs):
        return x

    with pytest.raises(ValueError, match="args"):
        methodize(foo, default_from="args")
    with pytest.raises(ValueError, match="not in the method"):
        methodize(foo, default_from="nope")


@pytest.mark.parametrize(
    "line, name, annotation, description",
    [
        (":param x: Note: foo", "x", None, "Note: foo"),
        (":param int x: X.", "x", int, "X."),
        (":param dict[str, int] x: X.", "x", "dict[str, int]", "X."),
    ],
)
def test_rest_param_lines(line, name, annotation, description):
    [param] = docstring_to_params(line, doc_style="rest")
    assert (param["name"], param["annotation"], param["description"]) == (
        name,
        annotation,
        description,
    )


def test_crlf_docstrings():
    doc = "Do it.\r\n\r\nArgs:\r\n    x (int): An integer.\r\n"
    assert detect_doc_style(doc) == "google"
    assert [p["name"] for p in docstring_to_params(doc, doc_style="auto")] == ["x"]

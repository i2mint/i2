"""Tests for ``i2.util.FunctionBuilder`` (argument editing) and ``deprecation_of``."""

import pytest

from i2.util import FunctionBuilder, deprecation_of


def test_arg_names_and_defaults_roundtrip():
    fb = FunctionBuilder(
        "g", args=["a", "b"], defaults=(2,), kwonlyargs=["c"], kwonlydefaults={"c": 3}
    )
    assert fb.get_arg_names() == ("a", "b", "c")
    assert fb.get_arg_names(only_required=True) == ("a",)
    assert fb.get_defaults_dict() == {"b": 2, "c": 3}


def test_add_arg_positional_and_kwonly():
    fb = FunctionBuilder("g", args=["a"])
    fb.add_arg("b", default=5)
    fb.add_arg("c", default=6, kwonly=True)
    fb.add_arg("d", kwonly=True)
    assert fb.get_arg_names() == ("a", "b", "c", "d")
    assert fb.get_defaults_dict() == {"b": 5, "c": 6}


def test_add_arg_duplicate_raises():
    fb = FunctionBuilder("g", args=["a"], kwonlyargs=["k"])
    with pytest.raises(Exception, match="already in func"):
        fb.add_arg("a")
    with pytest.raises(Exception, match="kwonly"):
        fb.add_arg("k")


def test_remove_arg_positional_kwonly_and_missing():
    fb = FunctionBuilder(
        "g", args=["a", "b"], defaults=(2,), kwonlyargs=["c"], kwonlydefaults={"c": 3}
    )
    fb.remove_arg("b")
    assert fb.get_arg_names() == ("a", "c")
    fb.remove_arg("c")
    assert fb.get_arg_names() == ("a",)
    assert fb.get_defaults_dict() == {}
    with pytest.raises(Exception) as ei:
        fb.remove_arg("zzz")
    assert ei.value.arg_name == "zzz"


def test_sig_and_invocation_strings():
    fb = FunctionBuilder("g", args=["a", "b"], defaults=(2,), kwonlyargs=["c"])
    assert (
        fb.get_sig_str(with_annotations=False) == "(a, b, *, c)"
    )  # (as in boltons, defaults are not rendered here)
    assert fb.get_invocation_str() == "a, b, c=c"


def test_get_func_builds_working_function_after_edit():
    fb = FunctionBuilder("g", args=["a"], body="return a + 1")
    fb.add_arg("b", default=10)
    fb.body = "return a + b"
    g = fb.get_func()
    assert g.__name__ == "g"
    assert g(1) == 11
    assert g(1, 2) == 3


def test_unexpected_kwargs_rejected():
    with pytest.raises(TypeError):
        FunctionBuilder("g", bogus=1)


def test_from_func_rejects_non_callables():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(3)


def test_deprecation_of_warns_and_delegates():
    def new(x):
        return x * 2

    old = deprecation_of(new, "old")
    with pytest.warns(DeprecationWarning, match="`old` is deprecated"):
        assert old(4) == 8

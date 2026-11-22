"""Tests for ``i2.base``: mints (Mapping views of an object's interface)."""

import inspect
import warnings

import pytest

from i2.base import (
    AttrFromKey,
    KeyFromAttr,
    Mint,
    MintOfCallable,
    ParameterMint,
    ParametersMint,
    is_not_empty,
    name_of_obj,
    not_found,
    no_name,
)


def f(a, b: int = 2, *, c="x") -> float:
    """Doc of f."""
    return 1.0


def test_sentinels_are_falsy_and_not_empty():
    assert not not_found
    assert repr(not_found) == "NotFound"
    assert not is_not_empty(not_found)
    assert not is_not_empty(no_name)
    assert not is_not_empty(inspect.Parameter.empty)
    assert is_not_empty(0)
    assert is_not_empty(None)


def test_attr_from_key_and_key_from_attr_are_inverse_views():
    assert AttrFromKey({"k": 1}).k == 1
    assert AttrFromKey({"k": 1})["k"] == 1

    class O:
        x = 3

    assert KeyFromAttr(O)["x"] == 3
    assert KeyFromAttr(O).x == 3
    with pytest.raises(KeyError):
        AttrFromKey({})["missing"]


def test_name_of_obj_is_deprecated_but_works():
    with pytest.warns(DeprecationWarning):
        assert name_of_obj(f) == "f"


def test_parameter_mint_from_parameter():
    p = inspect.signature(f).parameters["b"]
    m = ParameterMint(p, position=1)
    assert m["name"] == "b"
    assert m["default"] == 2
    assert m["annotation"] is int
    assert "default" in m
    assert m["position"] == 1
    state = m.__getstate__()
    assert state["name"] == "b" and state["position"] == 1
    assert "'name': 'b'" in repr(m)


def test_parameter_mint_from_mapping_fills_missing_with_empty():
    m = ParameterMint({"name": "z"})
    assert m.name == "z"
    assert m.default is inspect.Parameter.empty
    assert dict(m.items()) == {"name": "z"}
    assert "default" in m  # attribute exists (as empty); membership is attribute-based


def test_parameters_mint_mapping_interface():
    pm = ParametersMint(inspect.signature(f).parameters)
    assert list(pm) == ["a", "b", "c"]
    assert len(pm) == 3
    assert set(pm.keys()) == {"a", "b", "c"}
    assert "b" in pm
    d = pm.to_dict()
    assert d["b"]["default"] == 2
    assert d["c"]["kind"] == inspect.Parameter.KEYWORD_ONLY
    assert repr(pm).startswith("ParametersMint(")


def test_mint_of_function_and_non_callable():
    m = Mint(f)
    assert m.obj_name == "f"
    assert m.type_name == "function"
    assert dict(m)["obj_name"] == "f"
    assert len(m) == len(list(m))
    with pytest.raises(KeyError):
        m["nope"]

    v = Mint(10)
    assert dict(v.items()) == {"type_name": "int"}


def test_mint_rejects_attrs_that_are_not_properties():
    with pytest.raises(AttributeError):
        Mint(f, attrs=["not_a_property"])
    assert list(Mint(f, attrs=["obj_name"])) == ["obj_name"]


def test_mint_of_callable_attributes():
    m = MintOfCallable(f)
    assert m.doc_string == "Doc of f."
    assert m.return_annotation is float
    # (known quirk: params without a default also appear, mapped to ``Parameter.empty``)
    assert m.default_of["b"] == 2 and m.default_of["c"] == "x"
    assert m.annotation_of["b"] is int
    assert list(m.parameters) == ["a", "b", "c"]

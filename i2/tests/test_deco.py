"""Tests for ``i2.deco``.

In particular, pins the *signature contract* of ``FuncFactory`` instances, which
signature-driven consumers (UI generators like ``front``/``streamlitfront``, HTTP/OpenAPI
generators like ``py2http``) introspect.
"""

import inspect

from i2 import FuncFactory, Sig


def _g(wf, chk_size: int, name: str, *, sep="-") -> list:
    return [wf, chk_size, name, sep]


def test_func_factory_signature_keeps_required_params_required():
    """Regression guard for the revert of i2mint/i2#88 (see i2mint/i2#48).

    Giving the factory's non-defaulted params a ``NotSet`` sentinel default made
    signature consumers treat the sentinel as a real default value: ``front`` input
    elements crashed (``int(NotSet)``) or prefilled ``"NotSet"``, and ``py2http``
    produced OpenAPI specs that were not JSON-serializable. Until those consumers
    understand the sentinel, a ``FuncFactory``'s signature must show exactly the
    wrapped function's defaults (no sentinel ones).
    """
    factory = FuncFactory(_g, exclude="wf")
    sig = Sig(factory)

    assert list(sig.names) == ["chk_size", "name", "sep"]
    assert list(sig.required_names) == ["chk_size", "name"]
    assert sig.defaults == {"sep": "-"}
    g_params = inspect.signature(_g).parameters
    for name, p in inspect.signature(factory).parameters.items():
        assert p.default is g_params[name].default

    # The factory itself can still be called with none to all of the arguments
    assert factory()(1, 2, "a") == [1, 2, "a", "-"]
    assert factory(chk_size=3)(1, name="b") == [1, 3, "b", "-"]


def test_not_set_is_exported_and_recognisable():
    """``NotSet`` and ``is_not_set`` are public, so signature consumers can recognise
    the sentinel without importing a private object (see i2mint/i2#48)."""
    import pickle

    import i2
    from i2.deco import NotSet as deco_not_set

    assert i2.NotSet is deco_not_set
    assert i2.is_not_set(i2.NotSet)
    for other in (None, inspect.Parameter.empty, "NotSet", 0, False, object()):
        assert not i2.is_not_set(other)
    # Survives pickling as the same object (so identity checks stay valid)
    assert pickle.loads(pickle.dumps(i2.NotSet)) is i2.NotSet
    assert repr(i2.NotSet) == "NotSet"

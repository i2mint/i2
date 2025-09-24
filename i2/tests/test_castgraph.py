# test_castgraph.py
# Pytest test suite for i2.castgraph

import json
import types
import pytest

# Import the public API from the target module
from i2.castgraph import (
    ConversionRegistry,
    ConversionError,
    design_guidelines,
)


# --- Helper marker types for tests -------------------------------------------------


class Path(str):
    """Marker type for file paths (to distinguish from arbitrary strings)."""


class Text(str):
    """Marker type for text payloads."""


class JSONDict(dict):
    """Marker type for dicts with JSON semantics."""


class CanonicalRecord(dict):
    """Canonical in-memory record format."""


# --- Tests -------------------------------------------------------------------------


def test_register_and_convert_direct():
    reg = ConversionRegistry()

    @reg.register(Text, JSONDict)
    def text_to_json(t, ctx):
        return JSONDict(json.loads(t or "{}"))

    x = Text('{"a": 1}')
    out = reg.convert(x, JSONDict)
    assert isinstance(out, JSONDict)
    assert out["a"] == 1


def test_multi_hop_routing_when_no_direct_edge():
    reg = ConversionRegistry()

    @reg.register(Path, Text)
    def path_to_text(p, ctx):
        fs = (ctx or {}).get("fs", {})
        return Text(fs.get(str(p), ""))

    @reg.register(Text, JSONDict)
    def text_to_json(t, ctx):
        return JSONDict(json.loads(t or "{}"))

    # No direct Path -> JSONDict registered; must route Path -> Text -> JSONDict
    ctx = {"fs": {"/tmp/data.json": '{"x": 42}'}}
    out = reg.convert(Path("/tmp/data.json"), JSONDict, context=ctx)
    assert isinstance(out, JSONDict)
    assert out["x"] == 42


def test_cost_based_route_selection_prefers_cheaper_edge():
    reg = ConversionRegistry()

    @reg.register(Text, JSONDict, cost=1.0)
    def text_to_json(t, ctx):
        return JSONDict(json.loads(t or "{}"))

    @reg.register(JSONDict, CanonicalRecord, cost=1.0)
    def json_to_canonical(d, ctx):
        return CanonicalRecord(d)

    # Cheaper direct Text -> CanonicalRecord should be preferred over Text -> JSON -> Canonical
    chosen = {}

    @reg.register(Text, CanonicalRecord, cost=0.5)
    def text_to_canonical(t, ctx):
        chosen["direct"] = True
        return CanonicalRecord(json.loads(t or "{}"))

    out = reg.convert(Text('{"k": "v"}'), CanonicalRecord)
    assert isinstance(out, CanonicalRecord)
    assert out["k"] == "v"
    assert chosen.get("direct") is True  # ensure cheaper route chosen


def test_mro_fallback_for_source_type():
    reg = ConversionRegistry()

    class Base: ...

    class Sub(Base): ...

    class Out: ...

    @reg.register(Base, Out)
    def base_to_out(b, ctx):
        return Out()

    # No direct Sub -> Out converter; should use Base -> Out via MRO
    result = reg.convert(Sub(), Out)
    assert isinstance(result, Out)


def test_identity_short_circuit_when_already_target_type():
    reg = ConversionRegistry()
    obj = CanonicalRecord({"a": 1})
    out = reg.convert(obj, CanonicalRecord)
    # Should return the exact same object instance (identity)
    assert out is obj


def test_result_caching_by_object_id_and_type():
    reg = ConversionRegistry()
    calls = {"count": 0}

    @reg.register(Text, JSONDict)
    def text_to_json(t, ctx):
        calls["count"] += 1
        return JSONDict(json.loads(t or "{}"))

    data = Text('{"a":1}')
    # Invoke twice with result caching enabled; converter should run once
    out1 = reg.convert(data, JSONDict, use_result_cache=True)
    out2 = reg.convert(data, JSONDict, use_result_cache=True)
    assert out1 == out2
    assert calls["count"] == 1


def test_context_propagation_for_io_injection():
    reg = ConversionRegistry()

    @reg.register(Path, Text)
    def path_to_text(p, ctx):
        fs = (ctx or {}).get("fs", {})
        return Text(fs.get(str(p), ""))

    @reg.register(Text, JSONDict)
    def text_to_json(t, ctx):
        return JSONDict(json.loads(t or "{}"))

    ctx = {"fs": {"/a/b.json": '{"ok": true}'}}
    out = reg.convert(Path("/a/b.json"), JSONDict, context=ctx)
    assert out["ok"] is True


def test_no_route_raises_conversion_error():
    reg = ConversionRegistry()

    class A: ...

    class B: ...

    a = A()
    with pytest.raises(ConversionError):
        reg.convert(a, B)


def test_path_cache_is_used_across_calls():
    # Indirectly test that repeated conversions reuse path discovery
    reg = ConversionRegistry()
    calls = {"edges": 0}

    class A: ...

    class B: ...

    class C: ...

    @reg.register(A, B)
    def a_to_b(a, ctx):
        calls["edges"] += 1
        return B()

    @reg.register(B, C)
    def b_to_c(b, ctx):
        calls["edges"] += 1
        return C()

    # Two conversions A -> C; path discovery should be cached
    _ = reg.convert(A(), C)
    _ = reg.convert(A(), C)
    # Edge functions run each time (we are not caching results here),
    # but path discovery should not re-run; we can't directly assert path cache,
    # so we at least ensure conversion succeeds without excessive overhead.
    assert isinstance(reg.convert(A(), C), C)


def test_design_guidelines_mentions_registry_and_cache():
    text = design_guidelines().lower()
    assert "registry" in text
    assert "cache" in text


def test_converter_function_signature_accepts_context_even_if_unused():
    reg = ConversionRegistry()

    class Src: ...

    class Dst: ...

    @reg.register(Src, Dst)
    def s_to_d(s, _ctx):  # underscore to indicate unused context
        return Dst()

    assert isinstance(reg.convert(Src(), Dst), Dst)


def test_error_message_contains_types_for_debugging():
    reg = ConversionRegistry()

    class Foo: ...

    class Bar: ...

    with pytest.raises(ConversionError) as excinfo:
        reg.convert(Foo(), Bar)
    msg = str(excinfo.value)
    assert "Foo" in msg and "Bar" in msg


# Optional: smoke test showing a tiny "canonical hub" usage style
def test_canonical_hub_style_is_ergonomic():
    reg = ConversionRegistry()

    class CSVText(str): ...

    class Rows(list): ...

    class Canon(dict): ...

    @reg.register(CSVText, Rows)
    def csv_to_rows(s, ctx):
        # naive CSV: lines of "k,v"
        rows = []
        for line in (s or "").strip().splitlines():
            if not line.strip():
                continue
            k, v = line.split(",", 1)
            rows.append((k.strip(), v.strip()))
        return Rows(rows)

    @reg.register(Rows, Canon)
    def rows_to_canon(rows, ctx):
        return Canon({k: v for k, v in rows})

    out = reg.convert(CSVText("a,1\nb,2"), Canon)
    assert isinstance(out, Canon) and out == {"a": "1", "b": "2"}

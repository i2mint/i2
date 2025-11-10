# test_castgraph_kinds.py
# Comprehensive tests for the new kind-based interface in i2.castgraph

import json
import warnings
import pytest

from i2.castgraph import (
    TransformationGraph,
    ConversionError,
    Kind,
    KindMatch,
)


# --- Tests for KindMatch ---


def test_kind_match_is_truthy():
    """KindMatch should evaluate to True."""
    match = KindMatch()
    assert bool(match) is True
    assert match


def test_kind_match_with_metadata():
    """KindMatch can carry metadata."""
    match = KindMatch({"encoding": "utf-8", "analyzed": True})
    assert match.metadata == {"encoding": "utf-8", "analyzed": True}
    assert bool(match) is True


def test_kind_match_repr():
    """KindMatch has readable repr."""
    match = KindMatch({"foo": "bar"})
    assert "KindMatch" in repr(match)
    assert "foo" in repr(match)


# --- Tests for Kind class ---


def test_kind_with_identifier():
    """Kind wraps a hashable identifier."""
    k = Kind("text")
    assert k.identifier == "text"


def test_kind_with_type_identifier_has_implicit_isa():
    """Kind with type identifier automatically uses isinstance."""
    k = Kind(str)
    assert k.isa("hello") is True
    assert k.isa(123) is False


def test_kind_with_custom_isa():
    """Kind can have a custom predicate."""
    k = Kind("json_str", isa=lambda x: isinstance(x, str) and x.startswith("{"))
    assert k.isa('{"key": "value"}')
    assert not k.isa("plain text")


def test_kind_equality():
    """Kinds are equal if identifiers match."""
    k1 = Kind("text")
    k2 = Kind("text")
    assert k1 == k2
    # Can also compare with raw identifier
    assert k1 == "text"


def test_kind_hashable():
    """Kinds are hashable."""
    k1 = Kind("text")
    k2 = Kind("json")
    d = {k1: "value1", k2: "value2"}
    assert d[k1] == "value1"
    assert d[k2] == "value2"


# --- Tests for TransformationGraph basic operations ---


def test_add_node_with_string_kind():
    """Can add nodes with string identifiers."""
    graph = TransformationGraph()
    graph.add_node("text", isa=lambda x: isinstance(x, str))
    assert "text" in graph.kinds()


def test_add_node_with_type_kind():
    """Can add nodes with type identifiers."""
    graph = TransformationGraph()
    graph.add_node(str)
    assert str in graph.kinds()


def test_add_node_with_kind_object():
    """Can add nodes with Kind objects."""
    graph = TransformationGraph()
    text_kind = Kind("text", isa=lambda x: isinstance(x, str))
    graph.add_node(text_kind)
    assert "text" in graph.kinds()


def test_add_edge_basic():
    """Can add transformation edges."""
    graph = TransformationGraph()

    def text_to_int(s, ctx):
        return int(s)

    graph.add_edge("text", int, text_to_int)
    assert "text" in graph.kinds()
    assert int in graph.kinds()


def test_register_edge_decorator():
    """Can register edges using decorator."""
    graph = TransformationGraph()

    @graph.register_edge("text", int)
    def text_to_int(s, ctx):
        return int(s)

    # Should have added both nodes
    assert "text" in graph.kinds()
    assert int in graph.kinds()


def test_register_edge_infers_from_annotations():
    """register_edge can infer kinds from annotations."""
    graph = TransformationGraph()

    @graph.register_edge()
    def str_to_int(s: str, ctx) -> int:
        return int(s)

    # Should have added both types as nodes
    assert str in graph.kinds()
    assert int in graph.kinds()


# --- Tests for transform() ---


def test_transform_basic():
    """Basic transformation works."""
    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        return int(s)

    result = graph.transform("42", int)
    assert result == 42
    assert isinstance(result, int)


def test_transform_with_string_kinds():
    """Can transform using string kinds."""
    graph = TransformationGraph()
    graph.add_node("text", isa=lambda x: isinstance(x, str))
    graph.add_node("number", isa=lambda x: isinstance(x, int))

    @graph.register_edge("text", "number")
    def text_to_number(t, ctx):
        return int(t)

    result = graph.transform("123", "number", from_kind="text")
    assert result == 123


def test_transform_multi_hop():
    """Multi-hop transformations work."""
    graph = TransformationGraph()

    @graph.register_edge(str, float)
    def str_to_float(s, ctx):
        return float(s)

    @graph.register_edge(float, int)
    def float_to_int(f, ctx):
        return int(f)

    # Should route str -> float -> int
    result = graph.transform("42.7", int)
    assert result == 42
    assert isinstance(result, int)


def test_transform_with_context():
    """Context is passed through transformations."""
    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int_with_base(s, ctx):
        base = (ctx or {}).get("base", 10)
        return int(s, base)

    result = graph.transform("FF", int, context={"base": 16})
    assert result == 255


def test_transform_identity():
    """Identity transformations return same object."""
    graph = TransformationGraph()
    graph.add_node(int)

    obj = 42
    result = graph.transform(obj, int)
    assert result is obj


def test_transform_raises_on_no_path():
    """Raises ConversionError when no path exists."""
    graph = TransformationGraph()
    graph.add_node(str)
    graph.add_node(int)
    # No edge between them

    with pytest.raises(ConversionError):
        graph.transform("42", int)


# --- Tests for get_transformer() ---


def test_get_transformer_returns_callable():
    """get_transformer returns a callable."""
    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        return int(s)

    transformer = graph.get_transformer(str, int)
    assert callable(transformer)
    assert transformer("42") == 42


def test_get_transformer_with_baked_context():
    """get_transformer can bake in context."""
    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        base = (ctx or {}).get("base", 10)
        return int(s, base)

    transformer = graph.get_transformer(str, int, context={"base": 16})
    assert transformer("FF") == 255


# --- Tests for kind detection ---


def test_detect_kind_with_predicates():
    """detect_kind uses registered predicates."""
    graph = TransformationGraph()
    graph.add_node("json_str", isa=lambda x: isinstance(x, str) and x.startswith("{"))
    graph.add_node("text", isa=lambda x: isinstance(x, str))

    # Should match json_str (first matching predicate)
    assert graph.detect_kind('{"x": 1}') == "json_str"


def test_detect_kind_with_custom_detector():
    """detect_kind can use a custom detector function."""
    graph = TransformationGraph()

    def my_detector(obj):
        if isinstance(obj, str) and len(obj) > 10:
            return "long_text"
        return None

    graph.set_kind_detector(my_detector)
    assert graph.detect_kind("short") is None
    assert graph.detect_kind("this is a long string") == "long_text"


def test_detect_kind_returns_none_on_no_match():
    """detect_kind returns None if no kind matches."""
    graph = TransformationGraph()
    graph.add_node("text", isa=lambda x: isinstance(x, str))

    assert graph.detect_kind(123) is None


def test_detect_kind_fallback_to_type():
    """detect_kind falls back to type(obj) if registered."""
    graph = TransformationGraph()
    graph.add_node(int)

    assert graph.detect_kind(42) == int


# --- Tests for transform_any() ---


def test_transform_any_with_detection():
    """transform_any detects source kind automatically."""
    graph = TransformationGraph()
    graph.add_node("text", isa=lambda x: isinstance(x, str))
    graph.add_node(int)

    @graph.register_edge("text", int)
    def text_to_int(s, ctx):
        return int(s)

    result = graph.transform_any("42", int)
    assert result == 42


def test_transform_any_raises_if_kind_not_detected():
    """transform_any raises if kind cannot be detected."""
    graph = TransformationGraph()
    graph.add_node(int)

    with pytest.raises(ConversionError):
        graph.transform_any("42", int)  # "42" is str, not registered


# --- Tests for introspection methods ---


def test_reachable_from():
    """reachable_from returns all reachable kinds."""
    graph = TransformationGraph()

    @graph.register_edge(str, float)
    def str_to_float(s, ctx):
        return float(s)

    @graph.register_edge(float, int)
    def float_to_int(f, ctx):
        return int(f)

    reachable = graph.reachable_from(str)
    assert float in reachable
    assert int in reachable


def test_sources_for():
    """sources_for returns all source kinds."""
    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        return int(s)

    @graph.register_edge(float, int)
    def float_to_int(f, ctx):
        return int(f)

    sources = graph.sources_for(int)
    assert str in sources
    assert float in sources


def test_kinds_returns_all_registered():
    """kinds() returns all registered kind identifiers."""
    graph = TransformationGraph()
    graph.add_node("text")
    graph.add_node(int)
    graph.add_node("json")

    all_kinds = graph.kinds()
    assert "text" in all_kinds
    assert int in all_kinds
    assert "json" in all_kinds


# --- Tests for cost-based routing ---


def test_cost_based_routing_prefers_cheaper():
    """Routing prefers lower-cost paths."""
    graph = TransformationGraph()
    chosen = {}

    # Expensive two-hop path: str -> float -> int (cost 2.0)
    @graph.register_edge(str, float, cost=1.0)
    def str_to_float(s, ctx):
        return float(s)

    @graph.register_edge(float, int, cost=1.0)
    def float_to_int(f, ctx):
        return int(f)

    # Cheap direct path: str -> int (cost 0.5)
    @graph.register_edge(str, int, cost=0.5)
    def str_to_int_direct(s, ctx):
        chosen["direct"] = True
        return int(s)

    result = graph.transform("42", int)
    assert result == 42
    assert chosen.get("direct") is True  # Should use cheaper path


# --- Tests for backward compatibility ---


def test_deprecated_register_warns():
    """Deprecated register() method warns."""
    graph = TransformationGraph()

    with pytest.warns(DeprecationWarning):

        @graph.register(str, int)
        def str_to_int(s, ctx):
            return int(s)


def test_deprecated_convert_warns():
    """Deprecated convert() method warns."""
    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        return int(s)

    with pytest.warns(DeprecationWarning):
        result = graph.convert("42", int)

    assert result == 42


def test_old_conversionregistry_still_works():
    """Old ConversionRegistry still works (for now)."""
    from i2.castgraph import ConversionRegistry

    reg = ConversionRegistry()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        @reg.register(str, int)
        def str_to_int(s, ctx):
            return int(s)

        result = reg.convert("42", int)

    assert result == 42


# --- Tests for single-arg converters ---


def test_single_arg_converter_wrapped():
    """Single-arg converters are automatically wrapped."""
    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int(s):  # No context parameter
        return int(s)

    result = graph.transform("42", int)
    assert result == 42


# --- Tests with Kind objects ---


def test_using_kind_objects_for_edges():
    """Can use Kind objects when adding edges."""
    graph = TransformationGraph()

    text_kind = Kind("text", isa=lambda x: isinstance(x, str))
    number_kind = Kind("number", isa=lambda x: isinstance(x, int))

    @graph.register_edge(text_kind, number_kind)
    def text_to_number(t, ctx):
        return int(t)

    result = graph.transform("123", "number", from_kind="text")
    assert result == 123


# --- Tests for edge cases ---


def test_transform_with_result_cache():
    """Result caching works."""
    graph = TransformationGraph()
    calls = {"count": 0}

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        calls["count"] += 1
        return int(s)

    data = "42"
    result1 = graph.transform(data, int, use_result_cache=True)
    result2 = graph.transform(data, int, use_result_cache=True)

    assert result1 == result2 == 42
    assert calls["count"] == 1  # Should only call once


def test_kind_match_in_predicate():
    """Predicates can return KindMatch with metadata."""
    graph = TransformationGraph()

    def json_predicate(obj):
        if isinstance(obj, str) and obj.startswith("{"):
            try:
                json.loads(obj)
                return KindMatch({"valid_json": True})
            except:
                return False
        return False

    graph.add_node("json", isa=json_predicate)

    kind = graph.detect_kind('{"x": 1}')
    assert kind == "json"


# --- Integration test: realistic scenario ---


def test_realistic_file_to_data_pipeline():
    """Test a realistic multi-format transformation pipeline."""
    graph = TransformationGraph()

    # Define kinds
    graph.add_node("filepath", isa=lambda x: isinstance(x, str) and "/" in x)
    graph.add_node("json_text", isa=lambda x: isinstance(x, str) and x.startswith("{"))
    graph.add_node("data_dict", isa=lambda x: isinstance(x, dict))

    # Define transformations
    @graph.register_edge("filepath", "json_text")
    def read_file(path, ctx):
        fs = (ctx or {}).get("fs", {})
        return fs.get(path, "{}")

    @graph.register_edge("json_text", "data_dict")
    def parse_json(text, ctx):
        return json.loads(text)

    # Mock filesystem
    ctx = {"fs": {"/data/config.json": '{"setting": "value"}'}}

    # Multi-hop: filepath -> json_text -> data_dict
    result = graph.transform(
        "/data/config.json", "data_dict", from_kind="filepath", context=ctx
    )

    assert isinstance(result, dict)
    assert result["setting"] == "value"


# --- Test for MRO fallback (type-based) ---


def test_mro_fallback_for_types():
    """MRO fallback works for subclasses."""
    graph = TransformationGraph()

    class Base:
        pass

    class Sub(Base):
        pass

    class Out:
        pass

    @graph.register_edge(Base, Out)
    def base_to_out(b, ctx):
        return Out()

    # No direct Sub -> Out, should use Base -> Out via MRO
    result = graph.transform(Sub(), Out)
    assert isinstance(result, Out)

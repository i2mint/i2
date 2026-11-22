"""Tests for ``i2.routing_forest``: nodes, the default mini-language, and the runnable example."""

import pytest

from i2.routing_forest import (
    CondNode,
    FinalNode,
    RoutingForest,
    RoutingNode,
    SwitchCaseNode,
    test_routing_forest as _routing_forest_example,
    wrap_leafs_with_final_node,
)


def test_module_example_runs():
    # The module ships an end-to-end example as a function; run it under pytest.
    _routing_forest_example()


def test_final_node_yields_its_value():
    assert list(FinalNode("x")(123)) == ["x"]


def test_cond_node_only_fires_when_cond_holds():
    node = CondNode(cond=lambda x: x > 0, then=FinalNode("positive"))
    assert list(node(1)) == ["positive"]
    assert list(node(-1)) == []


def test_routing_forest_collects_all_matching_trees():
    rf = RoutingForest(
        [
            CondNode(lambda x: x > 0, FinalNode("pos")),
            CondNode(lambda x: x % 2 == 0, FinalNode("even")),
        ]
    )
    assert list(rf(2)) == ["pos", "even"]
    assert list(rf(-2)) == ["even"]
    assert list(rf(-1)) == []


def test_switch_case_node_default_and_no_default():
    with_default = SwitchCaseNode(
        switch=lambda x: x % 2,
        cases={0: FinalNode("even")},
        default=FinalNode("other"),
    )
    assert list(with_default(2)) == ["even"]
    assert list(with_default(3)) == ["other"]

    no_default = SwitchCaseNode(switch=lambda x: x % 2, cases={0: FinalNode("even")})
    assert list(no_default(2)) == ["even"]
    with pytest.raises(TypeError):
        list(no_default(3))


def test_wrap_leafs_with_final_node():
    existing = FinalNode("kept")
    out = list(wrap_leafs_with_final_node([existing, "raw"]))
    assert out[0] is existing
    assert isinstance(out[1], FinalNode)
    assert list(out[1](None)) == ["raw"]


def test_from_object_builds_tree_from_tuples():
    node = RoutingNode.from_object(
        [(lambda x: x > 0, FinalNode("pos")), (lambda x: x < 0, FinalNode("neg"))]
    )
    assert isinstance(node, RoutingForest)
    assert list(node(5)) == ["pos"]
    assert list(node(-5)) == ["neg"]
    assert list(node(0)) == []

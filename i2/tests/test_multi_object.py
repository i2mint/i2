"""``MultiObj``'s single-mapping special case and its opt-out (i2#46)."""

from i2.multi_object import ContextFanout, FuncFanout, MultiObj


class _Store(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.events = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, *exc):
        self.events.append("exit")


def test_default_single_mapping_is_the_objects_mapping():
    d = {"this": [1], "that": [1, 2]}
    assert dict(MultiObj(d)) == dict(MultiObj(**d)) == d
    assert dict(FuncFanout({"a": len})) == {"a": len}


def test_single_mapping_with_other_args_is_an_object():
    d = {"x": 1}
    assert list(MultiObj(d, [1]).values()) == [d, [1]]
    assert list(MultiObj(d, extra=2).values()) == [d, 2]


def test_opt_out_holds_a_lone_mapping_as_an_object():
    class StoreFanout(ContextFanout):
        single_mapping_is_objects = False

    store = _Store()
    with StoreFanout(store):
        pass
    assert store.events == ["enter", "exit"]
    assert list(StoreFanout(store).values()) == [store]
    # the default class still unpacks it, as before
    assert list(ContextFanout(store)) == []

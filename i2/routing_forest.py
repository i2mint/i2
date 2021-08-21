##########################################################################################################
from itertools import chain
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Mapping, Tuple


class RoutingNode:
    """A RoutingNode instance needs to be callable on a single object, yielding an iterable or a final value"""

    def __call__(self, obj):
        raise NotImplementedError('You should implement this.')


@dataclass
class FinalNode(RoutingNode):
    """A RoutingNode that is final. It yields (both with call and iter) it's single `.val` attribute."""

    val: Any

    def __call__(self, obj=None):
        yield self.val

    def __iter__(self):
        yield self.val

    # def __getstate__(self):
    #     return {'val': self.val}


@dataclass
class CondNode(RoutingNode):
    """A RoutingNode that implements the if/then (no else) logic"""

    cond: Callable[[Any], bool]
    then: Any

    def __call__(self, obj):
        if self.cond(obj):
            yield from self.then(obj)

    def __iter__(self):
        yield from self.then


@dataclass
class RoutingForest(RoutingNode):
    """

    >>> rf = RoutingForest([
    ...     CondNode(cond=lambda x: isinstance(x, int),
    ...              then=RoutingForest([
    ...                  CondNode(cond=lambda x: int(x) >= 10, then=FinalNode('More than a digit')),
    ...                  CondNode(cond=lambda x: (int(x) % 2) == 1, then=FinalNode("That's odd!"))])
    ...             ),
    ...     CondNode(cond=lambda x: isinstance(x, (int, float)),
    ...              then=FinalNode('could be seen as a float')),
    ... ])
    >>> assert list(rf('nothing I can do with that')) == []
    >>> assert list(rf(8)) == ['could be seen as a float']
    >>> assert list(rf(9)) == ["That's odd!", 'could be seen as a float']
    >>> assert list(rf(10)) == ['More than a digit', 'could be seen as a float']
    >>> assert list(rf(11)) == ['More than a digit', "That's odd!", 'could be seen as a float']
    """

    cond_nodes: Iterable

    def __call__(self, obj):
        yield from chain(*(cond_node(obj) for cond_node in self.cond_nodes))
        # for cond_node in self.cond_nodes:
        #     yield from cond_node(obj)

    def __iter__(self):
        yield from chain(*self.cond_nodes)


FeatCondThens = Iterable[Tuple[Callable, Callable]]


@dataclass
class FeatCondNode(RoutingNode):
    """A RoutingNode that yields multiple routes, one for each of several conditions met,
    where the condition is computed implements computes a feature of the obj and according to a"""

    feat: Callable
    feat_cond_thens: FeatCondThens

    def __call__(self, obj):
        feature = self.feat(obj)
        for cond, then in self.feat_cond_thens:
            if cond(feature):
                yield from then(obj)

    def __iter__(self):
        yield from chain(*self.feat_cond_thens.values())


NoDefault = type('NoDefault', (object,), {})
NO_DFLT = NoDefault()


@dataclass
class SwitchCaseNode(RoutingNode):
    """A RoutingNode that implements the switch/case/else logic.
    It's just a specialization (enhanced with a "default" option) of the FeatCondNode class to a situation
    where the cond function of feat_cond_thens is equality, therefore the routing can be
    implemented with a {value_to_compare_to_feature: then_node} map.
    :param switch: A function returning the feature of an object we want to switch on
    :param cases: The mapping from feature to RoutingNode that should be yield for that feature.
        Often is a dict, but only requirement is that it implements the cases.get(val, default) method.
    :param default: Default RoutingNode to yield if no

    >>> rf = RoutingForest([
    ...     SwitchCaseNode(switch=lambda x: x % 5,
    ...                    cases={0: FinalNode('zero_mod_5'), 1: FinalNode('one_mod_5')},
    ...                    default=FinalNode('default_mod_5')),
    ...     SwitchCaseNode(switch=lambda x: x % 2,
    ...                    cases={0: FinalNode('even'), 1: FinalNode('odd')},
    ...                    default=FinalNode('that is not an int')),
    ... ])
    >>>
    >>> assert(list(rf(5)) == ['zero_mod_5', 'odd'])
    >>> assert(list(rf(6)) == ['one_mod_5', 'even'])
    >>> assert(list(rf(7)) == ['default_mod_5', 'odd'])
    >>> assert(list(rf(8)) == ['default_mod_5', 'even'])
    >>> assert(list(rf(10)) == ['zero_mod_5', 'even'])
    """

    switch: Callable
    cases: Mapping
    default: Any = NO_DFLT

    def __call__(self, obj):
        feature = self.switch(obj)
        if self.default is NO_DFLT:
            yield from self.cases.get(feature)(obj)
        else:
            yield from self.cases.get(feature, self.default)(obj)

    def __iter__(self):
        yield from chain(*self.cases.values())
        if self.default:
            yield self.default


def wrap_leafs_with_final_node(x):
    for xx in x:
        if isinstance(xx, RoutingNode):
            yield xx
        else:
            yield FinalNode(xx)


if __name__ == '__main__':

    print(
        '##########################################################################################################'
    )

    import inspect

    def could_be_int(obj):
        if isinstance(obj, int):
            b = True
        else:
            try:
                int(obj)
                b = True
            except ValueError:
                b = False
        if b:
            print(f'{inspect.currentframe().f_code.co_name}')
        return b

    def could_be_float(obj):
        if isinstance(obj, float):
            b = True
        else:
            try:
                float(obj)
                b = True
            except ValueError:
                b = False
        if b:
            print(f'{inspect.currentframe().f_code.co_name}')
        return b

    print(
        could_be_int(30),
        could_be_int(30.3),
        could_be_int('30.2'),
        could_be_int('nope'),
    )
    print(
        could_be_float(30),
        could_be_float(30.3),
        could_be_float('30.2'),
        could_be_float('nope'),
    )
    assert could_be_int('30.2') is False
    assert could_be_float('30.2') is True

    st = RoutingForest(
        [
            CondNode(
                cond=could_be_int,
                then=RoutingForest(
                    [
                        CondNode(
                            cond=lambda x: int(x) >= 10,
                            then=FinalNode('More than a digit'),
                        ),
                        CondNode(
                            cond=lambda x: (int(x) % 2) == 1,
                            then=FinalNode("That's odd!"),
                        ),
                    ]
                ),
            ),
            CondNode(cond=could_be_float, then=FinalNode('could be seen as a float')),
        ]
    )
    assert list(st('nothing I can do with that')) == []
    assert list(st(8)) == ['could be seen as a float']
    assert list(st(9)) == ["That's odd!", 'could be seen as a float']
    assert list(st(10)) == ['More than a digit', 'could be seen as a float']
    assert list(st(11)) == [
        'More than a digit',
        "That's odd!",
        'could be seen as a float',
    ]

    print(
        '### RoutingForest ########################################################################################'
    )
    rf = RoutingForest(
        [
            SwitchCaseNode(
                switch=lambda x: x % 5,
                cases={0: FinalNode('zero_mod_5'), 1: FinalNode('one_mod_5')},
                default=FinalNode('default_mod_5'),
            ),
            SwitchCaseNode(
                switch=lambda x: x % 2,
                cases={0: FinalNode('even'), 1: FinalNode('odd')},
                default=FinalNode('that is not an int'),
            ),
        ]
    )

    assert list(rf(5)) == ['zero_mod_5', 'odd']
    assert list(rf(6)) == ['one_mod_5', 'even']
    assert list(rf(7)) == ['default_mod_5', 'odd']
    assert list(rf(8)) == ['default_mod_5', 'even']
    assert list(rf(10)) == ['zero_mod_5', 'even']

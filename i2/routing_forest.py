"""
Tools to specify functions through trees and forests.

Whaaa?!?

Well, you see, often -- especially when writing transformers -- you have a series of
if/then conditions nested into eachother, in code, where it gets ugly and un-reusable.

This module explores ways to objectivy this: That is, to give us the means to create
such nested conditions in a way that we can define the parts as reusable operable
components.

Think of the relationship between the for loop (code) and the iterator (object), along
with iterator tools (itertools).
This is what we're trying to explore, but for if/then conditions.

I said explore. Some more work is needed here to make it easily usable.

Let's look at an example involving the three main actors of our play.
Each of these are ``Iterable`` and ``Callable`` (``Generator`` to be precise).

- ``CondNode``: implements the if/then (no else) logic
- ``FinalNode``: Final -- yields (both with call and iter) it's single `.val` attribute.
- ``RoutingForest``: An Iterable of ``CondNode``

>>> import inspect
>>>
>>> def could_be_int(obj):
...     if isinstance(obj, int):
...         b = True
...     else:
...         try:
...             int(obj)
...             b = True
...         except ValueError:
...             b = False
...     if b:
...         print(f'{inspect.currentframe().f_code.co_name}')
...     return b
...
>>> def could_be_float(obj):
...     if isinstance(obj, float):
...         b = True
...     else:
...         try:
...             float(obj)
...             b = True
...         except ValueError:
...             b = False
...     if b:
...         print(f'{inspect.currentframe().f_code.co_name}')
...     return b
...
>>> print(
...     could_be_int(30),
...     could_be_int(30.3),
...     could_be_int('30.2'),
...     could_be_int('nope'),
... )
could_be_int
could_be_int
True True False False
>>> print(
...     could_be_float(30),
...     could_be_float(30.3),
...     could_be_float('30.2'),
...     could_be_float('nope'),
... )
could_be_float
could_be_float
could_be_float
True True True False
>>> assert could_be_int('30.2') is False
>>> assert could_be_float('30.2') is True
could_be_float
>>>
>>> st = RoutingForest(
...     [
...         CondNode(
...             cond=could_be_int,
...             then=RoutingForest(
...                 [
...                     CondNode(
...                         cond=lambda x: int(x) >= 10,
...                         then=FinalNode('More than a digit'),
...                     ),
...                     CondNode(
...                         cond=lambda x: (int(x) % 2) == 1,
...                         then=FinalNode("That's odd!"),
...                     ),
...                 ]
...             ),
...         ),
...         CondNode(cond=could_be_float, then=FinalNode('could be seen as a float')),
...     ]
... )
>>> assert list(st('nothing I can do with that')) == []
>>> assert list(st(8)) == ['could be seen as a float']
could_be_int
could_be_float
>>> assert list(st(9)) == ["That's odd!", 'could be seen as a float']
could_be_int
could_be_float
>>> assert list(st(10)) == ['More than a digit', 'could be seen as a float']
could_be_int
could_be_float
>>> assert list(st(11)) == [
...     'More than a digit',
...     "That's odd!",
...     'could be seen as a float',
... ]
could_be_int
could_be_float
>>>
>>> print(
...     '### RoutingForest ########################################################################################'
... )
### RoutingForest ########################################################################################
>>> rf = RoutingForest(
...     [
...         SwitchCaseNode(
...             switch=lambda x: x % 5,
...             cases={0: FinalNode('zero_mod_5'), 1: FinalNode('one_mod_5')},
...             default=FinalNode('default_mod_5'),
...         ),
...         SwitchCaseNode(
...             switch=lambda x: x % 2,
...             cases={0: FinalNode('even'), 1: FinalNode('odd')},
...             default=FinalNode('that is not an int'),
...         ),
...     ]
... )
>>>
>>> assert list(rf(5)) == ['zero_mod_5', 'odd']
>>> assert list(rf(6)) == ['one_mod_5', 'even']
>>> assert list(rf(7)) == ['default_mod_5', 'odd']
>>> assert list(rf(8)) == ['default_mod_5', 'even']
>>> assert list(rf(10)) == ['zero_mod_5', 'even']
>>>

"""
from itertools import chain
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Mapping, Tuple
from i2.util import Literal


# TODO: Think a bit harder about this mini-language
# TODO: Right now use has to explicitly declare final nodes. Can do better.
def _default_mini_lang(x):
    # TODO: One is really tempted to use CondNode here to define this, right?
    if isinstance(x, (CondNode, RoutingForest, SwitchCaseNode)):
        return x
    elif isinstance(x, Literal):
        return x.get_val()
    elif isinstance(x, tuple):
        if len(x) == 2:
            return CondNode(*map(_default_mini_lang, x))
        elif len(x) == 3:
            return SwitchCaseNode(*map(_default_mini_lang, x))
        else:
            raise ValueError(
                f'If a tuple, element must be a `(cond, then)` pair, '
                f'or a (switch, case, default) triple, or a `Literal`. Was: {x}'
            )
    elif isinstance(x, dict):
        keys = set(x)
        if 2 <= len(keys) <= 3:
            if {'cond', 'then'}.issubset(keys):
                return CondNode(**{k: _default_mini_lang(v) for k, v in x.items()})
            elif {'switch', 'case'}.issubset(keys) and keys == {
                'switch',
                'case',
                'default',
            }:
                return SwitchCaseNode(
                    **{k: _default_mini_lang(v) for k, v in x.items()}
                )
            else:
                raise ValueError(
                    "keys should be 'switch', 'case' and optionally 'default'. "
                    f'Were: {keys}'
                )
        else:
            raise ValueError(
                f"A non-Literal dict must have keys 'cond' and 'then' "
                f"or 'switch' and 'case' (and optionally 'default'). Was: {x}"
            )
    elif isinstance(x, list):
        return RoutingForest(cond_nodes=list(map(_default_mini_lang, x)))
    else:
        return x


class RoutingNode:
    """A RoutingNode instance needs to be callable on a single object, yielding an iterable or a final value"""

    def __call__(self, obj):
        raise NotImplementedError('You should implement this.')

    @staticmethod
    def from_object(x, mini_lang=_default_mini_lang):
        return mini_lang(x)


@dataclass
class FinalNode(RoutingNode):
    """A RoutingNode that is final.
    It yields (both with call and iter) it's single `.val` attribute."""

    val: Any

    def __call__(self, obj=None):
        yield self.val

    def __iter__(self):
        yield self.val

    # def __getstate__(self):
    #     return {'val': self.val}


# TODO: Add tooling to merge validation into `then` functions/values
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

    def __post_init__(self):
        self.cond_nodes = list(map(self.from_object, self.cond_nodes))

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
    where the condition is computed implements computes a feature of the obj and
    according to an iterable of conditions on the feature"""

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


def test_routing_forest():
    def could_be_int(obj):
        if isinstance(obj, int):
            b = True
        else:
            try:
                int(obj)
                b = True
            except ValueError:
                b = False
        # if b:
        #     print(f'{inspect.currentframe().f_code.co_name}')
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
        # if b:
        #     print(f'{inspect.currentframe().f_code.co_name}')
        return b

    assert could_be_int(30)
    assert could_be_int(30.3)
    assert not could_be_int('30.2')
    assert not could_be_int('nope')

    assert could_be_float(30)
    assert could_be_float(30.3)
    assert could_be_float('30.2')
    assert not could_be_float('nope')

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

    # testing default mini-language #####################################################

    st2 = RoutingNode.from_object(
        [  # RoutingForest
            (  # CondNode
                could_be_int,
                [  # RoutingForest
                    (lambda x: int(x) >= 10, FinalNode('More than a digit')),
                    (lambda x: (int(x) % 2) == 1, FinalNode("That's odd!")),
                ],
            ),
            (could_be_float, FinalNode('could be seen as a float')),
        ]
    )

    assert list(st2('nothing I can do with that')) == []
    assert list(st2(8)) == ['could be seen as a float']
    assert list(st2(9)) == ["That's odd!", 'could be seen as a float']
    assert list(st2(10)) == ['More than a digit', 'could be seen as a float']
    assert list(st2(11)) == [
        'More than a digit',
        "That's odd!",
        'could be seen as a float',
    ]

    rf2 = RoutingForest(
        [
            (
                lambda x: x % 5,
                Literal({0: FinalNode('zero_mod_5'), 1: FinalNode('one_mod_5')}),
                FinalNode('default_mod_5'),
            ),
            (
                lambda x: x % 2,
                Literal({0: FinalNode('even'), 1: FinalNode('odd')}),
                FinalNode('that is not an int'),
            ),
        ]
    )

    assert list(rf2(5)) == ['zero_mod_5', 'odd']
    assert list(rf2(6)) == ['one_mod_5', 'even']
    assert list(rf2(7)) == ['default_mod_5', 'odd']
    assert list(rf2(8)) == ['default_mod_5', 'even']
    assert list(rf2(10)) == ['zero_mod_5', 'even']

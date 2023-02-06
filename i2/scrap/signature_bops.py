"""Signature binary operators

See design issue: https://github.com/i2mint/i2/issues/50
"""

from functools import partial
from collections import defaultdict
from itertools import tee
from typing import Callable, TypeVar, Optional, Union, Iterable

B = TypeVar('B')
B.__doc__ = (
    "'Base' operand that we are able (example, builtin object) to operate on, as is"
)
A = TypeVar('A')
A.__doc__ = "'Abstract' type we want to operate on through a key-function"
KeyFunction = Callable[[A], B]
KeyFunction.__doc__ = 'Function that transforms an A to a B'

R = TypeVar('R')
R.__doc__ = 'The return type of a binary operator'

BinaryOperator = Callable[[B, B], R]
AbstractBinaryOperator = Callable[[A, A], R]


def _keyed_comparator(
    binary_operator: BinaryOperator, key: KeyFunction, x: A, y: A,
) -> R:
    """Apply a binary operator to two operands,
    after transforming them through a key function"""
    return binary_operator(key(x), key(y))


def keyed_comparator(
    binary_operator: BinaryOperator, key: KeyFunction,
):
    """Create a key-function enabled binary operator"""
    return partial(_keyed_comparator, binary_operator, key)


# For back-compatibility:
_key_function_enabled_operator = _keyed_comparator
_key_function_factory = keyed_comparator


def _key_mappings(x: Iterable, key: Optional[KeyFunction] = None):
    """Create a mapping from key to value, and a mapping from value to key"""
    if key is None:
        key = lambda x: x

    key_for_item = dict
    item_for_key = defaultdict(list)

    for item in x:
        keyed_item = key(item)
        key_for_item[item] = keyed_item
        item_for_key[keyed_item].append(item)

    return key_for_item, item_for_key

    # def _key_item_
    #     for xi in x:
    #         yield xi


def simple_match(x: Iterable, y: Iterable, key=None):
    key_for_x, x_for_key = _key_mappings(x, key)
    key_for_y, y_for_key = _key_mappings(y, key)
    return x_for_key.keys() & y_for_key.keys()


def _sig_func(sig1, sig2, params_match, score_param_pair, score_aggreg):
    params = params_match(sig1, sig2)
    return score_aggreg(score_param_pair(params))


# Moved to i2.signatures (keep import below for back comp
from i2.signatures import param_binary_func

# from pydantic import validate_arguments, ValidationError

# from graphviz import Digraph
#
#
# def get_edge_list(graph: Digraph) -> list:
#     """Gets a list of edges (as node id pairs) from a digraph."""
#     return [
#         (node, child)
#         for node in graph.body
#         if node.startswith('  ')
#         for child in graph.body[graph.body.index(node) + 1 :]
#         if child.startswith('  ')
#     ]
# import networkx as nx
# nx.nx_agraph.read_dot


# B = TypeVar('B')
# B.__doc__ = (
#     "A 'base' type that we are able (example, builtin object) to operate on, as is"
# )
# A = TypeVar('A')
# A.__doc__ = "The 'abstract' type we want to operate on through a key-function"
# KeyFunction = Callable[[A], B]
# KeyFunction.__doc__ = 'Function that transforms an A to a B'
#
# AorB = TypeVar('AorB', A, B)
# # AorB = Union[A, B]
# AorB.__doc__ = 'An A (abstract) or a B (base)'
# AB = TypeVar('AB', bound=AorB)
# AorB.__doc__ = 'A generic AorB'
# R = TypeVar('R')
# R.__doc__ = 'The return type of a binary operator'
#
# BaseBinaryOperator = Callable[[B, B], R]
# AbstractBinaryOperator = Callable[[A, A], R]
# # BinaryOperator = Union[BaseBinaryOperator, AbstractBinaryOperator]  # equivalent?
# BinaryOperator = Callable[[AB, AB], R]
# KeyEnabledBinaryOperator = Callable[[AB, AB, KeyFunction], R]
#

# # key is None => x is B => y is B
# # key is not None => x is A => y is A
# def key_function_enabled_operator(
#     binary_operator: Union[BaseBinaryOperator, AbstractBinaryOperator],
#     x: AB,
#     y: AB,  # has to be the same as x. If x is A, so should y, if x is B so should y
#     key: Optional[KeyFunction] = None,
# ) -> R:
#     if key is None:
#         return binary_operator(x, y)
#     else:
#         return binary_operator(key(x), key(y))
#
#
# def key_function_factory(binary_operator: BinaryOperator) -> KeyEnabledBinaryOperator:
#     return partial(key_function_enabled_operator, binary_operator)
#

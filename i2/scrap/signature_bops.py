"""Signature binary operators

See design issue: https://github.com/i2mint/i2/issues/50
"""

from functools import partial

from typing import Callable, TypeVar, Optional, Union

B = TypeVar('B')
B.__doc__ = (
    "A 'base' type that we are able (example, builtin object) to operate on, as is"
)
A = TypeVar('A')
A.__doc__ = "The 'abstract' type we want to operate on through a key-function"
KeyFunction = Callable[[A], B]
KeyFunction.__doc__ = 'Function that transforms an A to a B'

AorB = TypeVar('AorB', A, B)
AorB.__doc__ = 'An A (abstract) or a B (base)'
AB = TypeVar('AB', bound=AorB)
AorB.__doc__ = 'A generic AorB'
R = TypeVar('R')
R.__doc__ = 'The return type of a binary operator'

BaseBinaryOperator = Callable[[B, B], R]
AbstractBinaryOperator = Callable[[A, A], R]
# BinaryOperator = Union[BaseBinaryOperator, AbstractBinaryOperator]  # equivalent?
BinaryOperator = Callable[[AB, AB], R]
KeyEnabledBinaryOperator = Callable[[AB, AB, KeyFunction], R]


def key_function_enabled_operator(
    binary_operator: Union[BaseBinaryOperator, AbstractBinaryOperator],
    x: AB,
    y: AB,  # has to be the same as x. If x is A, so should y, if x is B so should y
    key: Optional[KeyFunction] = None,
) -> R:
    if key is None:
        return binary_operator(x, y)
    else:
        return binary_operator(key(x), key(y))


def key_function_factory(binary_operator: BinaryOperator) -> KeyEnabledBinaryOperator:
    return partial(key_function_enabled_operator, binary_operator)

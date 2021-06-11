from pprint import pprint
import inspect


def func(a, b: int = 0, c=None) -> float:
    """

    Args:
        a: Something about a
        b: To b or not
        c: c this one?

    Returns: of the Jedis

    >>> func(3, c=2)
    1.5
    """
    c = c or (a + b)
    return (a + b) / c


# from glom import glom, Spec, Path, T
#
#
# spec = Spec(
#     {
#         'name': '__name__',
#         'doc': '__doc__',
#         'return_annotation': (inspect.signature, Path('return_annotation')),
#         'parameters': (inspect.signature, Path('parameters')),
#     }
# )
# pprint(glom(func, spec))

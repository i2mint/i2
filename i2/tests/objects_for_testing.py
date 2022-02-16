"""
A module containing a bunch of objects for testing.

See also: https://github.com/i2mint/py2http/blob/master/py2http/tests/objects_for_testing.py
"""


###############################################################################
# Empty callables, meant to give an example of different kinds of callables ###
# Note that x can be applied to a number, string, or list #####################
def f(x):
    """Note that x can be applied to a number, string, or list"""
    return x * 2


class C:
    @classmethod
    def c(cls, x):
        return f(x)

    @staticmethod
    def s(x):
        return f(x)

    # instance method
    def i(self, x):
        return f(x)

    # making an instance itself callable
    def __call__(self, x):
        return f(x)


###############################################################################


def times_2(x):
    return x * 2


def plus_1(z):
    return z + 1


def add(a, b: float = 0.0) -> float:
    """Adds numbers"""
    return a + b


def mult(x: float, y=1):
    return x * y


def formula1(w, /, x: float, y=1, *, z: int = 1):
    """Has 3 kinds & every of the 4 combinations of (default y/n, annotated y/n)"""
    return ((w + x) * y) ** z


def formula2(w, /, x: float, y=2, *, z: int = 3):
    """Has 3 kinds & every of the 4 combinations of (default y/n, annotated y/n)

    >>> formula2(0, 1, 2)
    '(w:=0) + (x:=1) * (y:=2) ** (z:=3) == 8'
    """
    return f'(w:={w}) + (x:={x}) * (y:={y}) ** (z:={z}) == {w + x * y ** z}'


# Only *args and **kwargs. Also has a doc with a doc test
def sum_of_args(*args, **kwargs):
    """Sums the args, and returns a copy of kwargs with the sum of args added to each value.

    >>> sum_of_args(1,2,3, a=0, b=10)
    {'a': 6, 'b': 16}
    """
    t = sum(args)
    return {k: t + v for k, v in kwargs.items()}


def test_func_1(
    any_var,
    a_list: list,
    a_dict: dict,
    an_int: int,
    a_float: float = 3.14,
    a_bool=True,
    a_str: str = None,
) -> str:
    """
    This is the first line,
    continued on the next.

    This function is meant to demo several combinations of type declaration (in signature or/and in docs),
    as well as with or without defaults.

    :param any_var: no type declared in either signature of doc
    :param list a_list: type declared in signature AND doc
    :param a_dict: type declared in signature but NOT in doc
    :param int an_int: type declared in doc but NOT in signature
    :param float a_float: type declared in signature AND doc, and has a default of the desired type
    :param a_bool: type declared in signature but NOT in doc, and has a default of the desired type
    :param str a_str: type declared in signature AND doc, but has a default that is not of that type
    line description
    :return str: Will just return the 'test_func_1 returned' string
    :tags foo, bar

    >>> # here are sometests
    >>> test_func_1('anything', [1, 2], {'this': 0, 'that': 'foo'}, 1, 1.1, a_bool=True, a_str='hello')
    'test_func_1 returned'
    """
    # This is the first comment
    return 'test_func_1 returned'  # this is a comment on the same line as some code


class SomeClass(object):
    pass


class SomeOtherClass(object):
    pass


def test_func_2(
    any_var,
    an_int: int,
    a_float=0.1,
    a_tuple: tuple = (),
    an_obj: SomeClass = SomeClass(),
    another_obj=SomeOtherClass(),
):
    """
    This is the first line,
    continued on the next.

    And this is some more information about the function:
    Blah
    Blah
    Bloo

    :param any_var: foo
    :param an_int: you again?!
    :param a_float:
    :param a_tuple: For real
    :param an_obj: Does nothing
    :param another_obj: Multi-
    line
    :return: Just pi
    :tags any, old, tag
    :keyword one_keyword
    :keyword keyword1, keyword2
    """

    """
    This is the first line,
    continued on the next.

    And this is some more information about the function:
    Blah
    Blah
    Bloo

    :param any_var: this can be of any type
    :param an_int: an int
    :param a_float: a float
    :param a_bool: a bool
    :param a_str: a str
    :param a_list:
    :param a_dict: An example of a continued
    line description
    :return:

    >>> # here are sometests
    >>> test_func_1('anything', 1, 1.1, True, 'hello', [1, 2], {'this': 0, 'that': 'foo'})
    'returned'
    """
    return 3.14


class AClass(object):
    def __init__(self, a=1, b=0):
        """The doc of the __init__"""
        self.a = a
        self.b = b

    def __call__(self, x: float) -> float:
        """The actual callable"""
        return self.a * x + self.b


class TestObj:
    constarg: str = ''

    def __init__(self, constarg: str):
        print(constarg)
        self.constarg = constarg

    def methodnum(self, methodarg1: int) -> int:
        print(methodarg1)
        return methodarg1 + 1

    def methodstr(self, methodarg2: str = 'hi') -> str:
        print(methodarg2)
        return methodarg2 + ' test ' + self.constarg


from contextlib import suppress

with suppress(ModuleNotFoundError, ImportError):
    from numpy import ndarray, matrix
    from pandas import DataFrame, Series

    def a_func_with_df_in_and_sr_out(df: DataFrame) -> Series:
        return df.sum()

    def a_func_with_np_and_df(
        df: DataFrame, sr: Series, arr: ndarray, mat: matrix
    ) -> matrix:
        return (df.sum().sum() + sr.sum() + arr.sum()) * mat

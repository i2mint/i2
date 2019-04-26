def test_func_1(any_var,
                a_list: list,
                a_dict: dict,
                an_int: int,
                a_float: float = 3.14,
                a_bool=True,
                a_str: str = None) -> str:
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


def test_func_2(any_var,
                an_int: int,
                a_float=0.1,
                a_tuple: tuple = (),
                an_obj: SomeClass = SomeClass(),
                another_obj=SomeOtherClass()):
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


class TestObj():
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


from numpy import array, ndarray, matrix
from pandas import DataFrame, Series


def a_func_with_df_in_and_sr_out(df: DataFrame) -> Series:
    return df.sum()


def a_func_with_np_and_df(df: DataFrame, sr: Series, arr: ndarray, mat: matrix) -> matrix:
    return (df.sum().sum() + sr.sum() + arr.sum()) * mat

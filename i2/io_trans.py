from dataclasses import dataclass
from typing import (
    Mapping,
    Callable,
    Optional,
    _TypedDictMeta,  # TODO: lint complains, but TypedDict doesn't do the trick (do _TypedDictMeta = TypedDict to see)
    TypedDict,
)
from inspect import signature, Parameter
from pickle import dumps

from i2.signatures import Sig

import functools

# TODO: Get rid of _TypedDictMeta dependency
# _TypedDictMeta = TypedDict  # to show that TypedDict doesn't work
# Raises     TypeError: TypedDict does not support instance and class checks

# monkey patching WRAPPER_ASSIGNMENTS to get "proper" wrapping (adding defaults and kwdefaults
wrapper_assignments = (
    '__module__',
    '__name__',
    '__qualname__',
    '__doc__',
    '__annotations__',
    '__defaults__',
    '__kwdefaults__',
)

update_wrapper = functools.update_wrapper
update_wrapper.__defaults__ = (
    functools.WRAPPER_ASSIGNMENTS,
    functools.WRAPPER_UPDATES,
)
wraps = functools.wraps
wraps.__defaults__ = (functools.WRAPPER_ASSIGNMENTS, functools.WRAPPER_UPDATES)


def identity_func(x):
    return x


@dataclass
class IoTrans:
    def in_val_trans(self, argval, argname, func):
        return argval  # default is the value as is

    def out_trans(self, argval, func):
        return argval  # default is the value as is

    def __call__(self, func):
        sig = Sig(func)

        @wraps(
            func
        )  # Todo: Want empty mapping as default (use frozendict or __post_init__?)
        def wrapped_func(*args, **kwargs):
            original_kwargs = sig.extract_kwargs(*args, **kwargs)
            new_kwargs = {
                argname: self.in_val_trans(argval, argname, func)
                for argname, argval in original_kwargs.items()
            }
            new_args, new_kwargs = sig.args_and_kwargs_from_kwargs(new_kwargs)
            return self.out_trans(func(*new_args, **new_kwargs), func)

        return wrapped_func


@dataclass
class ArgnameIoTrans(IoTrans):
    """Transforms argument values according to their names

    >>> def foo(a, b=1.0):
    ...     return a + b
    >>>
    >>> input_trans = ArgnameIoTrans({'a': int, 'b': float})
    >>> foo2 = input_trans(foo)
    >>> assert foo2(3) == 4.0
    >>> assert foo2(-2, 2) == 0.0
    >>> assert foo2("3") == 4.0
    >>> assert foo2("-2", "2") == 0.0
    >>> assert signature(foo) == signature(foo2)
    """

    argname_2_trans_func: Mapping

    def in_val_trans(self, argval, argname, func):
        trans_func = self.argname_2_trans_func.get(argname, None)
        if trans_func is not None:
            return trans_func(argval)
        else:
            return super().in_val_trans(argval, argname, func)


empty = Parameter.empty


@dataclass
class AnnotAndDfltIoTrans(IoTrans):
    """Transforms argument values using annotations and default type

    >>> def foo(a: int, b=1.0):
    ...     return a + b
    >>>
    >>> input_trans = AnnotAndDfltIoTrans()
    >>> foo3 = input_trans(foo)
    >>> assert foo3(3) == 4.0
    >>> assert foo3(-2, 2) == 0.0
    >>> assert foo3("3") == 4.0
    >>> assert foo3("-2", "2") == 0.0
    >>> assert signature(foo) == signature(foo3)
    """

    annotations_handled = frozenset([int, float, str])
    dflt_types_handled = frozenset([int, float, str])

    def in_val_trans(self, argval, argname, func):
        param = signature(func).parameters[argname]
        if param.annotation in self.annotations_handled:
            return param.annotation(argval)
        else:
            dflt_type = type(param.default)
            if dflt_type in self.dflt_types_handled:
                return dflt_type(argval)
        return super().in_val_trans(argval, argname, func)


@dataclass
class JSONAnnotAndDfltIoTrans(AnnotAndDfltIoTrans):
    """Transforms argument values using annotations and default type,
    including lists, iterables, dicts, and booleans

    >>> def foo(a: dict, b=['dflt'], c=False):
    ...     return dict({}, a=a, b=b, c=c)
    >>>
    >>> input_trans = JSONAnnotAndDfltIoTrans()
    >>> foo4 = input_trans(foo)
    >>> assert foo4('{}') == {'a': {}, 'b': ['dflt'], 'c': False}
    >>> assert foo4({'d': 'e'}, '["some_value"]', 'true') == {'a': {'d': 'e'}, 'b': ['some_value'], 'c': True}
    >>> complex_types_result = foo4('{"None": null, "True": true, "False": false}', '[null, true, false]', 'false')
    >>> assert complex_types_result == {'a': {'None': None, 'True': True, 'False': False}, 'b': [None, True, False], 'c': False}
    >>> assert signature(foo) == signature(foo4)
    """

    def in_val_trans(self, argval, argname, func):
        param = signature(func).parameters[argname]
        if param.annotation != str and not isinstance(param.default, str):
            if (
                param.annotation == dict
                or isinstance(param.annotation, _TypedDictMeta)
                or isinstance(param.default, dict)
                or isinstance(type(param.default), _TypedDictMeta)
                or param.annotation == bool
                or isinstance(param.default, bool)
            ):
                return cast_to_jdict(argval)
            if hasattr(param.annotation, '__iter__') or hasattr(
                param.default, '__iter__'
            ):
                return cast_to_list(argval)
        return super().in_val_trans(argval, argname, func)


@dataclass
class TypedBasedOutIoTrans(IoTrans):
    """Transform output according to it's type.

    # TODO: Move this doctest to tests suite that ignores the test if pandas not there!

    # >>> import pandas as pd
    # >>> out_trans = TypedBasedOutIoTrans({
    # ...     (list, tuple, set): ', '.join,
    # ...     pd.DataFrame: pd.DataFrame.to_csv
    # ... })
    # >>>
    # >>>
    # >>> @out_trans
    # ... def repeat(a: int, b: list):
    # ...     return a * b
    # ...
    # >>> assert repeat(2, ['repeat', 'it']) == 'repeat, it, repeat, it'
    # >>>
    # >>> @out_trans
    # ... def transpose(df):
    # ...     return df.T
    # ...
    # >>> df = pd.DataFrame({'a': [1,2,3], 'b': [10, 20, 30]})
    # >>> print(df.to_csv())  # doctest: +NORMALIZE_WHITESPACE
    # ,a,b
    # 0,1,10
    # 1,2,20
    # 2,3,30
    # >>> print(transpose(df))  # doctest: +NORMALIZE_WHITESPACE
    # ,0,1,2
    # a,1,2,3
    # b,10,20,30

    """

    trans_func_for_type: Mapping = ()  # Todo: Want empty mapping as default (use frozendict or __post_init__?)
    dflt_trans_func: Optional[Callable] = None

    def out_trans(self, argval, func):
        for typ in self.trans_func_for_type:
            if isinstance(argval, typ):
                return self.trans_func_for_type[typ](argval)
        if isinstance(
            self.dflt_trans_func, Callable
        ):  # Question: use callable() instead? What's the difference?
            return self.dflt_trans_func(argval)


def pickle_out_trans(self, argval, func):
    return dumps(argval)


PickleFallbackTypedBasedOutIoTrans = functools.partial(
    TypedBasedOutIoTrans, dflt_trans_func=dumps
)

import json
import os


def cast_to_jdict(value):
    """Tries to cast to a json-friendly dictionary.

    >>> cast_to_jdict('3')
    [3]
    >>> cast_to_jdict("[3]")
    [3]
    >>> cast_to_jdict("[4,2]")
    [4, 2]
    >>> cast_to_jdict('[4, "string", ["another", "list"], {"nested": 10.2}]')
    [4, 'string', ['another', 'list'], {'nested': 10.2}]
    >>> cast_to_jdict('{"here": "is", "a": {"nested": "json"}, "with": [null, true, false, 1, 2.3]}')
    {'here': 'is', 'a': {'nested': 'json'}, 'with': [None, True, False, 1, 2.3]}

    And csvs too:

    >>> cast_to_jdict('1,2,3.4, "string" ,  null, true, false, ["a", "list"]')
    [1, 2, 3.4, 'string', None, True, False, ['a', 'list']]
    """
    if isinstance(value, str):
        value = value.strip()
        if value:
            first_char = value[0]
            if first_char in {'[', '{'}:
                return json.loads(value)
            elif value in ['true', 'false', 'null']:
                return json.loads(value)
            elif os.path.isfile(value):
                return json.load(value)
            else:
                return json.loads(
                    '[' + value + ']'
                )  # wrap in brackets and call json.loads
        else:
            return ''
    else:
        return value


def cast_to_list(value):
    """Tries to case to a list (with json friendly elements)

    >>> cast_to_list('3')
    [3]
    >>> cast_to_list("[3]")
    [3]
    >>> cast_to_list("[4,2]")
    [4, 2]
    >>> cast_to_list('[4, "string", ["another", "list"], {"nested": 10.2}]')
    [4, 'string', ['another', 'list'], {'nested': 10.2}]

    And csvs too:

    >>> cast_to_list('1,2,3.4, "string" ,  null, true, false, ["a", "list"]')
    [1, 2, 3.4, 'string', None, True, False, ['a', 'list']]
    """
    if isinstance(value, str):
        value = cast_to_jdict(value)
        assert isinstance(value, list)
        return value
    elif hasattr(value, 'tolist'):  # meant for numpy arrays
        # what other potential attributes to check for?
        return value.tolist()
    else:
        return list(
            value
        )  # will work with set, tuple, and other iterables (not recursively though: just level 0)


# @dataclass
# class PickleFallbackTypedBasedOutIoTrans(TypedBasedOutIoTrans):
#     dflt_trans_func = pickle_out_trans

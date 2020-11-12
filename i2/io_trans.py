from dataclasses import dataclass
from typing import Mapping, Callable, Optional, Union, Iterable
from inspect import signature, Parameter
from pickle import dumps

from i2.signatures import Sig

import functools

# monkey patching WRAPPER_ASSIGNMENTS to get "proper" wrapping (adding defaults and kwdefaults
# TODO: Verify this actually works.
functools.WRAPPER_ASSIGNMENTS = (
    '__module__', '__name__', '__qualname__', '__doc__',
    '__annotations__', '__defaults__', '__kwdefaults__')

wrapper_assignments = functools.WRAPPER_ASSIGNMENTS
update_wrapper = functools.update_wrapper
update_wrapper.__defaults__ = (functools.WRAPPER_ASSIGNMENTS, functools.WRAPPER_UPDATES)
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

        @wraps(func)  # Todo: Want empty mapping as default (use frozendict or __post_init__?)
        def wrapped_func(*args, **kwargs):
            original_kwargs = sig.extract_kwargs(*args, **kwargs)
            new_kwargs = {argname: self.in_val_trans(argval, argname, func) for argname, argval in
                          original_kwargs.items()}
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
class TypedBasedOutIoTrans(IoTrans):
    """Transform output according to it's type.

    >>> import pandas as pd
    >>> out_trans = TypedBasedOutIoTrans({
    ...     (list, tuple, set): ', '.join,
    ...     pd.DataFrame: pd.DataFrame.to_csv
    ... })
    >>>
    >>>
    >>> @out_trans
    ... def repeat(a: int, b: list):
    ...     return a * b
    ...
    >>> assert repeat(2, ['repeat', 'it']) == 'repeat, it, repeat, it'
    >>>
    >>> @out_trans
    ... def transpose(df):
    ...     return df.T
    ...
    >>> df = pd.DataFrame({'a': [1,2,3], 'b': [10, 20, 30]})
    >>> print(df.to_csv())
    ,a,b
    0,1,10
    1,2,20
    2,3,30

    >>> print(transpose(df))
    ,0,1,2
    a,1,2,3
    b,10,20,30
    """
    trans_func_for_type: Mapping = ()  # Todo: Want empty mapping as default (use frozendict or __post_init__?)
    dflt_trans_func: Optional[Callable] = None

    def out_trans(self, argval, func):
        for typ in self.trans_func_for_type:
            if isinstance(argval, typ):
                return self.trans_func_for_type[typ](argval)
        if isinstance(self.dflt_trans_func, Callable):  # Question: use callable() instead? What's the difference?
            return self.dflt_trans_func(argval)


def pickle_out_trans(self, argval, func):
    return dumps(argval)


PickleFallbackTypedBasedOutIoTrans = functools.partial(TypedBasedOutIoTrans, dflt_trans_func=dumps)

# @dataclass
# class PickleFallbackTypedBasedOutIoTrans(TypedBasedOutIoTrans):
#     dflt_trans_func = pickle_out_trans

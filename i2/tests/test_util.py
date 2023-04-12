"""Test utils"""

import pickle
from i2.signatures import Sig
from i2.wrapper import wrap
from i2.util import copy_func


def test_copy_func_with_wrap(copy_func=copy_func):
    f = lambda x, *, y=2: x * y
    wrapped_f = wrap(f)  # empty wrap
    assert Sig(copy_func(wrapped_f)) == Sig(f)
    assert wrapped_f(3) == f(3) == 6


def is_pickable(obj):
    try:
        _ = pickle.loads(pickle.dumps(obj))
        return True
    except Exception:
        return False


def unpickled_func_still_works(func, *args, **kwargs):
    expected = func(*args, **kwargs)
    unpickled_func = pickle.loads(pickle.dumps(func))
    result = unpickled_func(*args, **kwargs)
    return expected == result

"""Test utils"""

import pickle


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

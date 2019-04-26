from functools import reduce
import operator


def iter_key_path_items(d, key_path_prefix=None):
    """
    iterate through items of dict recursively, yielding (key_path, val) pairs for all nested values that are not dicts.
    That is, if a value is a dict, it won't generate a yield, but rather, will be iterated through recursively.
    :param d: input dict
    :param key_path_so_far: string to be prepended to all key paths (for use in recursion, not meant for direct use)
    :return: a (key_path, val) iterator
    >>> input_dict = {
    ...     'a': {
    ...         'a': 'a.a',
    ...         'b': 'a.b',
    ...         'c': {
    ...             'a': 'a.c.a'
    ...         }
    ...     },
    ...     'b': 'b',
    ...     'c': 3
    ... }
    >>> list(iter_key_path_items(input_dict))
    [('a.a', 'a.a'), ('a.c.a', 'a.c.a'), ('a.b', 'a.b'), ('c', 3), ('b', 'b')]
    """
    if key_path_prefix is None:
        for k, v in d.items():
            if not isinstance(v, dict):
                yield k, v
            else:
                for kk, vv in iter_key_path_items(v, k):
                    yield kk, vv
    else:
        for k, v in d.items():
            if not isinstance(v, dict):
                yield key_path_prefix + '.' + k, v
            else:
                for kk, vv in iter_key_path_items(v, k):
                    yield key_path_prefix + '.' + kk, vv


def extract_key_paths(d, key_paths, field_naming='full', use_default=False, default_val=None):
    """
    getting with a key list or "."-separated string
    :param d: dict
    :param key_path: list or "."-separated string of keys
    :param field_naming: 'full' (default) will use key_path strings as is, leaf will only use the last dot item
        (i.e. this.is.a.key.path will result in "path" being used)
    :return:
    >>> d = {
    ...     'a': {
    ...         'a': 'a.a',
    ...         'b': 'a.b',
    ...         'c': {
    ...             'a': 'a.c.a'
    ...         }
    ...     },
    ...     'b': 'b',
    ...     'c': 3
    ... }
    >>> extract_key_paths(d, 'a')
    {'a': {'a': 'a.a', 'b': 'a.b', 'c': {'a': 'a.c.a'}}}
    >>> extract_key_paths(d, 'a.a')
    {'a.a': 'a.a'}
    >>> extract_key_paths(d, 'a.c')
    {'a.c': {'a': 'a.c.a'}}
    >>> extract_key_paths(d, ['a.a', 'a.c'])
    {'a.a': 'a.a', 'a.c': {'a': 'a.c.a'}}
    >>> extract_key_paths(d, ['a.a', 'something.thats.not.there'])  # missing key just won't be included
    {'a.a': 'a.a'}
    >>> extract_key_paths(d, ['a.a', 'something.thats.not.there'], use_default=True, default_val=42)
    {'a.a': 'a.a', 'something.thats.not.there': 42}
    """
    dd = dict()
    if isinstance(key_paths, str):
        key_paths = [key_paths]
    if isinstance(key_paths, dict):
        key_paths = [k for k, v in key_paths.items() if v]

    for key_path in key_paths:

        if isinstance(key_path, str):
            field = key_path
            key_path = key_path.split('.')
        else:
            field = '.'.join(key_path)

        if field_naming == 'leaf':
            field = key_path[-1]
        else:
            field = field

        try:
            dd.update({field: reduce(operator.getitem, key_path, d)})
        except (TypeError, KeyError):
            if use_default:
                dd.update({field: default_val})

    return dd


def key_paths(d):
    key_path_list = list()
    for k, v in d.items():
        if not isinstance(v, dict):
            key_path_list.append(k)
        else:
            key_path_list.extend([k + '.' + x for x in key_paths(v)])
    return key_path_list


def get_value_in_key_path(d, key_path, default_val=None):
    """
    getting with a key list or "."-separated string
    :param d: dict
    :param key_path: list or "."-separated string of keys
    :return:
    """
    if isinstance(key_path, str):
        key_path = key_path.split('.')
    try:
        return reduce(operator.getitem, key_path, d)
    except (TypeError, KeyError):
        return default_val


def set_value_in_key_path(d, key_path, val):
    """
    setting with a key list or "."-separated string
    :param d: dict
    :param key_path: list or "."-separated string of keys
    :param val: value to assign
    :return:
    """
    if isinstance(key_path, str):
        key_path = key_path.split('.')
    get_value_in_key_path(d, key_path[:-1])[key_path[-1]] = val


def set_value_in_nested_key_path(d, key_path, val):
    """

    :param d:
    :param key_path:
    :param val:
    :return:
    >>> input_dict = {
    ...   "a": {
    ...     "c": "val of a.c",
    ...     "b": 1,
    ...   },
    ...   "10": 10,
    ...   "b": {
    ...     "B": {
    ...       "AA": 3
    ...     }
    ...   }
    ... }
    >>>
    >>> set_value_in_nested_key_path(input_dict, 'new.key.path', 7)
    >>> input_dict
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': {'path': 7}}}
    >>> set_value_in_nested_key_path(input_dict, 'new.key.old.path', 8)
    >>> input_dict
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': {'path': 7, 'old': {'path': 8}}}}
    >>> set_value_in_nested_key_path(input_dict, 'new.key', 'new val')
    >>> input_dict
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': 'new val'}}
    """
    if isinstance(key_path, str):
        key_path = key_path.split('.')
    first_key = key_path[0]
    if len(key_path) == 1:
        d[first_key] = val
    else:
        if first_key in d:
            set_value_in_nested_key_path(d[first_key], key_path[1:], val)
        else:
            d[first_key] = {}
            set_value_in_nested_key_path(d[first_key], key_path[1:], val)

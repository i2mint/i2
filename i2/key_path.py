"""Flattening maps and manipulating key paths"""

from functools import reduce, wraps, partial
import operator
from collections.abc import MutableMapping


def trans_generator_output(trans):
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            yield from (trans(x) for x in func(*args, **kwargs))

        return wrapped

    return decorator


class NoDefault:
    pass


NO_DFLT = NoDefault()


def flatten_dict(d, sep=None, prefix=''):
    """
    Computes a "flat" dict from a nested one. A flat dict's keys are the paths of the input dict.
    These paths will be expressed as tuples of the original keys by defaults.
    If these keys are strings though, you can use sep and prefix to get string representations of the paths.

    :param d: a nested dict
    :param sep: The separator character (or string) in a string representation of the paths.
    :param prefix: A string to prepend on all the paths
    :return: A flat dict

    >>> d = {'a': {
    ...         'a': '2a',
    ...         'c': {'a': 'aca', 'u': 4}
    ...         },
    ...      'c': 3
    ...     }
    >>> flatten_dict(d)
    {('a', 'a'): '2a', ('a', 'c', 'a'): 'aca', ('a', 'c', 'u'): 4, ('c',): 3}
    >>> flatten_dict(d, sep='.')
    {'a.a': '2a', 'a.c.a': 'aca', 'a.c.u': 4, 'c': 3}
    >>> flatten_dict(d, sep='/', prefix='/ROOT/')
    {'/ROOT/a/a': '2a', '/ROOT/a/c/a': 'aca', '/ROOT/a/c/u': 4, '/ROOT/c': 3}
    """
    if sep is None:
        kp = KeyPathMap(d)
    else:
        kp = StrKeyPath(d, sep=sep, prefix=prefix)
    return kp.flat_dict()


def rollout_dict(d, sep=None, prefix=''):
    """
    Get the nested path of a flat (key path) dict. This is the inverse of flatten_dict.

    :param d: a flat dict (i.e. one whose keys are paths of a nested dict)
    :param sep: If None (default), the paths should be key tuples. If a string, it it assumed to be
        the separator of string representations of the path
    :param prefix: A string that has be prepended to all each key (path) of the input dict
        (and therefore should be removed)
    :return: The corresponding nested path

    >>> flat_d = {('a', 'a'): '2a', ('a', 'c', 'a'): 'aca', ('a', 'c', 'u'): 4, ('c',): 3}
    >>> rollout_dict(flat_d)
    {'a': {'a': '2a', 'c': {'a': 'aca', 'u': 4}}, 'c': 3}
    >>> flat_d = {'a.a': '2a', 'a.c.a': 'aca', 'a.c.u': 4, 'c': 3}
    >>> rollout_dict(flat_d, sep='.')
    {'a': {'a': '2a', 'c': {'a': 'aca', 'u': 4}}, 'c': 3}
    >>> flat_d = {'/ROOT/a/a': '2a', '/ROOT/a/c/a': 'aca', '/ROOT/a/c/u': 4, '/ROOT/c': 3}
    >>> rollout_dict(flat_d, sep='/', prefix='/ROOT/')
    {'a': {'a': '2a', 'c': {'a': 'aca', 'u': 4}}, 'c': 3}
    """
    if sep is None:
        kp = KeyPathMap()
    else:
        kp = StrKeyPath(sep=sep, prefix=prefix)
    return kp.rollout(d)


# Note: Might need to extend control of setitem: overwrites, leaf->node, node->leaf
# Note: Could also control deletes more.
class KeyPathMap(MutableMapping):
    """
    Provides a key-path view to a nested mapping (by default, a dict).
    A nested mapping can be see as a tree, where if a value is itself a mapping, it is a non-terminal node,
    leaves (or terminal) holding the "actual values".

    When wrapping a mapping in KeyPathMap, you can pretend that you have a flat mapping from (root to leaf) paths
    instead of a nested structure, and do your mapping CRUD with that view.

    >>> d = {'a': {
    ...         'a': '2a',
    ...         'b': {'a': 'aba',
    ...               'b': 3}
    ...         },
    ...      'c': 3.14
    ...     }
    >>> kp = KeyPathMap(d)
    >>> list(kp.items())
    [(('a', 'a'), '2a'), (('a', 'b', 'a'), 'aba'), (('a', 'b', 'b'), 3), (('c',), 3.14)]
    >>> list(kp)
    [('a', 'a'), ('a', 'b', 'a'), ('a', 'b', 'b'), ('c',)]
    >>> len(kp)
    4
    >>> assert list(kp) == list(kp.keys())
    >>> list(kp.values())
    ['2a', 'aba', 3, 3.14]
    >>> kp['a']
    {'a': '2a', 'b': {'a': 'aba', 'b': 3}}
    >>> kp[('a',)]
    {'a': '2a', 'b': {'a': 'aba', 'b': 3}}
    >>> kp['a', 'a']
    '2a'
    >>> kp['a', 'b', 'b']
    3
    >>> ('a', 'new_key') in kp
    False
    >>> kp['a', 'new_key'] = 'new val'
    >>> ('a', 'new_key') in kp
    True
    >>> kp['a', 'new_key']
    'new val'
    >>> len(kp)
    5
    >>> del kp['a', 'b', 'a']
    >>> len(kp)
    4
    >>> list(kp.items())
    [(('a', 'a'), '2a'), (('a', 'b', 'b'), 3), (('a', 'new_key'), 'new val'), (('c',), 3.14)]
    >>>
    >>> # By default, you can only write on already created nodes. But if auto_node_writes=True, you can do this:
    >>> kp = KeyPathMap(auto_node_writes=True)
    >>> kp
    {}
    >>> kp['a', 'b', 'c'] = 'hi world!'
    >>> kp
    {'a': {'b': {'c': 'hi world!'}}}
    """

    def __init__(
        self,
        store=dict,
        key_type: type = None,
        node_type: type = None,
        auto_node_writes=False,
    ):
        """
        Initialize a KeyPathMap.
        :param store: Your mapping, or the type of your mapping.
        :param key_type: The type of the keys
        :param node_type: The node type (typically the same as the store type
        :param auto_node_writes: False by default, which means you can only write on already created nodes.
            But if True, you can assign to paths with prefixes that don't yet exist. See examples.
        """
        if isinstance(store, type):
            store = store()
        store_type = type(store)
        if node_type is None:
            node_type = store_type
        if key_type is None:
            try:  # consider the type of the first key found in the store as the type for all
                key_type = type(next(iter(store.keys())))
            except StopIteration:
                key_type = str  # default to string keys

        def is_node(x):
            return isinstance(x, node_type)

        def is_key(x):
            return isinstance(x, key_type)

        self.store = store
        self.is_node = is_node
        self.is_key = is_key
        self.mk_new_node = node_type
        self.auto_node_writes = auto_node_writes

    def __getitem__(self, k):
        if not self.is_key(k):
            d = self.store.__getitem__(k[0])
            for kk in k[1:]:
                d = d[kk]
            return d
        else:
            return self.store.__getitem__(k)

    def __setitem__(self, k, v):
        if self.auto_node_writes:
            self._recursive_setitem(k, v)
        else:
            if not self.is_key(k) and len(k) > 1:
                d = self.store.__getitem__(k[0])
                for kk in k[1:-1]:
                    d = d[kk]
                d[k[-1]] = v
            else:
                self.store.__setitem__(k, v)

    def __delitem__(self, k):
        if not self.is_key(k):
            if len(k) == 1:
                self.store.__delitem__(k[0])
            else:
                d = self.store.__getitem__(k[0])
                for kk in k[1:-1]:
                    d = d[kk]
                d.__delitem__(k[-1])
        else:
            self.store.__delitem__(k)

    def items(self):
        return self._items(self.store)

    def __iter__(self):
        return map(lambda x: x[0], self.items())

    def __contains__(self, k):
        if self.is_key(k):
            return self.store.__contains__(k)
        else:
            d = self.store
            for kk in k:
                if not d.__contains__(kk):
                    return False
                else:
                    d = d[kk]
            return True  # if you got so far, you have all the keys in the path

    def __len__(self):
        c = 0
        for _ in self.items():
            c += 1
        return c

    def _recursive_setitem(self, k, v):
        if not self.is_key(k):
            d = self.store
            for kk in k[:-1]:
                if kk not in d:
                    d[kk] = self.mk_new_node()
                d = d[kk]
                # else:
                #     d = d[kk]
                #     assert self.is_node(d), f"Trying to set a value for path {k}, but {kk} was a leaf"
            d[k[-1]] = v
        else:
            self.store.__setitem__(k, v)

    @trans_generator_output(lambda x: (tuple(x[0]), x[1]))
    def _items(self, d, key_path_prefix=None):
        if key_path_prefix is None:
            for k, v in d.items():
                if not self.is_node(v):
                    yield [k], v
                else:
                    for kk, vv in self._items(v, [k]):
                        yield kk, vv
        else:
            for k, v in d.items():
                if not self.is_node(v):
                    yield key_path_prefix + [k], v
                else:
                    if self.is_key(k):
                        k = [k]
                    for kk, vv in self._items(v, k):
                        yield key_path_prefix + list(kk), vv

    def __repr__(self):
        return self.store.__repr__()

    def flat_dict(self):
        return {k: v for k, v in self.items()}

    def rollout(self, d):
        target = self.__class__(auto_node_writes=True)
        for k, v in d.items():
            target[k] = v
        return target.store

    def extract(self, key_paths=None, default_val=NO_DFLT, target=dict):
        if key_paths is None:
            key_paths = self.keys()
        if isinstance(target, type):
            target = target()
        for kp in key_paths:
            try:
                target[kp] = self[kp]
            except KeyError:
                if default_val is not NO_DFLT:
                    target[kp] = default_val
                else:
                    raise
        return target


class StrKeyPath(KeyPathMap):
    """
    A KeyPathMap, but where the key paths are expressed as string with a separator.
    If sep = '.', then instead of using ('a', 'b', 'c') as a key, you can use 'a.b.c'.

    >>> d = {'a': {
    ...         'a': '2a',
    ...         'b': {'a': 'aba',
    ...               'b': 3}
    ...         },
    ...      'c': 3.14
    ...     }
    >>> # Example with sep='/'
    >>> kp = StrKeyPath(d, sep='/')
    >>> list(kp.items())
    [('a/a', '2a'), ('a/b/a', 'aba'), ('a/b/b', 3), ('c', 3.14)]
    >>> # You can also add a prefix to the keys
    >>> kp = StrKeyPath(d, sep='/', prefix="http://")
    >>> list(kp.items())
    [('http://a/a', '2a'), ('http://a/b/a', 'aba'), ('http://a/b/b', 3), ('http://c', 3.14)]
    >>>
    >>> # Default sep is '.', so we'll work with that:
    >>> kp = StrKeyPath(d)
    >>> kp
    {'a': {'a': '2a', 'b': {'a': 'aba', 'b': 3}}, 'c': 3.14}
    >>> list(kp.items())
    [('a.a', '2a'), ('a.b.a', 'aba'), ('a.b.b', 3), ('c', 3.14)]
    >>> list(kp)
    ['a.a', 'a.b.a', 'a.b.b', 'c']
    >>> len(kp)
    4
    >>> assert list(kp) == list(kp.keys())
    >>> list(kp.values())
    ['2a', 'aba', 3, 3.14]
    >>> kp['a']
    {'a': '2a', 'b': {'a': 'aba', 'b': 3}}
    >>> kp['a.a']
    '2a'
    >>> kp['a.b.b']
    3
    >>> ('a.new_key') in kp
    False
    >>> kp['a.new_key'] = 'new val'
    >>> 'a.new_key' in kp
    True
    >>> kp['a.new_key']
    'new val'
    >>> len(kp)
    5
    >>> del kp['a.b.a']
    >>> len(kp)
    4
    >>> list(kp.items())
    [('a.a', '2a'), ('a.b.b', 3), ('a.new_key', 'new val'), ('c', 3.14)]
    >>>
    >>> # By default, you can only write on already created nodes. But if auto_node_writes=True, you can do this:
    >>> kp = StrKeyPath(auto_node_writes=True)
    >>> kp
    {}
    >>> kp['a.b.c'] = 'hi world!'
    >>> kp
    {'a': {'b': {'c': 'hi world!'}}}
    >>>
    """

    def __init__(
        self,
        store=dict,
        key_type: type = None,
        node_type: type = None,
        auto_node_writes=False,
        sep: str = '.',
        prefix: str = '',
    ):
        prefix_length = len(prefix)
        self.sep = sep
        self.prefix = prefix
        self._id_of_key = lambda k: tuple(k[prefix_length:].split(sep))
        self._key_of_id = lambda _id: prefix + sep.join(_id)
        super().__init__(
            store=store,
            key_type=key_type,
            node_type=node_type,
            auto_node_writes=auto_node_writes,
        )

    def __getitem__(self, k):
        return super().__getitem__(self._id_of_key(k))

    def __setitem__(self, k, v):
        return super().__setitem__(self._id_of_key(k), v)

    def __delitem__(self, k):
        return super().__delitem__(self._id_of_key(k))

    def __contains__(self, k):
        return super().__contains__(self._id_of_key(k))

    def items(self):
        yield from ((self._key_of_id(_id), v) for _id, v in super().items())

    def rollout(self, d):
        target = self.__class__(sep=self.sep, prefix=self.prefix, auto_node_writes=True)
        for k, v in d.items():
            target[k] = v
        return target.store


# TODO: Not what is below should remain. Probably better to just use the above.
class KeyPathTrans:
    """
    Doing what StrKeyPath but where the store that is being operated on is not included in the object, but
    given to the method as input.
    """

    def __init__(self, sep: str = '.', node_type: type = dict, mk_new_node=None):
        """

        :param sep:
        :param node_type:
        """
        self.sep = sep
        self.node_type = node_type
        if mk_new_node is None:
            mk_new_node = node_type
        self.mk_new_node = mk_new_node

    def items(self, d, key_path_prefix=None):
        """
        iterate through items of store recursively, yielding (key_path, val) pairs for all nested values that are not
        store types.
        That is, if a value is a store_type, it won't generate a yield, but rather, will be iterated through
        recursively.
        :param d: input store
        :param key_path_so_far: string to be prepended to all key paths (for use in recursion, not meant for direct use)
        :return: a (key_path, val) iterator

        >>> kp = KeyPathTrans()
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
        >>> list(kp.items(input_dict))
        [('a.a', 'a.a'), ('a.b', 'a.b'), ('a.c.a', 'a.c.a'), ('b', 'b'), ('c', 3)]
        """
        if key_path_prefix is None:
            for k, v in d.items():
                if not isinstance(v, self.node_type):
                    yield k, v
                else:
                    for kk, vv in self.items(v, k):
                        yield kk, vv
        else:
            for k, v in d.items():
                if not isinstance(v, self.node_type):
                    yield key_path_prefix + self.sep + k, v
                else:
                    for kk, vv in self.items(v, k):
                        yield key_path_prefix + self.sep + kk, vv

    def keys(self, d):
        for k, v in d.items():
            if not isinstance(v, self.node_type):
                yield k
                # key_path_list.append(k)
            else:
                yield from (k + self.sep + x for x in self.keys(v))

    def getitem(self, d, key_path, default_val=None):
        """
        getting with a key list or "."-separated string
        :param d: dict-like
        :param key_path: list or "."-separated string of keys
        :return:
        """
        if isinstance(key_path, str):
            key_path = key_path.split(self.sep)
        try:
            return reduce(operator.getitem, key_path, d)
        except (TypeError, KeyError):
            return default_val

    def setitem(self, d, key_path, val):
        """
        setting with a key list or "."-separated string
        :param d: dict
        :param key_path: list or "."-separated string of keys
        :param val: value to assign
        :return:
        """
        if isinstance(key_path, str):
            key_path = key_path.split(self.sep)
        self.getitem(d, key_path[:-1])[key_path[-1]] = val

    def setitem_recursive(self, d, key_path, val):
        """

        :param d:
        :param key_path:
        :param val:
        :return:

        >>> kp = KeyPathTrans()
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
        >>> kp.setitem_recursive(input_dict, 'new.key.path', 7)
        >>> input_dict
        {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': {'path': 7}}}
        >>> kp.setitem_recursive(input_dict, 'new.key.old.path', 8)
        >>> input_dict
        {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': {'path': 7, 'old': {'path': 8}}}}
        >>> kp.setitem_recursive(input_dict, 'new.key', 'new val')
        >>> input_dict
        {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': 'new val'}}
        """
        if isinstance(key_path, str):
            key_path = key_path.split(self.sep)
        first_key = key_path[0]
        if len(key_path) == 1:
            d[first_key] = val
        else:
            if first_key in d:
                self.setitem_recursive(d[first_key], key_path[1:], val)
            else:
                d[first_key] = self.node_type()
                self.setitem_recursive(d[first_key], key_path[1:], val)

    def extract_key_paths(
        self, d, key_paths, field_naming='full', use_default=False, default_val=None,
    ):
        """
        getting with a key list or "."-separated string
        :param d: dict-like
        :param key_path: list or "."-separated string of keys
        :param field_naming: 'full' (default) will use key_path strings as is, leaf will only use the last dot item
            (i.e. this.is.a.key.path will result in "path" being used)
        :return:

        >>> kp = KeyPathTrans()
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
        >>> kp.extract_key_paths(d, 'a')
        {'a': {'a': 'a.a', 'b': 'a.b', 'c': {'a': 'a.c.a'}}}
        >>> kp.extract_key_paths(d, 'a.a')
        {'a.a': 'a.a'}
        >>> kp.extract_key_paths(d, 'a.c')
        {'a.c': {'a': 'a.c.a'}}
        >>> kp.extract_key_paths(d, ['a.a', 'a.c'])
        {'a.a': 'a.a', 'a.c': {'a': 'a.c.a'}}
        >>> kp.extract_key_paths(d, ['a.a', 'something.thats.not.there'])  # missing key just won't be included
        {'a.a': 'a.a'}
        >>> kp.extract_key_paths(d, ['a.a', 'something.thats.not.there'], use_default=True, default_val=42)
        {'a.a': 'a.a', 'something.thats.not.there': 42}
        """
        dd = self.mk_new_node()
        if isinstance(key_paths, str):
            key_paths = [key_paths]
        if isinstance(key_paths, self.node_type):
            key_paths = [k for k, v in key_paths.items() if v]

        for key_path in key_paths:

            if isinstance(key_path, str):
                field = key_path
                key_path = key_path.split(self.sep)
            else:
                field = self.sep.join(key_path)

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


# from collections.abc import MutableMapping
#
#
# class AttrStore(MutableMapping):
#     def __getitem__(self, k):
#         if not hasattr(self, k):
#             setattr(self, k, AttrStore())
#         return getattr(self, k)
#
#     def __setitem__(self, k, v):
#         return setattr(self, k, v)
#
#     def __delitem__(self, k):
#         return delattr(self, k)
#
#     def __iter__(self):
#         return iter(self.__dict__)
#
#     def keys(self):
#         return self.__dict__.keys()
#
#     def __contains__(self, k):
#         return hasattr(self, k)
#
#     def __len__(self):
#         return len(list(self))
#
#
# d = AttrStore()
# kp = KeyPathMap(d)
# kp['a'] = '2a'
# print(kp.store.a)
# print(list(kp))
# kp['c']['d'] = 3
# print(list(kp))
# kp['c', 'z'] = 'hi world'
# print(list(kp.items()))

## Not working yet:
# d = AttrStore()
# kp = StrKeyPath(d, sep='.')
# kp['a'] = '2a'
# print(kp.store.a)
# print(list(kp))
# kp['c']['d'] = 3
# print(list(kp))
# kp['c.z'] = 'hi world'
# list(kp.items())

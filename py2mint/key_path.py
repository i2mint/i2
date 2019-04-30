from functools import reduce
import operator


class KeyPathTrans:

    def __init__(self, sep: str='.', store_type: type=dict, mk_empty_store=None):
        """

        :param sep:
        :param store_type:
        """
        self.sep = sep
        self.store_type = store_type
        if mk_empty_store is None:
            mk_empty_store = store_type
        self.mk_empty_store = mk_empty_store

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
                if not isinstance(v, self.store_type):
                    yield k, v
                else:
                    for kk, vv in self.items(v, k):
                        yield kk, vv
        else:
            for k, v in d.items():
                if not isinstance(v, self.store_type):
                    yield key_path_prefix + self.sep + k, v
                else:
                    for kk, vv in self.items(v, k):
                        yield key_path_prefix + self.sep + kk, vv

    def keys(self, d):
        for k, v in d.items():
            if not isinstance(v, self.store_type):
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
                d[first_key] = self.store_type()
                self.setitem_recursive(d[first_key], key_path[1:], val)

    def extract_key_paths(self, d, key_paths, field_naming='full', use_default=False, default_val=None):
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
        dd = self.mk_empty_store()
        if isinstance(key_paths, str):
            key_paths = [key_paths]
        if isinstance(key_paths, self.store_type):
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

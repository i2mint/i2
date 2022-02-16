"""Merge mappings

Marked for deprecation.
"""

from collections import ChainMap

from collections.abc import Mapping
from itertools import chain

is_mapping = lambda x: isinstance(x, Mapping)
not_mapping = lambda x: not (is_mapping(x))

from collections.abc import Iterable


def is_iterable(x):
    """Similar in nature to :func:`callable`, ``is_iterable`` returns
    ``True`` if an object is `iterable`_, ``False`` if not.

    >>> is_iterable([])
    True
    >>> is_iterable(1)
    False"""
    return isinstance(x, Iterable)


# Note: Mark for moving --> used in only one known place
def unique_iter(src, key=None):
    """Yield unique elements from the iterable, *src*, based on *key*,
    in the order in which they first appeared in *src*.

    >>> repetitious = [1, 2, 3] * 10
    >>> list(unique_iter(repetitious))
    [1, 2, 3]

    By default, *key* is the object itself, but *key* can either be a
    callable or, for convenience, a string name of the attribute on
    which to uniqueify objects, falling back on identity when the
    attribute is not present.

    >>> pleasantries = ['hi', 'hello', 'ok', 'bye', 'yes']
    >>> list(unique_iter(pleasantries, key=lambda x: len(x)))
    ['hi', 'hello', 'bye']
    """
    if not is_iterable(src):
        raise TypeError('expected an iterable, not %r' % type(src))
    if key is None:
        key_func = lambda x: x
    elif callable(key):
        key_func = key
    elif isinstance(key, str):
        key_func = lambda x: getattr(x, key, x)
    else:
        raise TypeError('"key" expected a string or callable, not %r' % key)
    seen = set()
    for i in src:
        k = key_func(i)
        if k not in seen:
            seen.add(k)
            yield i
    return


# Note: Mark for moving to py2store --> used in only one place (py2dash).
class ChainMapTree(Mapping):
    """Combine/overlay multiple hierarchical mappings. This efficiently merges
    multiple hierarchical (could be several layers deep) dictionaries, producing
    a new view into them that acts exactly like a merged dictionary, but without
    doing any copying.
    Because it doesn't actually copy the data, it is intended to be used only
    with immutable mappings. It is safe to change *leaf* data values,
    and the results will be reflected here, but changing the structure of any
    of the trees will not work.

    >>> base1 = {
    ...     'a1': 'base1.a1',
    ...     'a2': 'base1.a2',
    ...     'a3': {
    ...         'b1': 'base1.a3.b1',
    ...         'b2': 'base1.a3.b2',
    ...     },
    ... }
    >>> base2 = {
    ...     'a2': 'base2.a2',
    ...     'a3': {
    ...         'b2': 'base2.a3.b2',
    ...         'b4': 'base2.a3.b4',
    ...     },
    ...     'a4': 'base2.a4',
    ... }
    >>>
    >>> cm = ChainMapTree(base1, base2)
    >>> cm['a1']
    'base1.a1'
    >>> cm['a2']
    'base1.a2'
    >>> cm['a4']
    'base2.a4'
    >>> cm['a3']
    ChainMapTree({'b1': 'base1.a3.b1', 'b2': 'base1.a3.b2'}, {'b2': 'base2.a3.b2', 'b4': 'base2.a3.b4'})
    >>> cm['a3']['b1']
    'base1.a3.b1'
    >>> cm['a3']['b4']
    'base2.a3.b4'
    >>> cm = ChainMapTree(base2, base1)
    >>> cm['a1']
    'base1.a1'
    >>> cm['a2']
    'base2.a2'
    >>> cm['a4']
    'base2.a4'
    >>> cm['a3']
    ChainMapTree({'b2': 'base2.a3.b2', 'b4': 'base2.a3.b4'}, {'b1': 'base1.a3.b1', 'b2': 'base1.a3.b2'})
    >>> cm['a3']['b2']
    'base2.a3.b2'
    >>> cm['a3']['b1']
    'base1.a3.b1'
    >>>
    >>> # Let's do a ChainMapTree with THREE bases now!
    >>> base3 = {
    ...     'a2': 'base3.a2',
    ...     'a3': {
    ...         'b2': 'base3.a3.b2',
    ...         'b4': 'base3.a3.b4',
    ...     },
    ...     'a4': 'base3.a4',
    ... }
    >>> cm = ChainMapTree(base3, base2, base1)
    >>> cm['a2']  # will get it from base3
    'base3.a2'
    >>> cm['a3']['b2']  # will get it from base3 (not base2)
    'base3.a3.b2'
    >>> cm['a3']['b1']  # will get it from base1 (since no one else has it)
    'base1.a3.b1'

    Based on: https://gist.github.com/Klortho/7d83975559bdcc47ac64fd7d877934f6
    """

    _max_repr_keys = (
        7  # Not used (anymore) at this point. Was to control __repr__ output length
    )

    def __init__(self, *maps):
        _maps = list(maps)

        # All keys of kids that are mappings
        kid_keys = set([key for m in maps for key in m.keys() if is_mapping(m[key])])

        # This will be a dictionary of lists of mappings
        kid_maps = {}
        for key in kid_keys:
            # The list of child mappings for this key
            kmaps = [m[key] for m in maps if key in m]
            # Make sure they are *all* mappings
            if any(map(not_mapping, kmaps)):
                raise KeyError(key)
            # Recurse
            kid_maps[key] = ChainMapTree(*kmaps)

        # If non-empty, prepend it to the existing list
        if len(kid_maps.keys()) > 0:
            _maps.insert(0, kid_maps)

        self._maps = _maps

    def __getitem__(self, key):
        for _map in self._maps:
            try:
                return _map[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __iter__(self):
        return unique_iter(chain(*(_map.__iter__() for _map in self._maps)))

    def __len__(self):
        return len(list(self.__iter__()))

    def _mk_keys_str(self, keys):
        first_keys = list()
        i = -1
        for i, k in enumerate(keys):
            if i < self._max_repr_keys:
                first_keys.append(k)
            else:
                break
        keys_str = ', '.join(first_keys)
        if i >= self._max_repr_keys:
            keys_str += ', ...'
        return keys_str

    def to_dict(self):
        """Convert to dict

        >>> a = {'a': {'x': 1, 'z': 3}, 'foo': "a's foo"}
        >>> b = {'a': {'y': 222, 'z': 333}, 'foo': "b's foo"}
        >>> cm = ChainMapTree(a, b)
        >>> # It acts like a dict when you ask for items, but is not a dict. If you want a dict, do this:
        >>> cm.to_dict()
        {'a': {'x': 1, 'z': 3, 'y': 222}, 'foo': "a's foo"}
        >>> # Compare to normal/flat/not-nested chaining:
        >>> dict(a, **b)   # Note the precedence is the inverse of ChainMapTree!
        {'a': {'y': 222, 'z': 333}, 'foo': "b's foo"}
        >>>
        >>> # See what you get if you specify b before a
        >>> ChainMapTree(b, a).to_dict()
        {'a': {'y': 222, 'z': 333, 'x': 1}, 'foo': "b's foo"}
        >>> # Compare to normal/flat/not-nested chaining:
        >>> dict(b, **a)  # Note the precedence is the inverse of ChainMapTree!
        {'a': {'x': 1, 'z': 3}, 'foo': "a's foo"}
        """
        d = dict()
        for k in self:
            v = self[k]
            if not isinstance(v, ChainMapTree):
                d[k] = v
            else:
                d[k] = v.to_dict()
        return d

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join((_map.__repr__() for _map in self._maps))})"
        #         map_keys_strs = self._mk_keys_str(self.keys())
        #         map_keys_strs = []
        #         for _map in self._maps:
        #             map_keys_strs.append(self._mk_keys_str(_map.keys()))
        #         map_keys_strs = ', '.join(map('({})'.format, map_keys_strs))
        # return f"{self.__class__.__name__}({map_keys_strs})"

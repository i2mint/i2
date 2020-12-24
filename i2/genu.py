from typing import Iterable, Callable, Optional, Any, Hashable
from collections import defaultdict


def groupby(
    items: Iterable[Any],
    key: Callable[[Any], Hashable],
    val: Optional[Callable[[Any], Any]] = None,
    group_factory=list,
) -> dict:
    """Groups items according to group keys updated from those items through the given (item_to_)key function.

    Args:
        items: iterable of items
        key: The function that computes a key from an item. Needs to return a hashable.
        val: An optional function that computes a val from an item. If not given, the item itself will be taken.
        group_factory: The function to make new (empty) group objects and accumulate group items.
            group_items = group_collector() will be called to make a new empty group collection
            group_items.append(x) will be called to add x to that collection
            The default is `list`

    Returns: A dict of {group_key: items_in_that_group, ...}

    >>> groupby(range(11), key=lambda x: x % 3)
    {0: [0, 3, 6, 9], 1: [1, 4, 7, 10], 2: [2, 5, 8]}
    >>>
    >>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
    >>> groupby(tokens, len)
    {3: ['the', 'fox', 'box'], 2: ['is', 'in'], 1: ['a']}
    >>> key_map = {1: 'one', 2: 'two'}
    >>> groupby(tokens, lambda x: key_map.get(len(x), 'more'))
    {'more': ['the', 'fox', 'box'], 'two': ['is', 'in'], 'one': ['a']}
    >>> stopwords = {'the', 'in', 'a', 'on'}
    >>> groupby(tokens, lambda w: w in stopwords)
    {True: ['the', 'in', 'a'], False: ['fox', 'is', 'box']}
    >>> groupby(tokens, lambda w: ['words', 'stopwords'][int(w in stopwords)])
    {'stopwords': ['the', 'in', 'a'], 'words': ['fox', 'is', 'box']}
    """
    groups = defaultdict(group_factory)
    if val is None:
        for item in items:
            groups[key(item)].append(item)
    else:
        for item in items:
            groups[key(item)].append(val(item))
    return dict(groups)


def regroupby(items, *key_funcs, **named_key_funcs):
    """REcursive groupby. Applies the groupby function recursively, using a sequence of key functions.

    Note: The named_key_funcs argument names don't have any external effect.
        They just give a name to the key function, for code reading clarity purposes.

    >>> # group by how big the number is, then by it's mod 3 value
    >>> # note that named_key_funcs argument names doesn't have any external effect (but give a name to the function)
    >>> regroupby([1, 2, 3, 4, 5, 6, 7], lambda x: 'big' if x > 5 else 'small', mod3=lambda x: x % 3)
    {'small': {1: [1, 4], 2: [2, 5], 0: [3]}, 'big': {0: [6], 1: [7]}}
    >>>
    >>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
    >>> stopwords = {'the', 'in', 'a', 'on'}
    >>> word_category = lambda x: 'stopwords' if x in stopwords else 'words'
    >>> regroupby(tokens, word_category, len)
    {'stopwords': {3: ['the'], 2: ['in'], 1: ['a']}, 'words': {3: ['fox', 'box'], 2: ['is']}}
    >>> regroupby(tokens, len, word_category)
    {3: {'stopwords': ['the'], 'words': ['fox', 'box']}, 2: {'words': ['is'], 'stopwords': ['in']}, 1: {'stopwords': ['a']}}
    """
    key_funcs = list(key_funcs) + list(named_key_funcs.values())
    assert len(key_funcs) > 0, 'You need to have at least one key_func'
    if len(key_funcs) == 1:
        return groupby(items, key=key_funcs[0])
    else:
        key_func, *key_funcs = key_funcs
        groups = groupby(items, key=key_func)
        return {
            group_key: regroupby(group_items, *key_funcs)
            for group_key, group_items in groups.items()
        }

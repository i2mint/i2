def dp_get(d, dot_path):
    """
    Get stuff from a dict, using dot_paths (i.e. 'foo.bar' instead of ['foo']['bar'])
    >>> d = {'foo': {'bar': 2, 'alice': 'bob'}, 3: {'pi': 3.14}}
    >>> assert dp_get(d, 'foo') == {'bar': 2, 'alice': 'bob'}
    >>> assert dp_get(d, 'foo.bar') == 2
    >>> assert dp_get(d, 'foo.alice') == 'bob'
    """
    components = dot_path.split('.')
    dd = d.get(components[0])
    for comp in components[1:]:
        dd = dd.get(comp)
    return dd


class lazyprop:
    """
    A descriptor implementation of lazyprop (cached property) from David Beazley's "Python Cookbook" book.
    It's
    >>> class Test:
    ...     def __init__(self, a):
    ...         self.a = a
    ...     @lazyprop
    ...     def len(self):
    ...         print('generating "len"')
    ...         return len(self.a)
    >>> t = Test([0, 1, 2, 3, 4])
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4]}
    >>> t.len
    generating "len"
    5
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4], 'len': 5}
    >>> t.len
    5
    >>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
    >>> t.a = [0, 1, 2]  # if we change a...
    >>> t.len  # ... we still get the old cached value of len
    5
    >>> del t.len  # if we delete the len prop
    >>> t.len  # ... then len being recomputed again
    generating "len"
    3
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class FrozenHashError(TypeError):
    pass


class FrozenDict(dict):
    """An immutable dict subtype that is hashable and can itself be used
    as a :class:`dict` key or :class:`set` entry. What
    :class:`frozenset` is to :class:`set`, FrozenDict is to
    :class:`dict`.

    There was once an attempt to introduce such a type to the standard
    library, but it was rejected: `PEP 416 <https://www.python.org/dev/peps/pep-0416/>`_.

    Because FrozenDict is a :class:`dict` subtype, it automatically
    works everywhere a dict would, including JSON serialization.

    """
    __slots__ = ('_hash',)

    def updated(self, *a, **kw):
        """Make a copy and add items from a dictionary or iterable (and/or
        keyword arguments), overwriting values under an existing
        key. See :meth:`dict.update` for more details.
        """
        data = dict(self)
        data.update(*a, **kw)
        return type(self)(data)

    @classmethod
    def fromkeys(cls, keys, value=None):
        # one of the lesser known and used/useful dict methods
        return cls(dict.fromkeys(keys, value))

    def __repr__(self):
        cn = self.__class__.__name__
        return '%s(%s)' % (cn, dict.__repr__(self))

    def __reduce_ex__(self, protocol):
        return type(self), (dict(self),)

    def __hash__(self):
        try:
            ret = self._hash
        except AttributeError:
            try:
                ret = self._hash = hash(frozenset(self.items()))
            except Exception as e:
                ret = self._hash = FrozenHashError(e)

        if ret.__class__ is FrozenHashError:
            raise ret

        return ret

    def __copy__(self):
        return self  # immutable types don't copy, see tuple's behavior

    # block everything else
    def _raise_frozen_typeerror(self, *a, **kw):
        "raises a TypeError, because FrozenDicts are immutable"
        raise TypeError('%s object is immutable' % self.__class__.__name__)

    __setitem__ = __delitem__ = update = _raise_frozen_typeerror
    setdefault = pop = popitem = clear = _raise_frozen_typeerror

    del _raise_frozen_typeerror

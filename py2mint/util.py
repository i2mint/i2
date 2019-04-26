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

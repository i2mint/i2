from typing import NewType, Optional, Iterable


def new_type(
    name,
    tp,
    doc: Optional[str] = None,
    aka: Optional[Iterable] = None,
    assign_to_globals=False,
):
    """
    Make a new type with (optional) doc and (optional) aka, set of var names it often appears as

    Args:
        name: Name to give the variable
        tp: type (see typing module)
        doc: Optional string to put in __doc__ attribute
        aka: Optional set (or any iterable) to put in _aka attribute,
            meant to list names the variables of this type often appear as.

    Returns: None

    >>> from typing import Any, Union, List
    >>> MyType = new_type('MyType', int)
    >>> type(MyType)
    <class 'function'>
    >>> Key = new_type('Key', Any, aka=['key', 'k'])
    >>> sorted(Key._aka)
    ['k', 'key']
    >>> Val = new_type('Val', Union[int, float, List[Union[int, float]]], doc="A number or list of numbers.")
    >>> Val.__doc__
    'A number or list of numbers.'
    """
    new_tp = NewType(name, tp)
    if doc is not None:
        setattr(new_tp, '__doc__', doc)
    if aka is not None:
        setattr(new_tp, '_aka', set(aka))
    if assign_to_globals:
        globals()[
            name
        ] = new_tp  # not sure how kosher this is... Should only use at top level of module, for sure!
    return new_tp

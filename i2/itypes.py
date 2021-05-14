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
        setattr(new_tp, "__doc__", doc)
    if aka is not None:
        setattr(new_tp, "_aka", set(aka))
    if assign_to_globals:
        globals()[
            name
        ] = new_tp  # not sure how kosher this is... Should only use at top level of module, for sure!
    return new_tp


class HasAttrs:
    """
    Make a protocol to express the existence of specific attributes.

    >>> SizedAndAppendable = HasAttrs["__len__", "append"]
    >>> assert isinstance([1, 2, 3], SizedAndAppendable)  # lists have both a length and an append
    >>> assert not isinstance((1, 2, 3), SizedAndAppendable)  # tuples don't have an append

    [Python Protocols](https://www.python.org/dev/peps/pep-0544/) are a way to be able to do
    "behavior typing" (my bad terminology).
    Basically, if you want your static analyzer
    (the swingles in your IDE, or linter validation process...)
    to check if you're manipulating the expected types, except the types
    (classes, subclasses, ABCs, abstract classes...) are too restrictive (they are!),
    you can use Protocols to fill the gap.

    Except writing them can sometimes be verbose.

    With HasAttrs you can have the basic "does it have these attributes" cases covered.

    >>> assert isinstance(dict(), HasAttrs["items"])
    >>> assert not isinstance(list(), HasAttrs["items"])
    >>> assert not isinstance(dict(), HasAttrs["append"])
    >>>
    >>> class A:
    ...     prop = 2
    ...
    ...     def method(self):
    ...         pass
    >>>
    >>> a = A()
    >>> assert isinstance(a, HasAttrs["method"])
    >>> assert isinstance(a, HasAttrs["method", "prop"])
    >>> assert not isinstance(a, HasAttrs["method", "prop", "this_attr_does_not_exist"])

    """

    def __class_getitem__(self, *attr_names):
        if len(attr_names) == 1 and isinstance(attr_names[0], tuple):
            attr_names = attr_names[0]
        str_to_exec = f"class HasAttrs(Protocol):\n" + "\n".join(
            f"\t{attr}: Any" for attr in attr_names
        )
        exec(str_to_exec)
        return locals()["HasAttrs"]

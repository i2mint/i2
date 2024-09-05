"""Types"""

from typing import (
    NewType,
    Optional,
    Iterable,
    Protocol,
    Any,
    runtime_checkable,
    get_args,
    Literal,
    get_origin,
)


from inspect import signature
from functools import wraps


# Note: Simplified func-applied version of i2.Sig.kwargs_for_args_and_kwargs
def _arg_name_and_val_dict(func, *args, **kwargs):
    """

    :param func:
    :param args:
    :param kwargs:
    :return:

    >>> def foo(x, /, y, *, z=3): ...
    >>> _arg_name_and_val_dict(foo, 1, 2, z=4)
    {'x': 1, 'y': 2, 'z': 4}

    """
    b = signature(func).bind(*args, **kwargs)
    b.apply_defaults()
    return dict(b.arguments)


def _annotation_is_literal(typ):
    return get_origin(typ) is Literal


def _literal_values(literal):
    """
    Get the values of a Literal type.

    :param literal:
    :return:

    >>> from typing import Literal
    >>> _literal_values(Literal[1, 2, 3])
    (1, 2, 3)
    """
    return literal.__args__


def _value_is_in_literal(value, literal):
    return value in _literal_values(literal)


def _validate_that_value_is_in_literal(name, value, literal):
    if not _value_is_in_literal(value, literal):
        error = ValueError(
            f'{value} is an invalid value for {name}. '
            f'Values should be one of the following: {literal.__args__}'
        )
        error.allowed_values = literal.__args__
        error.input_value = value
        raise error


# TODO: Not picklable. Make with wrapper module, or let this module be independent?
# TODO: Could use Sig. Should we, or let this module be independent?
# TODO: Add control over error message/type?
def validate_literal(func):
    """
    Decorator to validate (Literal-annotated) argument values at call time.

    Wraps a function to add validation of the input arguments annotated with Literal
    against the values listed by the literal. If the input argument is not one of the
    literal values, a ValueError is raised.

    >>> @validate_literal
    ... def f(x: Literal[1, 2, 3]):
    ...     return x
    >>> f(1)
    1
    >>> f(4)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    ValueError: 4 is an invalid value for x. Values should be one of the following: (1, 2, 3)

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        _kwargs = _arg_name_and_val_dict(func, *args, **kwargs)
        for arg_name, arg_type in func.__annotations__.items():
            if _annotation_is_literal(arg_type):
                arg_val = _kwargs[arg_name]
                _validate_that_value_is_in_literal(arg_name, arg_val, arg_type)
        return func(*args, **kwargs)

    return wrapper


def iterable_to_literal(iterable: Iterable):
    """
    Convert an iterable to a Literal type.

    >>> iterable_to_literal([1, 2, 3])
    typing.Literal[1, 2, 3]

    """
    return Literal.__getitem__(tuple(iterable))


def new_type(
    name,
    tp,
    doc: Optional[str] = None,
    aka: Optional[Iterable] = None,
    assign_to_globals=False,
):
    """
    Make a new type with (optional) doc and (optional) aka, set of var names it often
    appears as

    Args:
        name: Name to give the variable
        tp: type (see typing module)
        doc: Optional string to put in __doc__ attribute
        aka: Optional set (or any iterable) to put in _aka attribute,
            meant to list names the variables of this type often appear as.

    Returns: None

    >>> from typing import Any, Union, List
    >>> MyType = new_type('MyType', int)
    >>> # TODO: Skipping the next part because outputs <class 'typing.NewType'> in 3.10
    >>> type(MyType)  # doctest: +SKIP
    <class 'function'>
    >>> Key = new_type('Key', Any, aka=['key', 'k'])
    >>> sorted(Key._aka)
    ['k', 'key']
    >>> Val = new_type(
    ... 'Val', Union[int, float, List[Union[int, float]]],
    ... doc="A number or list of numbers.")
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

    def __class_getitem__(self, attr_names):
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        assert all(map(str.isidentifier, attr_names)), (
            f"The following are not valid python 'identifiers' "
            f"{', '.join(a for a in attr_names if not a.isidentifier())}"
        )

        annotations = {attr: Any for attr in attr_names}

        @runtime_checkable
        class HasAttrs(Protocol):
            __annotations__ = annotations

            # def __repr__(self):  # TODO: This is for the instance, need it for the class
            #     return "HasAttrs[{', '.join(annotations)}]"

        return HasAttrs

        # return type(
        #     "HasAttrs",
        #     (Protocol,),
        #     {"__annotations__": {attr: Any for attr in attr_names}},
        # )


# TODO: Complete scary hack: Find another way (see uses)
def is_a_new_type(typ):
    return (
        callable(typ)
        and getattr(typ, '__qualname__', '').startswith('NewType')
        and hasattr(typ, '__supertype__')
    )


def typ_name(typ):
    if is_a_new_type(typ):
        return typ.__name__
    else:
        return typ._name


def is_callable_kind(typ):
    """
    >>> from typing import Callable, Tuple
    >>> is_callable_kind(Callable)
    True
    >>> is_callable_kind(Callable[[int, float], str])
    True
    >>> is_callable_kind(Tuple[int, float, str])
    False
    """
    if is_a_new_type(typ):
        return is_callable_kind(typ.__supertype__)
    return typ_name(typ) == 'Callable'
    # Also possible: typ.mro()[0] == __import__('collections.abc').Callable


def input_and_output_types(typ: type):
    """Returns the input and output types

    >>> from typing import Callable, Tuple
    >>> input_types, output_type = input_and_output_types(Callable[[float, int], str])
    >>> assert input_types == [float, int] and output_type == str
    >>> input_types, output_type = input_and_output_types(Callable[[], str])
    >>> assert input_types == [] and output_type == str

    But will fail if `typ` isn't a `Callable`:

    >>> input_and_output_types(Tuple[float, int, str])  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    AssertionError: Is not a typing.Callable kind: typing.Tuple[float, int, str]

    Will also fail if `typ` is a Callable but not "parametrized".

    >>> input_and_output_types(Callable)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    AssertionError: Can only be used on a Callable[[...],...] kind: typing.Callable

    """
    if is_a_new_type(typ):
        return input_and_output_types(typ.__supertype__)
    assert is_callable_kind(typ), f'Is not a typing.Callable kind: {typ}'
    typ_args = get_args(typ)
    assert len(typ_args) > 0, f'Can only be used on a Callable[[...],...] kind: {typ}'
    return typ_args[0], typ_args[1]


def dot_string_of_callable_typ(typ):
    input_types, output_type = input_and_output_types(typ)
    return (
        ','.join(map(typ_name, input_types))
        + f' -> {typ_name(typ)} -> '
        + typ_name(output_type)
    )


def dot_strings_of_callable_types(*typs, func_shape='box'):
    for typ in typs:
        yield dot_string_of_callable_typ(typ)
        yield f'{typ_name(typ)} [shape="{func_shape}"]'


# --------------------------------------------------------------------------------------
# Misc

from typing import Callable, Any, Dict, Optional, Iterator, Iterable, KT

# Define a type alias for clarity in the code
ObjectType = Any


class ObjectClassifier:
    """
    A general-purpose classifier for objects based on a set of verifying functions.

    Each "verifier" checks whether an object belongs to a certain kind (category).

    Example usage:

    >>> from typing import Mapping, Iterable
    >>>
    >>> obj = "test"
    >>> isa = lambda typ: lambda obj: isinstance(obj, typ)
    >>> verifiers = {
    ...     'str': isa(str),
    ...     'mapping': isa(Mapping),
    ...     'iterable': isa(Iterable)
    ... }
    >>> classifier = ObjectClassifier(verifiers)

    Check if the object matches any kind

    >>> classifier.matches(obj)
    True

    Check if the object matches a specific kind

    >>> classifier.matches(obj, 'str')
    True
    >>> classifier.matches(obj, 'mapping')
    False

    Get all matches

    >>> classifier.all_matches(obj)
    {'str': True, 'mapping': False, 'iterable': True}

    Find all matching kinds

    >>> list(classifier.matching_kinds(obj))
    ['str', 'iterable']

    Find the first matching kind (default is to ensure uniqueness, which will fail here)

    >>> classifier.matching_kind(obj)
    Traceback (most recent call last):
      ...
    ValueError: Multiple matches found: ['str', 'iterable']

    Find the first matching kind without uniqueness check

    >>> classifier.matching_kind(obj, assert_unique=False)
    'str'

    """

    def __init__(self, verifiers: Dict[KT, Callable[[ObjectType], bool]]):
        """
        Initialize with a dictionary of verifiers. Each verifier is a function
        that returns True or False based on the object's classification.

        :param verifiers: A dictionary mapping kind names (keys) to verifying functions (values).
        """
        self.verifiers = verifiers

    def matches(self, obj: ObjectType, kind: Optional[KT] = None) -> bool:
        """
        Returns True if the object matches the given kind, or matches any kind
        if kind is None.

        :param obj: The object to classify.
        :param kind: The specific kind (verifier key) to check.
        :return: True if the object matches the given or any kind.
        """
        if kind is None:
            # Check against all kinds if kind is not specified
            return any(verifier(obj) for verifier in self.verifiers.values())
        # Check for the specific kind
        return self.verifiers.get(kind, lambda x: False)(obj)

    def all_matches(self, obj: ObjectType) -> Dict[KT, bool]:
        """
        Returns a dictionary indicating if the object matches each kind.

        :param obj: The object to classify.
        :return: A dictionary with kind names as keys and True/False as values.
        """
        return {kind: verifier(obj) for kind, verifier in self.verifiers.items()}

    def matching_kind(
        self, obj: ObjectType, *, assert_unique: bool = True
    ) -> Optional[KT]:
        """
        Returns the first kind that matches the object. If assert_unique is True,
        it asserts that only one match exists. Optionally, it can return the value instead of the key.

        :param obj: The object to classify.
        :param assert_unique: Ensures only one kind matches, if True.
        :return: The key of the first matching kind, or None if no match.
        """
        matches = list(self.matching_kinds(obj))
        if assert_unique and len(matches) > 1:
            raise ValueError(f'Multiple matches found: {matches}')
        return matches[0] if matches else None

    def matching_kinds(self, obj: ObjectType) -> Iterator[KT]:
        """
        Returns an iterator of kinds that match the object.

        :param obj: The object to classify.
        :return: An iterator of matching kinds.
        """
        for kind, verifier in self.verifiers.items():
            if verifier(obj):
                yield kind

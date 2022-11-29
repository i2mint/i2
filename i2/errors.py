"""Error objects"""
import dataclasses
from typing import Callable, Mapping, Union, Any, Type
from functools import partial
from contextlib import AbstractContextManager


class DataError(Exception):
    pass


class DuplicateRecordError(DataError):
    pass


class NotFoundError(DataError):
    pass


class AuthorizationError(Exception):
    pass


class OverwritesNotAllowed(AuthorizationError):
    """To raise when writes are only allowed if the item doesn't already exist"""

    def __init__(self, *args, forbidden_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden_keys = forbidden_keys

    @classmethod
    def for_key(cls, key):
        return OverwritesNotAllowed(
            f"You're not allowed to overwrite to the value of {key}",
            forbidden_keys={key},
        )

    @classmethod
    def for_keys(cls, keys):
        return OverwritesNotAllowed(
            f"You're not allowed to overwrite to the values of {', '.join(keys)}",
            forbidden_keys=keys,
        )


class ForbiddenError(AuthorizationError):
    pass


class InputError(Exception):
    pass


class ModuleNotFoundIgnore:
    """Context manager meant to ignore import errors.
    The use case in mind is when we want to condition some code on the existence of some package.
    """

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


OnErrorCallback = Union[None, str, Callable[[], Any]]
ExceptionType = Type[BaseException]


def log_and_return(msg, logger=print):
    logger(msg)
    return msg


class InterruptWithBlock(BaseException):
    """To be used to interrupt the march of a with"""


# Note: Can be extended to have more precise handled conditions and callbacks
#  (involving exc_val and exc_tb)
# Note: Efforts towards a more general version here:
#   https://github.com/thorwhalen/ut/blob/33c20ce76fe0f9dcc6aa197b0a2dbbf3d7b1d5be/errors.py#L90
@dataclasses.dataclass
class HandleExceptions(AbstractContextManager):
    """A context manager that catches and (specifically) handles specific exceptions.

    It takes one argument: A dict (or mapping) of exception type keys and callback
    values. If within a with block, the particular (listed) exception happens,
    the callback is called and it's returned value is assigned to the
    `HandleExceptions` instance's `.exit_value` attribute.
    That attribute will only exist if the with block existed with an exception
    caught by `HandleExceptions`.

    A callback is an argument-less function. If you need to specify arguments, you can
    envoke the command pattern, using `functools.partial` to make a argument-less
    function. See in the example below how we ask `HandleExceptions` to print a
    specific string if a `ZeroDivisionError` happens:

    >>> from functools import partial
    >>> def print_and_return(msg):
    ...     print(msg)
    ...     return msg
    >>> with HandleExceptions({
    ...     ZeroDivisionError: partial(print_and_return, "You interrupted me"),
    ...     KeyboardInterrupt: lambda: 'imagine this is code to notify someone'
    ... }) as he:
    ...     print('This works')
    ...     0 / 0
    ...
    This works
    You interrupted me

    You can check if the context exited with a handled exception, and if so, what
    the callback returned value was.

    >>> he.exited_with_handled_exception()
    True
    >>> he.exit_value
    'You interrupted me'

    Also available, whether the exception was a handled one or not: The exception
    instance itself:

    >>> he.exited_with_exception
    ZeroDivisionError('division by zero')

    If all you want to do though is print a string (and have the same string
    available in the `exit_value` attribute, we got you covered!
    Just specify a string and we'll make that printer callaback for you!

    >>> from functools import partial
    >>>
    >>> with HandleExceptions({ZeroDivisionError: "You interrupted me again!"}):
    ...     print('This also works')
    ...     0 / 0
    This also works
    You interrupted me again!

    Note that specifying `partial(print, "some message")` will work as a
    "printing callback", but the string won't be available in `.exit_value` since
    `print` returns None.

    A few recipes now...

    You can also use your own custom exception types to do things like interrupt
    a with block early given some condition(s).

    >>> with HandleExceptions(
    ...     {InterruptWithBlock: "The with block was interrupted early."}
    ... ):
    ...     print('before condition')
    ...     x = 5 % 2
    ...     if x:
    ...         raise InterruptWithBlock()
    ...     print('after condition')
    ...
    ...
    before condition
    The with block was interrupted early.

    Tip: If you need to do stuff with an exception, but reraise it, you can
    still do that in your callback. Just say `raise` at the end of the callback!

    >>> def print_and_raise(msg):
    ...     print(msg)
    ...     raise
    >>>
    >>> with HandleExceptions({  # doctest: +SKIP
    ...     ZeroDivisionError: partial(print_and_raise, "That again!"),
    ... }):
    ...     print('This also works')
    ...     0 / 0
    This also works
    That again!
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero
    """

    on_error: Mapping[ExceptionType, OnErrorCallback] = dataclasses.field(
        default_factory=dict
    )
    exited_with_exception = None

    def __post_init__(self):
        self.on_error = dict(self.on_error)
        for handled_exc_type, callback in self.on_error.items():
            if isinstance(callback, str):
                msg = callback
                self.on_error[handled_exc_type] = partial(log_and_return, msg)

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exited_with_exception = exc_val
            for handled_exc_type, callback in self.on_error.items():
                if issubclass(exc_type, handled_exc_type):
                    # if the exc_type is a subtype of handled_exc_type, call the callback
                    self.exit_value = callback()  # storing the result in exit_value
                    # and exit with True
                    return True  # suppress the exception
            # if there was an exception, but a handled one, raise it!
            raise

    def exited_with_handled_exception(self):
        return hasattr(self, 'exit_value')

    def initialize(self):
        if hasattr(self, 'exit_value'):
            delattr(self, 'exit_value')
        self.exited_with_exception = None

"""
Deprecated code I'm keeping around so I can at least reuse it's docs at some point.
"""

from functools import partial
from typing import Iterable, Callable, Any

from i2 import Sig, name_of_obj
from i2.signatures import extract_arguments


class Command:
    """A class that holds a `(caller, args, kwargs)` triple and allows one to execute
    `caller(*args, **kwargs)`

    :param func: A callable that will be called with (*args, **kwargs) argument.
    :param args: The positional arguments to call the func with.
    :param kwargs: The keyword arguments to call the func with.

    >>> c = Command(print, "hello", "world", sep=", ")
    >>> c()
    hello, world

    What happens (when a command is executed) if some of the arguments are commands
    themselves? Well, the sensible thing happens. These commands are executed.
    You can use this to define, declaratively, some pretty complex instructions, and
    only fetch the data you need and execute everything, once you're ready.

    >>> def show(a, b):
    ...     print(f"Showing this: {a=}, {b=}")
    >>> def take_five():
    ...     return 5
    >>> def double_val(val):
    ...     return val * 2
    >>> command = Command(
    ...     show,
    ...     Command(take_five),
    ...     b=Command(double_val, 'hello'),
    ... )
    >>> command
    Command(show, Command(take_five), b=Command(double_val, 'hello'))
    >>> command()
    Showing this: a=5, b='hellohello'

    Of course, as your use of Command gets more complex, you may want to subclass it
    and include some "validation" and "compilation" in the init.

    The usual way to call a function is to... erm... call it.
    But sometimes you want to do things differently.
    Like validate it, put it on a queue, etc.
    That's where specifying a different _caller will be useful.

    >>> class MyCommand(Command):
    ...     def _caller(self):
    ...         f, a, k = self.func, self.args, self.kwargs
    ...         print(f"Calling {f}(*{a}, **{k}) with result: {f(*a, **k)}")
    ...
    >>> c = MyCommand(print, "hello", "world", sep=", ")
    >>> c()
    hello, world
    Calling <built-in function print>(*('hello', 'world'), **{'sep': ', '}) with result: None

    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def curried(cls, func, **kw_defaults):
        """Get an Command maker for a specific function, with defaults and signature!

        >>> def foo(x: str, y: int):
        ...     return x * y
        ...
        >>> foo('hi', 3)
        'hihihi'
        >>>
        >>> foo_command = Command.curried(foo, y=2)
        >>> Sig(foo_command)
        <Sig (x: str, y: int = 2)>
        >>> f = foo_command('hi', y=4)
        >>> f()
        'hihihihi'
        >>> ff = foo_command('hi')
        >>> ff
        Command(foo, 'hi')
        >>> ff()
        'hihi'

        """
        sig = Sig(func)
        sig = sig.ch_defaults(**kw_defaults)

        if kw_defaults:
            func = partial(func, **kw_defaults)

        curried_command_cls = partial(Command, func)
        return sig(curried_command_cls)

    def __repr__(self):
        def to_str(x, quote="'"):
            if isinstance(x, str):
                return quote + x + quote
            else:
                return str(x)

        args_str = ', '.join(to_str(a) for a in self.args)
        kwargs_str = ', '.join(f'{k}={to_str(v)}' for k, v in self.kwargs.items())
        if args_str and kwargs_str:
            sep = ', '
        else:
            sep = ''
        args_kwargs_str = args_str + sep + kwargs_str

        func_name = name_of_obj(self.func)
        if args_kwargs_str:
            return f'{type(self).__name__}({func_name}, {args_kwargs_str})'
        else:
            return f'{type(self).__name__}({func_name})'

    def _caller(self):
        return self.func(*self.args, **self.kwargs)

    def _args_with_executed_commands(self):
        for v in self.args:
            if isinstance(v, Command):
                v = v()  # if a command, execute it
            yield v

    def _kwargs_with_executed_commands(self):
        for k, v in self.kwargs.items():
            if isinstance(v, Command):
                v = v()  # if a command, execute it
            yield k, v

    def _caller(self):
        return self.func(
            *self._args_with_executed_commands(),
            **dict(self._kwargs_with_executed_commands()),
        )

    def __call__(self):
        return self._caller()


def extract_commands(
    funcs: Iterable[Callable],
    *,
    mk_command: Callable[[Callable, tuple, dict], Any] = Command,
    what_to_do_with_remainding='ignore',
    **kwargs,
):
    """

    :param funcs: An iterable of functions
    :param mk_command: The function to make a command object
    :param kwargs: The argname=argval items that the functions should draw from.
    :return:

    >>> def add(a, b: float = 0.0) -> float:
    ...     return a + b
    >>> def mult(x: float, y=1):
    ...     return x * y
    >>> def formula1(w, /, x: float, y=1, *, z: int = 1):
    ...     return ((w + x) * y) ** z
    >>> commands = extract_commands(
    ...     (add, mult, formula1), a=1, b=2, c=3, d=4, e=5, w=6, x=7
    ... )
    >>> for command in commands:
    ...     print(
    ...         f"Calling {command.func.__name__} with "
    ...         f"args={command.args} and kwargs={command.kwargs}"
    ...     )
    ...     print(command())
    ...
    Calling add with args=() and kwargs={'a': 1, 'b': 2}
    3
    Calling mult with args=() and kwargs={'x': 7}
    7
    Calling formula1 with args=(6,) and kwargs={'x': 7}
    13
    """
    extract = partial(
        extract_arguments,
        what_to_do_with_remainding=what_to_do_with_remainding,
        include_all_when_var_keywords_in_params=False,
        assert_no_missing_position_only_args=True,
    )

    if callable(funcs):
        funcs = [funcs]

    for func in funcs:
        func_args, func_kwargs = extract(func, **kwargs)
        yield mk_command(func, *func_args, **func_kwargs)


def commands_dict(
    funcs,
    *,
    mk_command: Callable[[Callable, tuple, dict], Any] = Command,
    what_to_do_with_remainding='ignore',
    **kwargs,
):
    """

    :param funcs:
    :param mk_command:
    :param kwargs:
    :return:

    >>> def add(a, b: float = 0.0) -> float:
    ...     return a + b
    >>> def mult(x: float, y=1):
    ...     return x * y
    >>> def formula1(w, /, x: float, y=1, *, z: int = 1):
    ...     return ((w + x) * y) ** z
    >>> d = commands_dict((add, mult, formula1), a=1, b=2, c=3, d=4, e=5, w=6, x=7)
    >>> d[add]()
    3
    >>> d[mult]()
    7
    >>> d[formula1]()
    13

    """
    if callable(funcs):
        funcs = [funcs]
    it = extract_commands(
        funcs,
        what_to_do_with_remainding=what_to_do_with_remainding,
        mk_command=mk_command,
        **kwargs,
    )
    return dict(zip(funcs, it))

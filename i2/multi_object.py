"""A few fundamental tools to operate on a fixed pool of objects (e.g. functions).

For functions you have:

- `Pipe`: To compose functions (output of one fed as the input of the next)
- `FuncFanout`: To apply multiple functions to the same inputs
- `FlexFuncFanout`: Like `FuncFanout` but where the application of inputs is flexible.
    That is, the functions "draw" their inputs from the a common pool, but don't choke
    if there are extra unrecognized arguments.
- `ParallelFuncs`: To make a dict-to-dict function, applying a specific function for each
    input key (putting the result in that key in the output.

For context managers you have:

- `ContextFanout`: To hold multiple context managers as one (entering and exiting
    together)

![image](https://user-images.githubusercontent.com/1906276/138004878-bfe17115-c25f-4d22-9740-0fef983507c0.png)

"""
from typing import Mapping, Iterable
from inspect import Signature, signature


def ensure_iterable_of_callables(x):
    if isinstance(x, Iterable):
        if not all(callable(xx) for xx in x):
            non_callables = filter(lambda xx: not callable(xx), x)
            raise TypeError(f'These were not callable: {list(non_callables)}')
        return x
    else:
        assert callable(x)
        return (x,)


def merge_unnamed_and_named(*unnamed, **named):
    """To merge unnamed and named arguments into a single (named) dict of arguments

    >>> merge_unnamed_and_named(10, 20, thirty=30, fourty=40)
    {'_0': 10, '_1': 20, 'thirty': 30, 'fourty': 40}
    """
    # TODO: Could do what is done in meshed (try to get names of actual functions)
    #  and use _0, _1, _2... as fallback only?
    named_unnamed = {f'_{i}': obj for i, obj in enumerate(unnamed)}
    if not named_unnamed.keys().isdisjoint(named):
        raise ValueError(
            f"Some of your objects' names clashed: "
            f'{named_unnamed.keys() & named.keys()}'
        )
    return dict(named_unnamed, **named)


def _multi_func_init(self, *unnamed_funcs, **named_funcs):
    if len(unnamed_funcs) == 1 and isinstance(unnamed_funcs[0], Mapping):
        self.funcs = unnamed_funcs[0]
        expected_n_funcs = len(self.funcs)
    else:
        expected_n_funcs = len(unnamed_funcs) + len(named_funcs)
        self.funcs = merge_unnamed_and_named(*unnamed_funcs, **named_funcs)
    if len(self.funcs) != expected_n_funcs:
        raise ValueError(
            'Some of your func names clashed. Your unnamed funcs were: '
            f'{unnamed_funcs} and your named ones were: {named_funcs}'
        )
    ensure_iterable_of_callables(self.funcs.values())


_dflt_signature = Signature.from_callable(lambda *args, **kwargs: None)


def _signature_from_first_and_last_func(first_func, last_func):
    try:
        input_params = signature(first_func).parameters.values()
    except ValueError:  # function doesn't have a signature, so take default
        input_params = _dflt_signature.parameters.values()
    try:
        return_annotation = signature(last_func).return_annotation
    except ValueError:  # function doesn't have a signature, so take default
        return_annotation = _dflt_signature.return_annotation
    return Signature(input_params, return_annotation=return_annotation)


# TODO: Give it a __name__ and make it more like a "normal" function so it works
#  well when so assumed?
class Pipe:
    """Simple function composition. That is, gives you a callable that implements

    input -> f_1 -> ... -> f_n -> output.

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = Pipe(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5

    You can name functions, but this would just be for documentation purposes.
    The names are completely ignored.

    >>> g = Pipe(
    ...     add_numbers = lambda x, y: x + y,
    ...     multiply_by_2 = lambda x: x * 2,
    ...     stringify = str
    ... )
    >>> g(2, 3)
    '10'

    Notes:
        - Pipe instances don't have a __name__ etc. So some expectations of normal functions are not met.
        - Pipe instance are pickalable (as long as the functions that compose them are)
    """

    def __init__(self, *unnamed_funcs, **named_funcs):
        _multi_func_init(self, *unnamed_funcs, **named_funcs)
        callables = list(self.funcs.values())
        n_funcs = len(callables)
        other_funcs = ()
        if n_funcs == 0:
            raise ValueError('You need to specify at least one function!')
        elif n_funcs == 1:
            first_func = last_func = callables[0]
        else:
            first_func, *other_funcs, last_func = callables

        self.__signature__ = _signature_from_first_and_last_func(first_func, last_func)
        self.first_func = first_func
        self.other_funcs = tuple(other_funcs) + (last_func,)

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out


# TODO: Merge with lined.base.ParallelFuncs
class FuncFanout:
    """Applies multiple functions to the same argument(s) and returns a dict of results.

    You know how `map(func, iterable_of_inputs)` applies a same function to an iterable
    of inputs.
    `FuncFanout` (we could call it `pam`) is a sort of dual; used to apply multiple
    functions to a same input.

    >>> def foo(a):
    ...     return a + 2
    ...
    >>> def bar(a):
    ...     return a * 2
    ...
    >>> def groot(a):
    ...     return 'I am groot'
    ...
    >>> m = FuncFanout(foo, bar, groot)
    >>>
    >>> m(3)
    {'_0': 5, '_1': 6, '_2': 'I am groot'}
    >>>

    If you specify names to the input functions, they'll be used in the dict

    >>> m = FuncFanout(foo, bar_results=bar, groot=groot)
    >>> m(10)
    {'_0': 12, 'bar_results': 20, 'groot': 'I am groot'}

    `FuncFanout` uses the `call_generator` method to iterate through the functions,
    yielding `(func_key, func_output)` pairs on the way.
    The result of calling a `FuncFanout` is simply the gathering of those pairs in
    a `dict`.
    Sometimes you may want/need more control though, and prefer to iterate through the
    pairs yourself, and in that case use `call_generator` directly.

    >>> gen = m.call_generator(10)
    >>> next(gen)
    ('_0', 12)
    >>> next(gen)
    ('bar_results', 20)
    >>> next(gen)
    ('groot', 'I am groot')

    You know how you can do `dict(a=1, b=2)` or `dict({'a': 1, 'b': 2})` with dicts?
    You can do that here too.
    It's useful when the names (keys) of your functions aren't valid python names,
    or not even strings.

    >>> m = FuncFanout({2: foo, "Bar results": bar, "I am groot": groot})
    >>> m(10)
    {2: 12, 'Bar results': 20, 'I am groot': 'I am groot'}

    """

    funcs: dict
    __init__ = _multi_func_init

    def call_generator(self, *args, **kwargs):
        for name, func in self.funcs.items():
            yield name, func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return dict(self.call_generator(*args, **kwargs))


from i2.signatures import ch_func_to_all_pk, tuple_the_args, Sig


# TODO: Finish this!
# TODO: Test the handling var positional and var keyword
class FlexFuncFanout:
    """
    Call multiple functions, using a pool of arguments that they will draw from.

    >>> from i2.tests.objects_for_testing import formula1, sum_of_args, mult, add
    >>> mf1 = FlexFuncFanout(formula1=formula1, mult=mult, add=add)
    >>> kwargs_for_func = mf1.kwargs_for_func(w=1, x=2, z=3, a=4, b=5)

    What's this for? Well, the raison d'etre of `FlexFuncFanout` is to be able to do this:

    >>> assert add(a=4, b=5) == add(**kwargs_for_func['add'])

    This wouldn't work on all functions since some functions have position only arguments (e.g. ``formula1``).
    Therefore ``FlexFuncFanout`` holds a "normalized" form of the functions; namely one that handles such things as
    postion only and varargs.

    # TODO: Make this work!
    #   Right now raises: TypeError: formula1() got some positional-only arguments passed as keyword arguments: 'w'
    # >>> assert formula1(1, x=2, z=3) == mf1.normalized_funcs[formula1](**kwargs_for_func[formula1])

    Note: In the following, it looks like ``FlexFuncFanout`` instances return dicts whose keys are strings.
    This is not the case.
    The keys are functions: The same functions that were input.
    The reason for not using functions is that when printed, they include their hash, which invalidates the doctests.

    # >>> def print_dict(d):  # just a util for this doctest
    # ...     from pprint import pprint
    # ...     pprint({k.__name__: d[k] for k in sorted(d, key=lambda x: x.__name__)})

    >>> mf1 = FlexFuncFanout(formula1, mult=mult, add=add)
    >>> assert mf1.kwargs_for_func(w=1, x=2, z=3, a=4, b=5) == {
    ... '_0': {'w': 1, 'x': 2, 'z': 3},
    ... 'add': {'a': 4, 'b': 5},
    ... 'mult': {'x': 2}
    ... }

    Oh, and you can actually see the signature of kwargs_for_func:

    >>> from inspect import signature
    >>> signature(mf1)
    <Signature (w, x: float, a, y=1, z: int = 1, b: float = 0.0)>

    >>> mf2 = FlexFuncFanout(formula1, mult, add=add, sum_of_args=sum_of_args)
    >>> assert mf2.kwargs_for_func(
    ...     w=1, x=2, z=3, a=4, b=5, args=(7,8), kwargs={'a': 42}, extra_stuff='ignore'
    ... ) == {
    ... '_0': {'w': 1, 'x': 2, 'z': 3},
    ... '_1': {'x': 2},
    ... 'add': {'a': 4, 'b': 5},
    ... 'sum_of_args': {
    ...     'kwargs': {
    ...         'w': 1, 'x': 2, 'z': 3, 'a': 4, 'b': 5,
    ...         'args': (7, 8), 'kwargs': {'a': 42}, 'extra_stuff': 'ignore'
    ...         }
    ...     }
    ... }

    """

    funcs: dict

    # FIXME: TODO: This does indeed change the signature, but not the functionality (position only still raise errors!)
    def normalize_func(self, func):
        return ch_func_to_all_pk(tuple_the_args(func))

    def __init__(self, *unnamed_funcs, **named_funcs):
        _multi_func_init(self, *unnamed_funcs, **named_funcs)
        self.sigs = {name: Sig(func) for name, func in self.funcs.items()}
        self.normalized_funcs = {
            name: self.normalize_func(func) for name, func in self.funcs.items()
        }
        multi_func_sig = Sig.from_objs(*self.normalized_funcs.values())
        # TODO: Finish attempt to add **all_other_kwargs_ignored to the signature
        # multi_func_sig = (Sig.from_objs(
        #     *self.normalized_funcs.values(),
        #     Parameter(name='all_other_kwargs_ignored', kind=Parameter.VAR_KEYWORD)))
        multi_func_sig.wrap(self)
        # multi_func_sig.wrap(self.kwargs_for_func)

    def kwargs_for_func(self, *args, **kwargs):
        return dict(
            (name, self.sigs[name].source_kwargs(**kwargs))
            for name, func in self.funcs.items()
        )

    # TODO: Give it a signature (needs to be done in __init__)
    # TODO: Validation of inputs
    def __call__(self, *args, **kwargs):
        return dict(
            (name, self.sigs[name].source_kwargs(**kwargs))
            for name, func in self.funcs.items()
        )


class ParallelFuncs:
    """Make a multi-channel function from a {name: func, ...} specification.

    >>> multi_func = ParallelFuncs(
    ...     say_hello=lambda x: f"hello {x}", say_goodbye=lambda x: f"goodbye {x}"
    ... )
    >>> multi_func({'say_hello': 'world', 'say_goodbye': 'Lenin'})
    {'say_hello': 'hello world', 'say_goodbye': 'goodbye Lenin'}

    :param spec: A map between a name (str) and a function associated to that name
    :return: A function that takes a dict as an (multi-channel) input and a dict as a
    (multi-channel) output

    Q: Why can I specify the specs both with named_funcs_dict and **named_funcs?
    A: Look at the ``dict(...)`` interface. You see the same thing there.
    Different reason though (here we assert that the keys don't overlap).
    Usually named_funcs is more convenient, but if you need to use keys that are not
    valid python variable names,
    you can always use named_funcs_dict to express that!

    >>> multi_func = ParallelFuncs({
    ...     'x+y': lambda d: f"sum is {d}",
    ...     'x*y': lambda d: f"prod is {d}"}
    ... )
    >>> multi_func({
    ...     'x+y': 5,
    ...     'x*y': 6
    ... })
    {'x+y': 'sum is 5', 'x*y': 'prod is 6'}

    You can also use both. Like with ``dict(...)``.

    Here's a more significant example.

    >>> chunkers = {
    ...     'a': lambda x: x[0] + x[1],
    ...     'b': lambda x: x[0] * x[1]
    ... }
    >>> featurizers = {
    ...     'a': lambda z: str(z),
    ...     'b': lambda z: [z] * 3
    ... }
    >>> multi_chunker = ParallelFuncs(**chunkers)
    >>> multi_chunker({'a': (1, 2), 'b': (3, 4)})
    {'a': 3, 'b': 12}
    >>> multi_featurizer = ParallelFuncs(**featurizers)
    >>> multi_featurizer({'a': 3, 'b': 12})
    {'a': '3', 'b': [12, 12, 12]}
    >>> my_pipe = Pipe(multi_chunker, multi_featurizer)
    >>> my_pipe({'a': (1, 2), 'b': (3, 4)})
    {'a': '3', 'b': [12, 12, 12]}

    #{'a': '(1, 2)', 'b': [(3, 4), (3, 4), (3, 4)]}

    """

    funcs: dict

    __init__ = _multi_func_init

    def _key_output_gen(self, d: dict):
        for key, func in self.funcs.items():
            yield key, func(d[key])

    def __call__(self, d: dict):
        return dict(self._key_output_gen(d))


class ContextFanout:
    """Encapsulates multiple objects into a single context manager that will enter and
    exit all objects that are context managers themselves.

    Context managers show up in situations where you need to have some setup and tear
    down before performing some tasks. It's what you get when you open a file to read
    or write in it, or open a data-base connection, etc.

    Sometimes you need to perform a task that involves more than one context managers,
    or even some objects that may or may not be context managers.
    What `ContextFanout` does for you is allow you to bundle all those (perhaps)
    context managers together, and use them as one single context manager.

    In python 3.10+ you can bundle contexts together by specifying a tuple of context
    managers, as such:

    ```python
    with (open('file.txt'), another_context_manager):
        ...
    ```

    But
    - Python will complain if one of the members of the tuple is not a context manager.
    - A tuple of context managers is not a context manager itself, it's just understood
    by the with (in python 3.10+).

    As an example, let's take two objects. One is a context manager, the other not.

    >>> from contextlib import contextmanager
    >>> @contextmanager
    ... def some_context_manager(x):
    ...     print('open')
    ...     yield f'x + 1 = {x + 1}'
    ...     print('close')
    ...
    >>> def not_a_context_manager(x):
    ...     return x - 1
    ...


    >>> c = ContextFanout(
    ...     some_context_manager=some_context_manager(2),
    ...     not_a_context_manager=not_a_context_manager
    ... )

    See from the prints that "with-ing" c triggers the enter and exit of
    `some_context_manager`:

    >>> with c:
    ...     pass
    open
    close

    Further, know that within (and only within) the context's scope, a `ContextFanout`
    instance will have the context managers it contains available, and having the
    value it is supposed to have "under context".

    >>> c = ContextFanout(
    ...     some_context_manager=some_context_manager(2),
    ...     not_a_context_manager=not_a_context_manager
    ... )
    >>> # first c doesn't have the some_context_manager attribute
    >>> assert not hasattr(c, 'some_context_manager')
    >>> with c:
    ...     # inside the context, c indeed has the attribute, and it has the expected value
    ...     assert c.some_context_manager == 'x + 1 = 3'
    open
    close
    >>> # outside the context, c doesn't have the some_context_manager attribute any more again
    >>> assert not hasattr(c, 'some_context_manager')

    If you don't specify a name for a given context manager, you'll still have access
    to it via a hidden attribute ("_i" where i is the index of the object when
    the `ContextFanout` instance was made.

    >>> c = ContextFanout(some_context_manager(10), not_a_context_manager)
    >>> with c:
    ...     assert c._0 == 'x + 1 = 11'
    open
    close

    """

    def __init__(self, *unnamed_objects, **objects):
        self.objects = merge_unnamed_and_named(*unnamed_objects, **objects)

    def __enter__(self):
        for name, obj in self.objects.items():
            if hasattr(obj, '__enter__'):
                setattr(self, name, obj.__enter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for name, obj in self.objects.items():
            if hasattr(obj, '__exit__'):
                obj.__exit__(exc_type, exc_val, exc_tb)
                delattr(self, name)

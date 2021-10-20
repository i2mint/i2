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
from typing import Mapping, Iterable, Union, Callable, Any, TypeVar
from inspect import Signature, signature


def name_of_obj(o: object) -> Union[str, None]:
    """
    Tries to find the (or "a") name for an object, even if `__name__` doesn't exist.

    >>> name_of_obj(map)
    'map'
    >>> name_of_obj([1, 2, 3])
    'list'
    >>> name_of_obj(print)
    'print'
    >>> name_of_obj(lambda x: x)
    '<lambda>'
    >>> from functools import partial
    >>> name_of_obj(partial(print, sep=","))
    'print'
    """
    if hasattr(o, '__name__'):
        return o.__name__
    elif hasattr(o, '__class__'):
        name = name_of_obj(o.__class__)
        if name == 'partial':
            if hasattr(o, 'func'):
                return name_of_obj(o.func)
        return name
    else:
        return None


def ensure_iterable_of_callables(x):
    """Assert that the input is an iterable of callables,
    or wrap a single callable in an iterable.
    """
    if isinstance(x, Iterable):
        if not all(callable(xx) for xx in x):
            non_callables = filter(lambda xx: not callable(xx), x)
            raise TypeError(f'These were not callable: {list(non_callables)}')
        return x
    else:
        assert callable(x)
        return (x,)


Obj = TypeVar('Obj')


def uniquely_named_objects(
    objects: Iterable[Obj],
    exclude_names: Iterable[str] = (),
    obj_to_name: Callable[[Obj], str] = name_of_obj,
):
    """Generate (name, object) pairs from an iterable of objects

    :param objects: Objects to be named
    :param exclude_names: Names that can't be used
    :param obj_to_name: Function that tries to get/make a name from an object

    >>> from functools import partial
    >>> objects = [map, [1], [1, 2], lambda x: x, partial(print, sep=",")]
    >>> g = uniquely_named_objects(objects)
    >>> names_and_objects = dict(g)
    >>> list(names_and_objects)
    ['map', 'list', '_2', 'lambda_3', 'print']

    That '_2' is there because both [1] and [1, 2] would be named `'list'`, so to avoid
    that, a default name (revealing the position of the object in the input `objects`)
    is given.

    If we wanted to not allow the function to choose 'lambda_3' as a name, we can do
    so with the `exclude_names` argument;

    >>> list(dict(uniquely_named_objects(objects, exclude_names={'lambda_3'})))
    ['map', 'list', '_2', '_3', 'print']

    """
    _exclude_names = set(exclude_names)
    for i, obj in enumerate(objects):
        name = obj_to_name(obj)
        if name == '<lambda>':
            name = f'lambda_{i}'
        if name is None or name in _exclude_names:
            name = f'_{i}'
            assert (
                name not in _exclude_names
            ), '{name} already used in {_exclude_names}!'
        yield name, obj
        _exclude_names.add(name)


class MultiObj:
    """A base class that holds several named objects

    >>> from functools import partial

    Let's make a `MultiObj` with some miscellaneous objects.
    (Note that `MultiObj` will usually be used for specific kinds of objects such as
    callables or context managers. Here we chose the objects to demo what the
    auto-naming does.)

    >>> mo = MultiObj([1], [1, 2], partial(print, sep=","), i='hi', ident=lambda x: x)

    A MultiObj will give provide you with a `.objects` attribute that will give you
    access to the objects you entered, with the names you specified, or the names
    it decided for you, through it's `auto_namer` staticmethod (which you can
    overwrite if necessary).

    To see the names `.objects` is using, do:

    >>> list(mo.objects)
    ['list', '_1', 'print', 'i', 'ident']

    Or do what ever you do with `dicts`;

    >>> mo.objects['_1']
    [1, 2]
    >>> 'list' in mo.objects
    True
    >>> 'not a key of mo' in mo.objects
    False

    When a key (always a string) is also a valid identifier, and in-so-far as
    it doesn't clash with other attributes, `MultiObj` will also give
    you access to the names/keys of your objects via attributes.
    (Note, this is similar to what `pandas.DataFrame` does with it's columns names.)

    >>> mo.list
    [1]
    >>> mo.print
    functools.partial(<built-in function print>, sep=',')

    A `MultiObj` is `Sizable`:

    >>> len(mo)
    5

    and `Iterable`, but be aware that iterating over
    an `MultiObj` will not give you the keys of `.objects`, but the `.values()`.
    Again, that's:

    >>> assert list(mo.objects) != list(mo)

    This is to enable to get the objects as such:

    >>> a, b, c, d, e = mo
    >>> a
    [1]
    >>> b
    [1, 2]

    You can also specify an object mapping directly through a mapping:

    >>> mo = MultiObj({'this': [1], 'that': [1, 2]})
    >>> mo.objects
    {'this': [1], 'that': [1, 2]}
    """

    # Placing as staticmethod so that subclasses can overwrite if necessary
    auto_namer = staticmethod(uniquely_named_objects)

    def __init__(self, *unnamed, **named):
        if len(unnamed) == 1 and isinstance(unnamed[0], Mapping) and len(named) == 0:
            # Special case where a single input is given: a mapping between names and obj
            self.objects = unnamed[0]
        else:
            # Normal case: Some named and unnamed objects are given,
            # so we need to get unique names for the unnamed
            named_unnamed = dict(self.auto_namer(unnamed, exclude_names=set(named)))
            # The uniquely_name_objects should take care of this, but extra precaution:
            if not named_unnamed.keys().isdisjoint(named):
                raise ValueError(
                    f"Some of your objects' names clashed: "
                    f'{named_unnamed.keys() & named.keys()}'
                )
            # Merge these
            self.objects = dict(named_unnamed, **named)

    def __iter__(self):
        yield from self.objects.values()

    def __len__(self):
        return len(self.objects)

    def __getattr__(self, item):
        """Access to those objects that have proper identifier names"""
        if item in self.objects and item.isidentifier():
            return self.objects[item]
        else:
            raise AttributeError(f'Not an attribute: {item}')


def iterable_of_callables_validation(funcs: Iterable[Callable]):
    """Validates that the input is an iterable of callables"""
    if not isinstance(funcs, Iterable):
        raise TypeError(f'Not an iterable: {funcs}')
    elif not all(callable(xx) for xx in funcs):
        non_callables = filter(lambda f: not callable(f), funcs)
        raise TypeError(f'These were not callable: {list(non_callables)}')


class MultiFunc(MultiObj):
    """A MultiObj, but specialized to contain callable objects only"""

    def __init__(self, *unnamed_funcs, **named_funcs):
        # The basic MultiObj initialization
        super().__init__(*unnamed_funcs, **named_funcs)
        # The extra stuff we want to do with functions
        self.funcs = self.objects  # an alias, for better readability
        iterable_of_callables_validation(self)


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
class Pipe(MultiFunc):
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
        # The basic MultiFunc initialization
        super().__init__(*unnamed_funcs, **named_funcs)

        # The extra initialization for pipelines
        callables = list(self)
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


class FuncFanout(MultiFunc):
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
    >>> groot = lambda a: 'I am groot'
    >>> m = FuncFanout(foo, bar, groot)
    >>>
    >>> m(3)
    {'foo': 5, 'bar': 6, 'lambda_2': 'I am groot'}

    Don't like that `lambda_2`?
    Well, If you specify names to the input functions, they'll be used instead of the
    ones found by the `MultObj.auto_namer`.

    >>> m = FuncFanout(foo, bar_results=bar, groot=groot)
    >>> m(10)
    {'foo': 12, 'bar_results': 20, 'groot': 'I am groot'}

    `FuncFanout` uses the `call_generator` method to iterate through the functions,
    yielding `(func_key, func_output)` pairs on the way.
    The result of calling a `FuncFanout` is simply the gathering of those pairs in
    a `dict`.
    Sometimes you may want/need more control though, and prefer to iterate through the
    pairs yourself, and in that case use `call_generator` directly.

    >>> gen = m.call_generator(10)
    >>> next(gen)
    ('foo', 12)
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

    def call_generator(self, *args, **kwargs):
        for name, func in self.funcs.items():
            yield name, func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return dict(self.call_generator(*args, **kwargs))


from i2.signatures import ch_func_to_all_pk, tuple_the_args, Sig


# TODO: Finish this!
# TODO: Test the handling var positional and var keyword
class FlexFuncFanout(MultiFunc):
    """
    Call multiple functions, using a pool of arguments that they will draw from.

    >>> from i2.tests.objects_for_testing import formula1, sum_of_args, mult, add
    >>> mf1 = FlexFuncFanout(formula1=formula1, mult=mult, add=add)
    >>> kwargs_for_func = mf1.kwargs_for_func(w=1, x=2, z=3, a=4, b=5)

    What's this for? Well, the raison d'etre of `FlexFuncFanout` is to be able to do this:

    >>> assert add(a=4, b=5) == add(**kwargs_for_func['add'])

    This wouldn't work on all functions since some functions have position only
    arguments (e.g. ``formula1``).
    Therefore ``FlexFuncFanout`` holds a "normalized" form of the functions;
    namely one that handles such things as postion only and varargs.

    # TODO: Make this work!
    #   Right now raises: TypeError: formula1() got some positional-only arguments
    # passed as keyword arguments: 'w'
    # >>> assert formula1(1, x=2, z=3) == mf1.normalized_funcs[formula1](**kwargs_for_func[formula1])

    Note: In the following, it looks like ``FlexFuncFanout`` instances return dicts
    whose keys are strings.
    This is not the case.
    The keys are functions: The same functions that were input.
    The reason for not using functions is that when printed, they include their hash,
    which invalidates the doctests.

    # >>> def print_dict(d):  # just a util for this doctest
    # ...     from pprint import pprint
    # ...     pprint({k.__name__: d[k] for k in sorted(d, key=lambda x: x.__name__)})

    >>> mf1 = FlexFuncFanout(formula1, mult=mult, addition=add)
    >>> assert mf1.kwargs_for_func(w=1, x=2, z=3, a=4, b=5) == {
    ... 'formula1': {'w': 1, 'x': 2, 'z': 3},
    ... 'mult': {'x': 2},
    ... 'addition': {'a': 4, 'b': 5},
    ... }

    Oh, and you can actually see the signature of kwargs_for_func:

    >>> from inspect import signature
    >>> signature(mf1)
    <Signature (w, x: float, a, y=1, z: int = 1, b: float = 0.0)>

    >>> mf2 = FlexFuncFanout(formula1, mult, addition=add, mysum=sum_of_args)
    >>> assert mf2.kwargs_for_func(
    ...     w=1, x=2, z=3, a=4, b=5, args=(7,8), kwargs={'a': 42}, extra_stuff='ignore'
    ... ) == {
    ... 'formula1': {'w': 1, 'x': 2, 'z': 3},
    ... 'mult': {'x': 2},
    ... 'addition': {'a': 4, 'b': 5},
    ... 'mysum': {
    ...     'kwargs': {
    ...         'w': 1, 'x': 2, 'z': 3, 'a': 4, 'b': 5,
    ...         'args': (7, 8), 'kwargs': {'a': 42}, 'extra_stuff': 'ignore'
    ...         }
    ...     }
    ... }

    """

    # FIXME: TODO: This does indeed change the signature, but not the functionality (position only still raise errors!)
    def normalize_func(self, func):
        return ch_func_to_all_pk(tuple_the_args(func))

    def __init__(self, *unnamed_funcs, **named_funcs):
        # The basic MultiFunc initialization
        super().__init__(*unnamed_funcs, **named_funcs)

        # The extra initialization for FlexFuncFanouts
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


class ParallelFuncs(MultiFunc):
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

    def _key_output_gen(self, d: dict):
        for key, func in self.funcs.items():
            yield key, func(d[key])

    def __call__(self, d: dict):
        return dict(self._key_output_gen(d))


class ContextFanout(MultiObj):
    """Encapsulates multiple objects into a single context manager that will enter and
    exit all objects that are context managers themselves (and just leave those
    that are not context managers alone).

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

    `ContextFanout` is therefore useful when you need some resources to perform a task,
    but don't know in advance (or don't want to assume) which resources might need to
    be used within a context.

    Consider the following code that reads off af a stream, computes some stats from
    what it read, then "writes" those stats in a target stream:

    ```
    with ContextFanout(source_stream, target_stream) as (src, target):
        for b in src:
            stats = compute_statistics(b)
            target.append(stats)
    ```

    The source and/or target streams could need to open and close files or data base
    connections, or could simply be lists.
    Nothing in the code reveals (or should reveal) what they are.
    Yet you'll get some complaints if you try to do a `with` on a list, or iterate over
    `src` outside of a context manager.

    `ContextFanout` allows you to move that concern out of the business logic code.

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


    >>> c = ContextFanout(not_a_context_manager, some_context_manager(2))
    >>> list(c.objects)
    ['not_a_context_manager', '_GeneratorContextManager']

    The name (chosen by `MultiObj.auto_namer`) '_GeneratorContextManager' isn't the best.
    Let's give an explicit name:

    >>> c = ContextFanout(not_a_context_manager, context=some_context_manager(2))
    >>> list(c.objects)
    ['not_a_context_manager', 'context']

    See from the prints that "with-ing" c triggers the enter and exit of 'context'

    >>> with c:
    ...     print(c.not_a_context_manager(10))
    open
    9
    close

    Further, know that within the context's scope, a `ContextFanout` instance
    (because it's a `MultiObj`)
    will have the context managers it contains available, and having the
    value it is supposed to have "under context".


    >>> with ContextFanout(not_a_context_manager, some_context_manager(2)) as context:
    ...     print(context.not_a_context_manager(10))
    open
    9
    close

    You can also use the `with CONTEXT() as (x, y, z):` pattern to assign the
    objects of `ContextFanout` to some variables of your choice.

    IMPORTANT: Remember to include PARENTHESES around these variables, or it won't work!

    >>> with ContextFanout(not_a_context_manager, some_context_manager(2)) as (f, m):
    ...     print(f(10))
    open
    9
    close

    """

    def __enter__(self):
        for obj in self:
            if hasattr(obj, '__enter__'):
                obj.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for obj in self:
            if hasattr(obj, '__exit__'):
                obj.__exit__(exc_type, exc_val, exc_tb)

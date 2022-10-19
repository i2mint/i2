"""A few fundamental tools to operate on a fixed collection of objects (e.g. functions).

For functions you have:

- ``Pipe``: To compose functions (output of one fed as the input of the next)

- ``FuncFanout``: To apply multiple functions to the same inputs

- ``FlexFuncFanout``: Like `FuncFanout` but where the application of inputs is flexible.
That is, the functions "draw" their inputs from the a common pool, but don't choke
if there are extra unrecognized arguments.

- ``ParallelFuncs``: To make a dict-to-dict function, applying a specific function for
each input key (putting the result in that key in the output.

For context managers you have:

- ``ContextFanout``: To hold multiple context managers as one (entering and exiting
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
    *,
    name_for_position: dict = (),
):
    """Generate (name, object) pairs from an iterable of objects

    :param objects: Objects to be named
    :param exclude_names: Names that can't be used
    :param obj_to_name: Function that tries to get/make a name from an object
    :param name_for_position: A ``{position_idx: name,...}`` mapping that instructs
    ``uniquely_named_objects`` to use a specific name for a given position.

    >>> from functools import partial
    >>> objects = [map, [1], [1, 2], lambda x: x, partial(print, sep=",")]
    >>> g = uniquely_named_objects(objects)
    >>> names_and_objects = dict(g)
    >>> list(names_and_objects)
    ['map', 'list', '_2', '_3', 'print']

    That ``'_2'`` is there because both ``[1]`` and ``[1, 2]`` would be named `'list'`,
    so to avoid that, a default name (revealing the position of the object in the
    input `objects`) is given.
    The '_3' comes from the fact that the ``lambda`` function doesn't have a proper
    name (one that is a python identifier).

    If we wanted the name for ``[1]`` to revert to the default positional name ``'_1'``,
    we can achieve this by forbidding the name ``'list'``:

    >>> list(dict(uniquely_named_objects(objects, exclude_names={'list'})))
    ['map', '_1', '_2', '_3', 'print']

    You could also acheive this by specifying this ``'_1'`` explicitly in the
    ``name_for_position`` argument:

    >>> list(dict(uniquely_named_objects(objects, name_for_position={1: '_1'})))
    ['map', '_1', 'list', '_3', 'print']

    The reason this `list` reappears as a name is that we didn't exclude it, and
    the name is not taken by the ``[1]`` argument anymore.
    To get the desired effect with ``name_for_position`` we could therefore do this:

    >>> list(dict(uniquely_named_objects(objects, name_for_position={1: '_1', 2: '_2'})))
    ['map', '_1', '_2', '_3', 'print']

    Obviously, ``exclude_names`` is the right argument for the problem above, but
    what ``name_for_position`` does give you is the ability to explicitly chose the
    names you want to assign to all or some of the elements of your iterable.

    >>> list(dict(uniquely_named_objects(
    ...     objects, name_for_position={1: 'first_list', 2: 'second_list', 3: 'lambda'}))
    ... )
    ['map', 'first_list', 'second_list', 'lambda', 'print']

    Extra notes:

    See what ``uniquely_named_objects`` offers as parametrization:

    - You can provide an exclusion list (though the handing of a conflict is hardcoded
    and questionable)

    - You can provide a ``obj_to_name`` function to control the naming of objects.
    One trick to be aware of if objects have unique hashes: Make a
    ``d = {obj: name,...}`` mapping and specify ``obj_to_name=d.get``.

    - Any controllable way to decide on a name based on the position of the function
    in the iterable (this could be useful!)

    What you DO NOT have:

    - Any way to choose names non-myopically: An object’s name cannot “see” the
    objects around it to decide on a name (it can only see the names use by those behind
    it through ``exclude_names``).

    - Any “retries” or “alternative naming logic” if a chosen name conflicts with
    ``exclude_names``
    """
    _exclude_names = set(exclude_names)
    for i, obj in enumerate(objects):
        if i in name_for_position:
            name = name_for_position[i]
        else:
            name = obj_to_name(obj)
        # if name == '<lambda>':
        # name = f'lambda_{i}'
        if name is None or not name.isidentifier() or name in _exclude_names:
            name = f'_{i}'
            assert (
                name not in _exclude_names
            ), '{name} already used in {_exclude_names}!'
        yield name, obj
        _exclude_names.add(name)


def merge_unnamed_and_named(*unnamed, **named) -> dict:
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


# TODO: Refactoring: Remove once not used anymore
def _multi_func_init(self, *unnamed_funcs, **named_funcs) -> None:
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


class MultiObj(Mapping):
    """A base class that holds several named objects

    >>> from functools import partial

    Let's make a `MultiObj` with some miscellaneous objects.
    (Note that `MultiObj` will usually be used for specific kinds of objects such as
    callables or context managers. Here we chose the objects to demo what the
    auto-naming does.)

    >>> mo = MultiObj([1], [1, 2], partial(print, sep=","), i='hi', ident=lambda x: x)

    You now have a mapping and can do mapping things such as being able to list keys,
    getting the length, seeing if a key is present, and getting the value for a key.

    Note that the first and second list cannot be assigned the same name without creating a conflict.
    In general one of the following will happen:
    -you give it a name
    -it tries to figure out a non conflicting name (if the object has a dunder name, etc)
    -it falls back to a naming that is just the stringification of the argument's positional index

    >>> list(mo) # not that the second item cannot be 'list', so a different name is given
    ['list', '_1', 'print', 'i', 'ident']
    >>> len(mo)
    5
    >>> mo['_1']
    [1, 2]
    >>> 'list' in mo
    True
    >>> 'not a key of mo' in mo
    False

    When a key (always a string) is also a valid identifier, and in-so-far as
    it doesn't clash with other attributes, `MultiObj` will also give
    you access to the names/keys of your objects via attributes.
    (Note, this is similar to what `pandas.DataFrame` does with it's columns names.)

    >>> mo.list
    [1]
    >>> mo.print
    functools.partial(<built-in function print>, sep=',')

    You can also specify an object mapping directly through a mapping:

    >>> mo = MultiObj({'this': [1], 'that': [1, 2]})
    >>> dict(mo)
    {'this': [1], 'that': [1, 2]}

    You can specify an instance name and/or doc with the special (reserved) argument
    names ``__name__`` and ``__doc__`` (which therefore can't be used as object names:

    >>> mo = MultiObj(
    ... this=[1], that=[1, 2], __name__='this_and_that', __doc__='Nothing much'
    ... )
    >>> dict(mo)
    {'this': [1], 'that': [1, 2]}
    >>> mo.__name__
    'this_and_that'
    >>> mo.__doc__
    'Nothing much'
    """

    # Placing as staticmethod so that subclasses can overwrite if necessary
    auto_namer = staticmethod(uniquely_named_objects)

    def __init__(self, *unnamed, **named):
        named = self._process_reserved_names(named)

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

    _reserved_names = ('__name__', '__doc__')

    def _process_reserved_names(self, named_funcs):
        for name in self._reserved_names:
            if (value := named_funcs.pop(name, None)) is not None:
                setattr(self, name, value)
        return named_funcs

    def __iter__(self):
        yield from self.objects

    def __getitem__(self, k):
        return self.objects[k]

    def __contains__(self, k):
        return k in self.objects

    def __len__(self):
        return len(self.objects)

    def __getattr__(self, item):
        """Access to those objects that have proper identifier names"""
        if item in self and item.isidentifier():
            return self.objects[item]
        else:
            raise AttributeError(f'Not an attribute: {item}')

    def __repr__(self):
        return f"<{type(self).__name__} containing: {', '.join(self)}>"

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


def iterable_of_callables_validation(funcs: Iterable[Callable]):
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
        iterable_of_callables_validation(self.funcs.values())

    @property
    def funcs(self):
        """Alias of .objects, for better readability in the context of MultiFunc"""
        return self.objects


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

    You can specify a single functions:

    >>> Pipe(lambda x: x + 1)(1)
    2

    but

    >>> Pipe()
    Traceback (most recent call last):
      ...
    ValueError: You need to specify at least one function!

    You can specify an instance name and/or doc with the special (reserved) argument
    names ``__name__`` and ``__doc__`` (which therefore can't be used as function names):

    >>> f = Pipe(map, add_it=sum, __name__='map_and_sum', __doc__='Apply func and add')
    >>> f(lambda x: x * 10, [1, 2, 3])
    60
    >>> f.__name__
    'map_and_sum'
    >>> f.__doc__
    'Apply func and add'
    """

    def __init__(self, *unnamed_funcs, **named_funcs):
        # The basic MultiFunc initialization
        super().__init__(*unnamed_funcs, **named_funcs)

        # The extra initialization for pipelines
        callables = list(self.funcs.values())
        n_funcs = len(callables)
        if n_funcs == 0:
            raise ValueError('You need to specify at least one function!')

        elif n_funcs == 1:
            other_funcs = ()
            first_func = last_func = callables[0]
        else:
            first_func, *other_funcs = callables
            *_, last_func = other_funcs

        self.__signature__ = _signature_from_first_and_last_func(first_func, last_func)
        self.first_func = first_func
        self.other_funcs = other_funcs

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out

    def __len__(self):
        return len(self.funcs)

    def __eq__(self, other):
        return pipes_are_equal(self, other)


from operator import eq


def _flatten_pipe(pipe):
    for func in pipe.funcs.values():
        if isinstance(func, Pipe):
            yield from _flatten_pipe(func)
        else:
            yield func


def flatten_pipe(pipe):
    """Unravel nested Pipes to get a flat 'sequence of functions' version of input.

    >>> def f(x): return x + 1
    >>> def g(x): return x * 2
    >>> def h(x): return x - 3
    >>> a = Pipe(f, g, h)
    >>> b = Pipe(f, Pipe(g, h))
    >>> len(a)
    3
    >>> len(b)
    2
    >>> c = flatten_pipe(b)
    >>> len(c)
    3
    >>> assert a(10) == b(10) == c(10) == 19
    """
    return Pipe(*_flatten_pipe(pipe))


def pipes_are_equal(p1, p2, *, func_equality=eq, verbose=False):
    """Determine if two pipelines are equal.

    Pipelines are equal if their flattened versions have equal functions.
    Function equality can be controlled by the ``func_equality`` argument.
    The ``verbose`` argument will print some more information about why the pipelines
    are not equal.

    >>> def f(x): return x + 1
    >>> def g(x): return x * 2
    >>> def h(x): return x - 3
    >>> a = Pipe(f, g, h)
    >>> b = Pipe(f, g, h)
    >>> c = Pipe(f, Pipe(g, h))
    >>> assert a(10) == b(10) == c(10) == 19
    >>> pipes_are_equal(a, b)
    True
    >>> pipes_are_equal(a, c)
    True
    >>> pipes_are_equal(Pipe(f, g), Pipe(g, h))
    False
    >>> pipes_are_equal(Pipe(f, g), Pipe(f, g, h))
    False

    Get more information when pipes are not equal.

    >>> pipes_are_equal(Pipe(f, g), Pipe(f, g, h), verbose=True)
    --> Flattened pipes do not have the same number of functions: len(p1)=2 != len(p2)=3
    False

    Change how functions are compared for equality:

    >>> pipes_are_equal(Pipe(lambda x: x), Pipe(lambda x: x))
    False
    >>> from inspect import getsource
    >>> source_equality = lambda f, ff: getsource(f) == getsource(ff)
    >>> pipes_are_equal(
    ...     Pipe(lambda x: x), Pipe(lambda x: x), func_equality=source_equality
    ... )
    True

    """
    p1, p2 = map(flatten_pipe, (p1, p2))
    if len(p1) != len(p2):
        if verbose:
            print(
                f'--> Flattened pipes do not have the same number of functions:'
                f' {len(p1)=} != {len(p2)=}'
            )
        return False  # if not same number of keys or keys different, not equality
    else:
        for func1, func2 in zip(p1.funcs.values(), p2.funcs.values()):
            if not func_equality(func1, func2):
                if verbose:
                    print(f'--> func1 and func2 are not equal: {func1=} != {func2=}')
                return False  # these two funcs are not equal
            # else continue
    return True


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
    >>> list(m(3))
    [('foo', 5), ('bar', 6), ('_2', 'I am groot')]
    >>> dict(m(3))
    {'foo': 5, 'bar': 6, '_2': 'I am groot'}

    Don't like that `_2` name?
    Well, If you specify names to the input functions, they'll be used instead of the
    ones found by the `MultObj.auto_namer`.

    >>> m = FuncFanout(foo, bar_results=bar, groot=groot)
    >>> dict(m(10))
    {'foo': 12, 'bar_results': 20, 'groot': 'I am groot'}

    Or if you want your results as a tuple, you could do:

    >>> tuple(dict(m(10)).values())
    (12, 20, 'I am groot')

    The above, gather in a ``dict`` is one way to get your data, but what calling a
    ``FuncFanout`` instance actually gives you is a generator that yields the
    ``(func_key, func_output)`` pairs one at a time

    Sometimes you may want/need more control though, and prefer to iterate through the
    pairs yourself, and in that case use `call_generator` directly.

    >>> gen = m(10)
    >>> next(gen)
    ('foo', 12)
    >>> next(gen)
    ('bar_results', 20)
    >>> next(gen)
    ('groot', 'I am groot')

    So this gives you control on how you want your data.
    Here's a recipe: Say you want to make a function that gives you the data as a dict
    automatically. You can do this, using ``i2.Pipe``:

    >>> f = Pipe(m, dict)
    >>> f(10)
    {'foo': 12, 'bar_results': 20, 'groot': 'I am groot'}

    Or if you want a tuple:

    >>> from operator import itemgetter, methodcaller
    >>> from functools import partial
    >>> f = Pipe(m, partial(map, itemgetter(1)), tuple)
    >>> f(10)
    (12, 20, 'I am groot')

    """

    def call_generator(self, *args, **kwargs):
        for name, func in self.funcs.items():
            yield name, func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call_generator(*args, **kwargs)


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

    >>> mf1 = FlexFuncFanout(formula1, mult=mult, addition=add)
    >>> assert mf1.kwargs_for_func(w=1, x=2, z=3, a=4, b=5) == {
    ... 'formula1': {'w': 1, 'x': 2, 'z': 3},
    ... 'mult': {'x': 2},
    ... 'addition': {'a': 4, 'b': 5},
    ... }

    Oh, and you can actually see the signature of kwargs_for_func:

    >>> from inspect import signature
    >>> signature(mf1)
    <Sig (w, x: float, a, y=1, z: int = 1, b: float = 0.0)>

    >>> mf2 = FlexFuncFanout(formula1, mult, addition=add, mysum=sum_of_args)
    >>> assert mf2.kwargs_for_func(
    ...     w=1, x=2, z=3, a=4, b=5, args=(7,8), kwargs={'a': 42}, extra_stuff='ignore'
    ... ) == {
    ... 'formula1': {'w': 1, 'x': 2, 'z': 3},
    ... 'mult': {'x': 2},
    ... 'addition': {'a': 4, 'b': 5},
    ... 'mysum':
    ...     {'kwargs': {'w': 1,
    ...                 'x': 2,
    ...                 'z': 3,
    ...                 'a': 42,
    ...                 'b': 5,
    ...                 'args': (7, 8),
    ...                 'extra_stuff': 'ignore'}}}

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


# from collections.abc import ValuesView as BaseValuesView
# class MongoValuesView(ValuesView):
#     def __contains__(self, v):
#         m = self._mapping
#         cursor = m.mgc.find(filter=m._merge_with_filt(v), projection=())
#         return next(cursor, end_of_cursor) is not end_of_cursor
#
#     def __iter__(self):
#         m = self._mapping
#         return m.mgc.find(filter=m.filter, projection=m._getitem_projection)


class ContextFanout(MultiObj):
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


    >>> c = ContextFanout(not_a_context_manager, some_context_manager(2))
    >>> list(c)
    ['not_a_context_manager', '_GeneratorContextManager']

    The name (chosen by `MultiObj.auto_namer`) '_GeneratorContextManager' isn't the best.
    Let's give an explicit name:

    >>> c = ContextFanout(not_a_context_manager, context=some_context_manager(2))
    >>> list(c.objects)
    ['not_a_context_manager', 'context']

    See from the prints that "with-ing" c triggers the enter and exit of 'context'

    >>> with c:
    ...     pass
    open
    close

    # Further, know that within the context's scope, a `ContextFanout`
    # instance will have the context managers it contains available, and having the
    # value it is supposed to have "under context".
    #
    # >>> with ContextFanout(not_a_context_manager, some_context_manager(2)) as (f, m):
    # ...     print(f(10))
    #
    #
    # >>> c = ContextFanout(not_a_context_manager, context=some_context_manager(2))
    #
    # >>> c = ContextFanout(
    # ...     some_context_manager=some_context_manager(2),
    # ...     not_a_context_manager=not_a_context_manager
    # ... )
    # >>> # first c doesn't have the some_context_manager attribute
    # >>> assert not hasattr(c, 'some_context_manager')
    # >>> with c:
    # ...     # inside the context, c indeed has the attribute, and it has the expected value
    # ...     assert c.some_context_manager == 'x + 1 = 3'
    # open
    # close
    # >>> # outside the context, c doesn't have the some_context_manager attribute any more again
    # >>> assert not hasattr(c, 'some_context_manager')
    #
    # If you don't specify a name for a given context manager, you'll still have access
    # to it via a hidden attribute ("_i" where i is the index of the object when
    # the `ContextFanout` instance was made.
    #
    # >>> c = ContextFanout(some_context_manager(10), not_a_context_manager)
    # >>> with c:
    # ...     assert c._0 == 'x + 1 = 11'
    # open
    # close

    """

    def __enter__(self):
        for name, obj in self.items():
            if hasattr(obj, '__enter__'):
                obj.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for name, obj in self.items():
            if hasattr(obj, '__exit__'):
                obj.__exit__(exc_type, exc_val, exc_tb)

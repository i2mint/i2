"""Testing wrapper"""

from i2.wrapper import wrap, mk_ingress_from_name_mapper


def _test_ingress(a, b: str, c='hi'):
    return (a + len(b) % 2,), dict(string=f'{c} {b}')


def _test_func(times, string):
    return times * string


def test_wrap():
    import pickle
    from inspect import signature

    func = _test_func

    # Just wrapping the func gives you a sort of copy of the func.
    wrapped_func = wrap(func)  # no transformations
    assert wrapped_func(2, 'co') == 'coco' == func(2, 'co')

    # If you give the wrap an ingress function
    ingress = _test_ingress
    wrapped_func = wrap(func, ingress=ingress)
    # it will use it to (1) determine the signature of the wrapped_func
    assert (
        str(signature(wrapped_func)) == "(a, b: str, c='hi')"
    )  # "(a, b: str, c='hi')"
    # and (2) to map inputs
    assert wrapped_func(2, 'world! ', 'Hi') == 'Hi world! Hi world! Hi world! '

    # An egress function can be used to transform outputs
    wrapped_func = wrap(func, egress=len)
    assert wrapped_func(2, 'co') == 4 == len('coco') == len(func(2, 'co'))

    # Both ingress and egress can be used in combination
    wrapped_func = wrap(func, ingress=ingress, egress=len)
    assert (
        wrapped_func(2, 'world! ', 'Hi') == 30 == len('Hi world! Hi world! Hi world! ')
    )

    # A wrapped function is pickle-able (unlike the usual way decorators are written)

    unpickled_wrapped_func = pickle.loads(pickle.dumps(wrapped_func))
    assert (
        unpickled_wrapped_func(2, 'world! ', 'Hi')
        == 30
        == len('Hi world! Hi world! Hi world! ')
    )


def _test_foo(a, b: int, c=7):
    return a + b * c


def _test_bar(a, /, b: int, *, c=7):
    return a + b * c


def test_mk_ingress_from_name_mapper():
    import pickle
    from inspect import signature

    foo = _test_foo
    # Define the mapping (keys are inner and values are outer names)
    name_mapper = dict(a='aa', c='cc')
    # Make an ingress function with that mapping
    ingress = mk_ingress_from_name_mapper(foo, name_mapper)
    # Use the ingress function to wrap a function
    wrapped_foo = wrap(foo, ingress=ingress)
    # See that the signature of the wrapped func uses the mapped arg names
    assert (
        str(signature(wrapped_foo)) == str(signature(ingress)) == '(aa, b: int, cc=7)'
    )
    # And that wrapped function does compute correctly
    assert (
        foo(1, 2, c=4)
        == foo(a=1, b=2, c=4)
        == wrapped_foo(aa=1, b=2, cc=4)
        == wrapped_foo(1, 2, cc=4)
    )
    # The ingress function returns args and kwargs for wrapped function
    assert ingress('i was called aa', b='i am b', cc=42) == (
        (),
        {'a': 'i was called aa', 'b': 'i am b', 'c': 42},
    )
    # See above that the args is empty. That will be the case most of the time.
    # Keyword arguments will be favored when there's a choice. If wrapped
    # function uses position-only arguments though, ingress will have to use them
    bar = _test_bar
    assert str(signature(bar)) == '(a, /, b: int, *, c=7)'
    ingress_for_bar = mk_ingress_from_name_mapper(bar, name_mapper)
    assert ingress_for_bar('i was called aa', b='i am b', cc=42) == (
        ('i was called aa',),
        {'b': 'i am b', 'c': 42},
    )
    wrapped_bar = wrap(bar, ingress=ingress_for_bar)
    assert (
        bar(1, 2, c=4)
        == bar(1, b=2, c=4)
        == wrapped_bar(1, b=2, cc=4)
        == wrapped_bar(1, 2, cc=4)
    )

    # Note that though bar had a positional only and a keyword only argument,
    # we are (by default) free of argument kind constraints in the wrapped function:
    # We can can use a positional args on `cc` and keyword args on `aa`
    assert str(signature(wrapped_bar)) == '(aa, b: int, cc=7)'
    assert wrapped_bar(1, 2, 4) == wrapped_bar(aa=1, b=2, cc=4)

    # If you want to conserve the argument kinds of the wrapped function, you can
    # specify this with `conserve_kind=True`:
    ingress_for_bar = mk_ingress_from_name_mapper(bar, name_mapper, conserve_kind=True)
    wrapped_bar = wrap(bar, ingress=ingress_for_bar)
    assert str(signature(wrapped_bar)) == '(aa, /, b: int, *, cc=7)'

    # A wrapped function is pickle-able (unlike the usual way decorators are written)
    unpickled_wrapped_foo = pickle.loads(pickle.dumps(wrapped_foo))
    assert (
        str(signature(unpickled_wrapped_foo))
        == str(signature(ingress))
        == '(aa, b: int, cc=7)'
    )
    assert (
        foo(1, 2, c=4)
        == foo(a=1, b=2, c=4)
        == unpickled_wrapped_foo(aa=1, b=2, cc=4)
        == unpickled_wrapped_foo(1, 2, cc=4)
    )


def test_arg_val_converter():
    from i2.wrapper import arg_val_converter
    from i2.tests.objects_for_testing import formula1, times_2, plus_1

    assert formula1(4, 3, 2, z=1) == 14
    assert formula1(4, 3 * 2, 2, z=1 + 1) == 400

    # See that "transparently" converting the function doesn't change anything

    formula2 = arg_val_converter(formula1)
    assert formula2(4, 3, 2, z=1) == 14 == formula1(4, 3, 2, z=1)

    # But now if we ask to convert x and z...

    formula2 = arg_val_converter(formula1, x=times_2, z=plus_1)
    assert formula2(4, 3, 2, z=1) == 400 == formula1(4, 3 * 2, 2, z=1 + 1)

    from i2.tests.test_util import unpickled_func_still_works

    assert unpickled_func_still_works(formula1, 4, 3, 2, z=1)
    assert unpickled_func_still_works(formula2, 4, 3, 2, z=1)


from i2.wrapper import Wrapx


def test_wrapx():
    from inspect import signature

    # Test that an trivial Wrapx instance (i.e. no ingress, caller, or egress
    # modifications) gives us an object behaving like the wrapped function.

    # TODO: Make it work with param kinds: e.g. func(x: int, *, y=1) -> int:
    def func(x: int, y=1) -> int:
        return x + y

    wrapped_func = Wrapx(func)
    assert (
        str(signature(wrapped_func)) == '(x: int, y=1) -> int' == str(signature(func))
    )

    # Test egress that has a single param z

    def func(x, y):
        return x + y

    def egress(v, *, z):
        return v * z

    wrapped_func = Wrapx(func, egress=egress)

    assert func(1, 2) == 3

    # TODO: should be '(x, y=1, *, z)' --> Need to work on the merge for this.
    assert str(signature(wrapped_func)) == '(x, y, z)'
    assert wrapped_func(1, 2, z=3) == 9 == func(1, 2) * 3

    # A more realistic application: Saving outputs to a specific location on output

    def func(x, y):
        return x + y

    def save_on_output_egress(v, *, k, s):
        s[k] = v
        return v

    save_on_output = Wrapx(func, egress=save_on_output_egress)
    # TODO: should be `(x, y, *, k, s)` --> Need to work on the merge for this.
    assert str(signature(save_on_output)) == '(x, y, k, s)'

    store = dict()
    save_on_output(1, 2, k='save_here', s=store)
    assert save_on_output(1, 2, k='save_here', s=store) == 3 == func(1, 2)
    assert store == {'save_here': 3}

    # Trying out a caller: Here, we want to wrap the function so it will apply to an
    # iterable of inputs, returning a list of results

    def func(x, y=2):
        return x + y

    def iterize(func, args, kwargs):
        first_arg_val = next(iter(kwargs.values()))
        return list(map(func, first_arg_val))

    iterized_func = Wrapx(func, caller=iterize)

    assert iterized_func([1, 2, 3, 4]) == [3, 4, 5, 6]

    # The same as above, except generalized to allow other variables (here ``y``) to
    # be input as well

    from functools import partial

    def func(x, y):
        return x + y

    def iterize_first_arg(func, args, kwargs):
        first_arg_name = next(iter(kwargs))
        remaining_kwargs = {k: v for k, v in kwargs.items() if k != first_arg_name}
        return list(map(partial(func, **remaining_kwargs), kwargs[first_arg_name]))

    iterized_func = Wrapx(func, caller=iterize_first_arg)

    assert iterized_func([1, 2, 3, 4], 10) == [11, 12, 13, 14]

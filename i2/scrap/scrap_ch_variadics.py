from i2 import Sig
from i2.signatures import ch_variadics_to_non_variadic_kind


def foo(a, *args, bar, **kwargs):
    return f'{a=}, {args=}, {bar=}, {kwargs=}'


assert str(Sig(foo)) == '(a, *args, bar, **kwargs)'
wfoo = ch_variadics_to_non_variadic_kind(foo)
str(Sig(wfoo))
#'(a, args=(), *, bar, kwargs={})'

# And now to do this:

foo(1, 2, 3, bar=4, hello='world')
# "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"

# We can do it like this instead:

wfoo(1, (2, 3), bar=4, kwargs=dict(hello='world'))
# "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"

# Note, the outputs are the same. It's just the way we call our function that has
# changed.

assert wfoo(1, (2, 3), bar=4, kwargs=dict(hello='world')) == foo(
    1, 2, 3, bar=4, hello='world'
)
assert wfoo(1, (2, 3), bar=4) == foo(1, 2, 3, bar=4)
assert wfoo(1, (), bar=4) == foo(1, bar=4)

# Note that if there is not variadic positional arguments, the variadic keyword
# will still be a keyword-only kind.


@ch_variadics_to_non_variadic_kind
def func(a, bar=None, **kwargs):
    return f'{a=}, {bar=}, {kwargs=}'


str(Sig(func))
#'(a, bar=None, *, kwargs={})'
assert (
    func(1, bar=4, kwargs=dict(hello='world'))
    == "a=1, bar=4, kwargs={'hello': 'world'}"
)

# If the function has neither variadic kinds, it will remain untouched.


def func(a, /, b, *, c=3):
    return a + b + c


ch_variadics_to_non_variadic_kind(func) == func
# True

# If you only want the variadic positional to be handled, but leave leave any
# VARIADIC_KEYWORD kinds (**kwargs) alone, you can do so by setting
# `ch_variadic_keyword_to_keyword=False`.
# If you'll need to use `ch_variadics_to_non_variadic_kind` in such a way
# repeatedly, we suggest you use `functools.partial` to not have to specify this
# configuration repeatedly.

from functools import partial

tuple_the_args = partial(
    ch_variadics_to_non_variadic_kind, ch_variadic_keyword_to_keyword=False
)


@tuple_the_args
def foo(a, *args, bar=None, **kwargs):
    return f'{a=}, {args=}, {bar=}, {kwargs=}'


Sig(foo)
# <Sig (a, args=(), *, bar=None, **kwargs)>
foo(1, (2, 3), bar=4, hello='world')
# "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"

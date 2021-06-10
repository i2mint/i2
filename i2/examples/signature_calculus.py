"""
# Signature Calculus

`Sig` is an extension of the `inspect.Signature` object that puts more
goodies at your fingertips.

First of all, a `Sig` instance can be made from a variety of types


>>> from i2.signatures import *
>>> Sig()
<Sig ()>
>>> Sig('self')
<Sig (self)>
>>> Sig(lambda a, b, c=0: None)
<Sig (a, b, c=0)>

>>> Sig(Parameter('foo', Parameter.POSITIONAL_ONLY, default='foo', annotation=int))
<Sig (foo: int = 'foo', /)>
>>> Sig() + 'self' + (lambda a, b, c=0: None) - 'c' + P('c', default=1)
<Sig (self, a, b, c=1)>


Note a difference between `Sig.extract_args_and_kwargs` and binding by hand.

>>> def formula1(w, /, x: float, y=1, *, z: int = 1):
...     return ((w + x) * y) ** z
...
>>>
>>> sig = Sig(formula1)
>>> b = sig.bind(1,2,y=10)
>>> b.apply_defaults()
>>> b.args, b.kwargs
((1, 2, 10), {'z': 1})
>>>
>>> args, kwargs = sig.extract_args_and_kwargs(1, 2, y=10, _apply_defaults=True)
>>> assert args == (1,)
>>> assert kwargs == {'x': 2, 'y': 10, 'z': 1}

bind method seems to favor args over kwargs.

"""

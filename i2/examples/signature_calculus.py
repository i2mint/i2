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

"""

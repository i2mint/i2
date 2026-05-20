---
name: i2-wrapper
description: >-
  Wrapping functions to transform their interface, inputs, and output with
  i2.wrapper — wrap, Ingress/Egress, ch_names, include_exclude, rm_params,
  arg_val_converter, partialx, and caller-based wrapping (Wrapx). Trigger on
  "wrap a function and change its signature", "transform arguments before
  calling", "rename/remove/reorder a function's parameters but keep it
  working", "convert/cast argument values", "make a CLI/HTTP-friendly version
  of a function", "currying with a clean signature", "apply a function over an
  iterable of inputs (iterize)", "ingress / egress", "decorator factory".
audience: developers
---

# `i2.wrapper` — interface-transforming function wrappers

`i2/wrapper.py` builds **wrapped functions whose interface differs from the
wrapped function's**, while keeping the body and the advertised signature
consistent. Where `Sig` (`i2-sig-arithmetic`) only restamps a signature,
`i2.wrapper` also rewires the *call* so the wrapper actually works.

```python
from i2.wrapper import wrap, Ingress, ch_names, include_exclude, rm_params, partialx
```

## The mental model

```
outer call  ──▶  ingress  ──▶  inner call  ──▶  func  ──▶  egress  ──▶  output
(*args,**kw)   (transform)    (*args,**kw)              (transform)
```

- **`ingress`**: defines the *outer* (wrapper) signature **and** maps outer
  arguments to the inner `(args, kwargs)` the wrapped `func` expects.
- **`egress`**: post-processes `func`'s return value; its return annotation
  becomes the wrapper's return annotation.
- **`caller`**: an alternative to ingress/egress — full control over *how*
  `func` is invoked (routes to `Wrapx`).

## When to use it

- You need a function variant with a **different signature** (renamed,
  reordered, fewer params, added params) that **still works**.
- You need to **transform argument values** before the call (cast strings to
  numbers for a CLI/web interface, normalize, coerce types).
- You need to **post-process output** (wrap, format, take `len`, etc.).
- You need a **custom call strategy** — e.g. apply the function over an iterable
  of inputs, retry, dispatch.

If you only need an unchanged signature copied over, use `functools.wraps`. If
you only need to *advertise* a signature, use `Sig` as a decorator. Use
`i2.wrapper` when the *behavior* must change too.

## 1. `wrap(func, *, ingress=None, egress=None, caller=None)`

The one entry point. With no transformers it's a near-no-op; add `ingress` /
`egress` / `caller` as needed. Usable directly or as a decorator factory.

```python
def func(a, b):
    return a * b

# transform inputs: double the first arg
def ingress(a, b):
    return (2 * a,), dict(b=b)        # -> (inner_args, inner_kwargs)

wrapped = wrap(func, ingress=ingress)
wrapped(2, 'Hi')                      # 'HiHiHiHi'

# post-process output
wrap(func, egress=len)(2, 'co')       # 4   == len('coco')
```

An `ingress` function must return `(args_tuple, kwargs_dict)`. Its **own
signature becomes the wrapper's signature**, so it both defines the new
interface and bridges to the old one. `ingress` can add parameters:

```python
def ingress(a, b, c):                 # wrapper gains a 'c' parameter
    return (a, b + c), dict()
wrap(func, ingress=ingress)(2, 'Hi', 'world!')   # 'Hiworld!Hiworld!'
```

## 2. `Ingress` — a templated ingress builder

Writing an ingress by hand is often simplest, but `Ingress` handles the
mechanics for the common case "transform argument **values** by name". You give
it the inner signature and a `kwargs_trans` (a `dict -> dict` transform):

```python
from inspect import signature

def f(w, /, x: float, y=2, *, z: int = 3):
    return w + x * y ** z

# cast every argument to int — e.g. for a CLI where all inputs are strings
cli_f = wrap(f, ingress=Ingress(signature(f),
                                kwargs_trans=lambda d: {k: int(v) for k, v in d.items()}))
cli_f("2", "3", "4")                  # works: strings cast to ints
```

`kwargs_trans` may be 1-to-many or many-to-1 across argument names. To also
change the *outer* parameter names, pass an `outer_sig`:

```python
ingress = Ingress(inner_sig=signature(f),
                  kwargs_trans=my_trans,
                  outer_sig=Sig(f).ch_names(y='you'))
wrapped = ingress.wrap(f)             # Ingress has a .wrap convenience method
```

> Prefer a plain ingress function for one-offs; reach for `Ingress` when
> building general/reusable wrapping tooling.

## 3. Ready-made wrappers — use these before hand-rolling an ingress

These cover the most common interface edits:

```python
from i2.wrapper import ch_names, include_exclude, rm_params, arg_val_converter

def f(w, /, x: float, y=2, *, z: int = 3):
    return w + x * y ** z

# rename parameters
g = ch_names(f, w='DoubleYou', z='Zee')          # sig: (DoubleYou, /, x, y=2, *, Zee=3)

# keep / drop / reorder parameters
def foo(a, b, c='C', d='D'): ...
include_exclude(foo, include='b a', exclude='c d')   # sig: (b, a)

# remove parameters (must have defaults unless you opt out)
def h(x, y=1, z=2): return x + y * z
rm_params(h, params_to_remove='z')               # sig: (x, y=1)

# convert argument values per-name
arg_val_converter(f, x=float, w=int)             # casts x with float(), w with int()
```

All can also be used as decorator factories: `@ch_names(a='alpha')`,
`@include_exclude(include=...)`, `@rm_params(params_to_remove=...)`.

## 4. `partialx` — `functools.partial` with a clean signature

`partialx` fixes `partial`'s ugliness: it can drop the fixed params from the
signature, name the result, and reorder so defaults end up last.

```python
from i2.wrapper import partialx

def f(a, b=2, c=3):
    return a + b * c

curried = partialx(f, c=10, _rm_partialize=True)
curried.__name__                                  # 'f'
str(Sig(curried))                                 # '(a, b=2)'   — 'c' gone
```

## 5. `caller` / `Wrapx` — custom call strategy

When ingress/egress isn't enough (you need to control *how* `func` is invoked,
possibly calling it many times), pass a `caller(func, args, kwargs)`. `wrap`
auto-routes to the `Wrapx` class.

```python
def iterize(func, args, kwargs):
    first_arg = next(iter(kwargs.values()))
    return list(map(func, first_arg))

@wrap(caller=iterize)
def add2(x, y=2):
    return x + y

add2([1, 2, 3])                                   # [3, 4, 5]
```

## Quick decision guide

| You want to…                                            | Use                            |
|----------------------------------------------------------|--------------------------------|
| Rename a function's parameters (keep it working)         | `ch_names`                     |
| Keep / drop / reorder parameters                         | `include_exclude`              |
| Remove (defaulted) parameters                            | `rm_params`                    |
| Cast/convert argument values by name                     | `arg_val_converter` / `Ingress`|
| Post-process the return value                            | `wrap(f, egress=...)`          |
| Arbitrary input transform + new signature                | `wrap(f, ingress=...)`         |
| `partial` but with a clean, named signature              | `partialx`                     |
| Control *how* the function is called (iterize/retry/...) | `wrap(f, caller=...)` (`Wrapx`)|
| Only advertise a signature, no behavior change           | `Sig` decorator (`i2-sig-arithmetic`) |

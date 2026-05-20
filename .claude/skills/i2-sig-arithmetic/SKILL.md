---
name: i2-sig-arithmetic
description: >-
  Building, merging, and editing function signatures with i2's Sig — signature
  "+/-" arithmetic, changing parameter names/kinds/defaults/annotations, and
  giving a function a different signature (a la functools.wraps but
  signature-driven). Trigger on "merge two signatures", "combine the parameters
  of several functions", "give this function a new signature", "rename/reorder
  function parameters", "change a parameter's default", "make all params
  positional-or-keyword", "add an optional keyword argument to a function",
  "Sig() decorator", "ch_func_to_all_pk".
audience: developers
---

# `i2.Sig` — signature construction, arithmetic & rewriting

This skill covers building signatures, **merging/subtracting** them with `+` /
`-`, **editing** parameters, and **stamping** a signature onto a function. For
plain introspection and binding call arguments to names, see `i2-signatures`.

```python
from i2 import Sig
```

## When to use it

- You write a decorator/wrapper and the wrapper's signature must be **derived**
  from one or more wrapped functions (so IDEs, `help()`, and CLIs see it right).
- You build a function dynamically (`*args, **kwargs` body) and want it to
  *advertise* a real, meaningful signature.
- You need a function variant with renamed, reordered, redefaulted, or removed
  parameters — without rewriting the function.
- You want to relax kind constraints (e.g. make positional-only params
  keyword-able).

## 1. Make a signature

```python
Sig(['a', 'b'])                            # <Sig (a, b)>
Sig('x y z')                               # <Sig (x, y, z)>
Sig('(a: int = 0, b: str = None) -> str')  # parses a signature string
Sig([('a', 1), ('b', 2.0, float),          # (name, default[, annotation]) tuples
     {'name': 'c', 'kind': Sig.KEYWORD_ONLY, 'default': 0}])
Sig(some_function)                         # from a callable
```

## 2. Signature arithmetic: `+` (merge) and `-` (remove)

`+` merges parameters from another signature / param-spec; `-` removes named
parameters. This is the workhorse for deriving a wrapper's signature.

```python
def foo(x): pass
def bar(y: int, *, z=2): pass

Sig(foo) + ['a', 'b'] + Sig(bar)
# <Sig (x, a, b, y: int, z=2)>    note: bar's keyword-only-ness is re-flowed

Sig('a b c d') - 'b'             # <Sig (a, c, d)>
Sig('a b c d') - ['a', 'c']      # <Sig (b, d)>
```

`Sig.from_objs(*objs)` merges several functions/param-specs at once:

```python
def f(w, /, x=1): pass
def g(i, w, /, j=2): pass
Sig.from_objs(f, g, ['a', ('b', 3.14)])
# <Sig (w, i, /, a, x=1, j=2, b=3.14)>     (shared 'w' merged, not duplicated)
```

Merging respects parameter-ordering rules (positional-only, then PK, then
keyword-only) and de-duplicates shared names. Conflicting param definitions
raise `IncompatibleSignatures`.

## 3. Edit parameters — `ch_*` and `modified`

Each returns a **new** `Sig` (immutable-style):

```python
def f(a, b: int = 2, *, c=3): pass
s = Sig(f)

s.ch_names(a='alpha')          # <Sig (alpha, b: int = 2, *, c=3)>
s.ch_defaults(b=20)            # <Sig (a, b: int = 20, *, c=3)>
s.ch_annotations(a=str)        # <Sig (a: str, b: int = 2, *, c=3)>
s.ch_kinds(c=Sig.POSITIONAL_OR_KEYWORD)
s.modified(b={'default': 99, 'annotation': float})   # change several attrs at once
s.without_defaults             # <Sig (a)>     drop the defaulted params
s.add_optional_keywords(verbose=False)               # append a keyword-only param
```

## 4. Stamp a signature onto a function — `Sig` as a decorator

A `Sig` instance is **callable**: it writes `__signature__`, `__annotations__`,
`__defaults__`, and `__kwdefaults__` onto the function (more thorough than
`functools.wraps`, which doesn't update defaults).

```python
@Sig('a b c')
def func(*args, **kwargs):
    print(args, kwargs)

Sig(func)                      # <Sig (a, b, c)>
```

The classic use — derive a dynamic function's signature from real functions:

```python
sig = Sig(f) + g + ['a', ('b', 3.14)] - 'b'

@sig
def some_func(*args, **kwargs):
    ...
# some_func now advertises the merged signature
```

> **Warning:** stamping a signature changes only what the function *advertises*.
> Changing names or kinds without a matching body (or an `ingress`, see the
> `i2-wrapper` skill) produces a function that lies about its interface and may
> fail at call time. `wrap`/`Ingress` exist precisely to keep body and
> signature consistent. Changing **defaults** *does* change behavior, since the
> function reads `__defaults__`/`__kwdefaults__`.

## 5. Normalize argument kinds — `ch_func_to_all_pk`

`ch_func_to_all_pk(func)` returns an equivalent function whose parameters are
all `POSITIONAL_OR_KEYWORD` — no positional-only `/`, no keyword-only `*`. This
removes the "how was this argument passed" ambiguity, which simplifies any
meta-programming over the function.

```python
from i2.signatures import ch_func_to_all_pk

def f(a, /, b, *, c=None, **kwargs):
    return a + b * c

Sig(f)                       # (a, /, b, *, c=None, **kwargs)
ff = ch_func_to_all_pk(f)
Sig(ff)                      # (a, b, c=None, **kwargs)
ff(1, 2, 3)                  # 7   — now callable all-positional
```

Related: `all_pk_signature(sig)` (just the signature transform),
`tuple_the_args` / `ch_variadics_to_non_variadic_kind` (replace `*args`/`**kwargs`
with an explicit tuple/dict parameter).

## Quick decision guide

| You want to…                                       | Use                               |
|-----------------------------------------------------|-----------------------------------|
| Build a signature from names/strings/tuples         | `Sig([...])`, `Sig('a b c')`      |
| Merge parameters of several functions               | `sigA + sigB`, `Sig.from_objs(*)` |
| Remove parameters                                   | `sig - 'name'`                    |
| Rename / redefault / re-annotate / re-kind params   | `ch_names` / `ch_defaults` / …    |
| Change several attrs of one param                   | `sig.modified(name={...})`        |
| Give a dynamic function a real signature            | `@sig` (Sig as decorator)         |
| Drop positional-only / keyword-only constraints     | `ch_func_to_all_pk(func)`         |
| Keep body+signature consistent while rewriting      | use `i2.wrapper` (`i2-wrapper`)   |

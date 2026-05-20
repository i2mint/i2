---
name: i2-signatures
description: >-
  Introspecting function signatures and binding call arguments to parameter
  names with i2's Sig class — turning an arbitrary (*args, **kwargs) call into a
  named-argument dict (with defaults filled in), and back. Trigger on "bind args
  to parameter names", "get the full kwargs of a call", "apply defaults to a
  call", "what are this function's parameter names/kinds/defaults", "call a
  function forgivingly / ignoring extra kwargs", "normalize a call for
  caching/hashing", "inspect.signature is not enough", "Sig", "map_arguments".
audience: developers
---

# `i2.Sig` — signature introspection & call-argument binding

`Sig` (in `i2/signatures.py`) is a subclass of `inspect.Signature` with a
dict-like interface and a lot of extra sugar. Reach for it whenever you need to
**reason about a function's parameters** or **manipulate how a call's arguments
map to parameter names** — especially in meta-programming, dispatching,
validation, and caching code.

This skill covers **introspection** and **call-argument binding**. For merging
signatures, `+`/`-` arithmetic, and giving a function a new signature, see the
companion skill `i2-sig-arithmetic`.

```python
from i2 import Sig          # top-level export
# or: from i2.signatures import Sig
```

## When to use it

- You have `*args, **kwargs` at runtime and need the **named** form
  (`{param_name: value}`) — e.g. to validate, log, transform, or hash a call.
- You need defaults filled in for parameters the caller did not pass.
- You want to call a function but your argument pool has **extra** keys it
  doesn't accept (don't choke — use `call_forgivingly`).
- You need to know parameter **names, kinds, defaults, annotations** without
  hand-rolling `inspect` boilerplate.
- A function has positional-only / keyword-only constraints and you need to
  produce a valid `(args, kwargs)` pair to call it.

If you only need a plain `inspect.Signature`, you don't need `Sig` — but `Sig`
is a drop-in superset (it *is* a `Signature`).

## 1. Introspection — the column views

`Sig` gives you per-parameter "columns" as dicts, plus name lists:

```python
def func(z, a: float = 1.0, /, b=2, *, c: int = 3):
    pass

sig = Sig(func)
sig.names                # ['z', 'a', 'b', 'c']
sig.defaults             # {'a': 1.0, 'b': 2, 'c': 3}   (only defaulted params)
sig.annotations          # {'a': float, 'c': int}
sig.kinds                # {'z': POSITIONAL_ONLY, 'a': POSITIONAL_ONLY,
                         #  'b': POSITIONAL_OR_KEYWORD, 'c': KEYWORD_ONLY}
sig.required_names       # ('z',)        — no default, must be supplied
sig.n_required           # 1
sig.has_var_positional   # False  (also .has_var_keyword, .var_keyword_name, ...)
```

`Sig` is also a `Mapping` keyed by parameter name:

```python
sig['b']                 # the Parameter object for 'b'
list(sig)                # ['z', 'a', 'b', 'c']
'c' in sig               # True
```

Construct a `Sig` from many things, not just functions:

```python
Sig(['a', 'b', 'c'])                       # <Sig (a, b, c)>
Sig('x y z')                               # <Sig (x, y, z)>  (whitespace-split)
Sig('(a: int = 0, b: str = None) -> str')  # parses a signature string
Sig(['a', ('b', None), ('c', 42, int)])    # <Sig (a, b=None, c: int = 42)>
```

## 2. THE core idea — bind a call to named arguments: `map_arguments`

`Sig.map_arguments(args, kwargs, *, apply_defaults=False, ...)` maps a call's
positional + keyword arguments to a single `{param_name: value}` dict.

```python
def f(a, b=2, c=3):
    pass

Sig(f).map_arguments((2, 10), {}, apply_defaults=True)
# {'a': 2, 'b': 10, 'c': 3}      <- f(2, 10) fully resolved, with defaults filled
```

Key options (8 combinations — see the docstring):

- `apply_defaults=True` — fill in defaults for params the caller omitted.
- `allow_excess=True` — silently drop kwargs keys the function doesn't accept
  (lets several functions draw from one shared argument pool).
- `allow_partial=True` — don't error on missing required args (for a call you
  intend to complete later).
- `ignore_kind=True` — accept positional-only as keyword and vice versa.

```python
def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
    ...

Sig(foo).map_arguments((11,), {"x": 22})                    # {'w': 11, 'x': 22}
Sig(foo).map_arguments((11,), {"x": 22, "extra": -1},
                       allow_excess=True)                   # {'w': 11, 'x': 22}
```

There is a variadic-input convenience wrapper — same thing, but you pass the
call directly instead of an `(args, kwargs)` pair:

```python
Sig(foo).map_arguments_from_variadics(11, 22, _apply_defaults=True)
# {'w': 11, 'x': 22, 'y': 'YY', 'z': 'ZZ'}
```

> Note: `kwargs_from_args_and_kwargs` and `extract_kwargs` are **deprecated
> aliases** of `map_arguments` / `map_arguments_from_variadics`. Use the new
> names.

## 3. The inverse — back to a callable `(args, kwargs)`: `mk_args_and_kwargs`

Once you have a named-arguments dict, `mk_args_and_kwargs` turns it back into an
`(args, kwargs)` pair you can actually splat into the function — respecting
positional-only / keyword-only constraints.

```python
def f(w, /, x, y=1, *, z=1):
    return ((w + x) * y) ** z

s = Sig(f)
named = s.map_arguments((4, 3), {}, apply_defaults=True)  # {'w':4,'x':3,'y':1,'z':1}
args, kwargs = s.mk_args_and_kwargs(named)                # ((4,), {'x':3,'y':1,'z':1})
f(*args, **kwargs)                                        # callable again
```

`args_limit` controls how many params go into `args` vs `kwargs`
(`0` = fewest possible args, `None` = most, positive int = first N).

The map-then-process-then-rebuild pattern is the canonical use:
`map_arguments` → mutate the named dict → `mk_args_and_kwargs` → call.

## 4. Call a function "forgivingly" — ignore extra arguments

When you have a fat argument pool and want each function to take only what it
needs, don't hand-filter — use `call_forgivingly`:

```python
from i2.signatures import call_forgivingly

def foo(a, b: int = 0, c=None):
    return a, b, c

call_forgivingly(foo, "for_a", c=42, intruder="ignored")
# ('for_a', 0, 42)        <- 'intruder' silently dropped
```

It is `map_arguments(..., allow_excess=True)` followed by `mk_args_and_kwargs`.
If your kwargs may contain a key named `func`, use `_call_forgivingly(func,
args, kwargs)` (explicit tuple/dict form) to avoid the name clash.

## Quick decision guide

| You want to…                                          | Use                              |
|-------------------------------------------------------|----------------------------------|
| Get param names / kinds / defaults / annotations      | `Sig(f).names` / `.kinds` / …    |
| Turn a `(args, kwargs)` call into a named dict         | `Sig(f).map_arguments(...)`      |
| …directly from `*args, **kwargs`                       | `map_arguments_from_variadics`   |
| Fill in defaults for omitted params                    | `apply_defaults=True`            |
| Turn a named dict back into a callable `(args, kwargs)`| `Sig(f).mk_args_and_kwargs(...)` |
| Call `f` while ignoring extra arguments                | `call_forgivingly(f, ...)`       |
| Merge signatures / give `f` a new signature            | see `i2-sig-arithmetic` skill    |

## Recipe: stable "full kwargs" dict for content-addressed caching

To turn an arbitrary call `f(*args, **kwargs)` into a deterministic, fully
named, defaults-filled dict (the basis of a cache key):

```python
def full_kwargs(f, *args, **kwargs):
    return Sig(f).map_arguments(args, kwargs, apply_defaults=True)

full_kwargs(lambda a, b=2, c=3: None, 2, 10)   # {'a': 2, 'b': 10, 'c': 3}
```

The result is order-independent (keyed by name) and complete (defaults filled),
so hashing/JSON-serializing it gives a stable call fingerprint. Caveat: the
*values* must themselves be hashable/serializable; `map_arguments` only
normalizes the argument-to-name mapping.

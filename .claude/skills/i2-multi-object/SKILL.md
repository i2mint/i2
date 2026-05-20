---
name: i2-multi-object
description: >-
  Composing and combining a fixed collection of functions (or context managers)
  with i2.multi_object — Pipe (function composition), FuncFanout (apply many
  functions to the same input), ParallelFuncs (per-key dict-to-dict), FlexFuncFanout
  (shared argument pool), and ContextFanout (bundle context managers). Trigger
  on "compose functions into a pipeline", "apply several functions to the same
  arguments", "run a function per dict key", "chain transformations", "bundle
  multiple context managers as one", "function pipeline", "fan out / fan in".
audience: developers
---

# `i2.multi_object` — combine a fixed set of functions

`i2/multi_object.py` provides small, composable tools for operating on a
**fixed collection of objects** — mostly functions, plus context managers.
They are plain callables with sensible signatures, picklable, and nestable.

```python
from i2.multi_object import Pipe, FuncFanout, ParallelFuncs, ContextFanout
```

## When to use which — decision guide

| You want to…                                                  | Use              |
|---------------------------------------------------------------|------------------|
| Chain functions: output of one is input of the next            | `Pipe`           |
| Apply many functions to the **same** argument(s)               | `FuncFanout`     |
| Apply many functions to the same args, **tolerating extras**   | `FlexFuncFanout` |
| Route a dict: a function per key, producing a dict             | `ParallelFuncs`  |
| Bundle several context managers into one `with`                | `ContextFanout`  |

## `Pipe` — function composition

`Pipe(f1, f2, ..., fn)` returns a callable `input -> f1 -> f2 -> ... -> fn`.
The first function may take any signature; the rest receive one argument.

```python
def foo(a, b=2):
    return a + b

f = Pipe(foo, str)
f(3)                                  # '5'

# names are documentation-only (ignored at call time)
g = Pipe(add=lambda x, y: x + y, double=lambda x: x * 2, stringify=str)
g(2, 3)                               # '10'
```

`Pipe` adopts a signature from its first and last functions, is picklable, and
raises a richly contextual error (which function, which input) when a step
fails. Use `flatten_pipe` to flatten nested `Pipe`s.

## `FuncFanout` — one input, many functions

The dual of `map`: apply several functions to the *same* arguments. Calling a
`FuncFanout` yields `(name, result)` pairs (a generator).

```python
def foo(a): return a + 2
def bar(a): return a * 2

m = FuncFanout(foo, bar_results=bar, groot=lambda a: 'I am groot')
dict(m(10))                           # {'foo': 12, 'bar_results': 20, 'groot': 'I am groot'}
```

Names come from keyword args, or are auto-assigned (`_0`, `_1`, ...) for
positional ones. To get a dict directly, compose with `Pipe`:

```python
to_dict = Pipe(FuncFanout(add2=foo, mul2=bar), dict)
to_dict(10)                           # {'add2': 12, 'mul2': 20}
```

`FlexFuncFanout` is the forgiving variant: each member function draws only the
arguments it needs from a shared pool and ignores the rest — useful when
functions have different (and overlapping) signatures.

## `ParallelFuncs` — dict-to-dict, a function per key

Build a multi-channel function: each key of the input dict is handled by its
own function, producing a same-keyed output dict.

```python
multi = ParallelFuncs(
    say_hello=lambda x: f"hello {x}",
    say_goodbye=lambda x: f"goodbye {x}",
)
multi({'say_hello': 'world', 'say_goodbye': 'Lenin'})
# {'say_hello': 'hello world', 'say_goodbye': 'goodbye Lenin'}
```

Keys that aren't valid identifiers? Pass a dict positionally:
`ParallelFuncs({'x+y': fn1, 'x*y': fn2})`. `ParallelFuncs` instances chain well
inside a `Pipe` for staged per-channel processing.

## `ContextFanout` — many context managers as one

Bundle objects (some of which are context managers, some not) into a single
context manager that enters/exits all of them together. Non-context-manager
members are simply held, not entered.

```python
from contextlib import contextmanager

@contextmanager
def cm(x):
    print('open'); yield x + 1; print('close')

with ContextFanout(a=cm(2), b=open(__file__)) as c:
    ...                               # both 'a' and 'b' are entered
# both are exited on block exit
```

Members are reachable by name (`c.a`, `c.b`) inside the block.

## Notes

- All of these subclass `MultiObj` / `MultiFunc` and behave as `Mapping`s over
  their member objects (`list(pipe)` gives member names, etc.).
- Members can be named (keyword args) or unnamed (positional, auto-named).
- For richer interface-transforming wrapping (changing signatures, casting
  arguments), see the `i2-wrapper` skill — `multi_object` is for *combining*
  functions, `wrapper` is for *reshaping* them.

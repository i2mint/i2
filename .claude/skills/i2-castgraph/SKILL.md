---
name: i2-castgraph
description: >-
  Routing data through a graph of type/representation conversions with
  i2.castgraph's TransformationGraph — register pairwise transformers, then
  convert any source representation to any reachable target via multi-hop
  shortest-path routing. Trigger on "convert between many data
  representations", "type conversion registry / service", "transformation
  graph", "filepath -> text -> record style pipeline", "find a conversion path
  between two kinds", "adapter / anti-corruption layer for external formats",
  "cast a thing whose form varies but role is stable".
audience: developers
---

# `i2.castgraph` — a routing graph for data conversions

`i2/castgraph.py` solves the **"stable role, unstable representation"** problem:
a resource has one semantic role (config, text, record) but shows up in many
forms (filepath, string, dict, custom class), while consumers want a specific
form. Instead of writing every pairwise converter, you register the
transformations you have as **edges in a graph**, and `castgraph` finds the
best multi-hop route between any two representations.

```python
from i2.castgraph import TransformationGraph, ConversionError
```

## When to use it

- You have **N data representations** and conversions between some of them, and
  want to convert from any one to any other without writing N² converters.
- You want a **central conversion registry** for a bounded context (a
  "type converter" / "anti-corruption layer" keeping external formats out of
  your core domain).
- You want to **prefer faster/more-accurate conversions** via per-edge `cost`,
  and have the shortest-cost path chosen automatically.

If you only convert between two fixed types, a plain function is enough — use
`castgraph` when the *combinatorics* or the *routing* is the problem.

## Core concepts

- **Kind** — any hashable identifier for a representation: a `type` (`str`,
  `dict`), a string label (`'text'`, `'json_dict'`), or any custom marker.
- **Edge / transformation** — a function `(obj, ctx) -> obj` converting one
  kind to another, with a `cost` (default `1.0`).
- **isa predicate** — optional `obj -> bool` attached to a kind, used to
  auto-detect an object's kind.

## The recommended API: `TransformationGraph`

```python
graph = TransformationGraph()

# 1. (optional) declare kinds with predicates for auto-detection
graph.add_node('text',   isa=lambda x: isinstance(x, str))
graph.add_node('record', isa=lambda x: isinstance(x, dict))

# 2. register transformation edges (decorator form)
@graph.register_edge('text', 'record', cost=0.5)
def text_to_record(t, ctx):
    import json
    return json.loads(t or "{}")

# 3. transform — multi-hop routing happens automatically
graph.transform('{"x": 1}', 'record', from_kind='text')   # {'x': 1}
```

`transform(obj, to_kind, *, from_kind=None, context=None, use_result_cache=False)`:

- If `from_kind` is given, it routes `from_kind -> to_kind` directly.
- If omitted, it tries `type(obj)` and its MRO as the source kind.
- Raises `ConversionError` if no path exists.

When kinds are *types*, `from_kind` is inferred for free:

```python
g = TransformationGraph()
@g.register_edge(str, int)
def s2i(s, ctx): return int(s)

g.transform("42", int)            # 42  — from_kind inferred as str
```

When a kind is a **string label** (not a type), pass `from_kind` explicitly, or
register an `isa` predicate and use `transform_any` for auto-detection:

```python
graph.transform_any('{"x": 1}', 'record')   # detects kind via isa predicates
```

## Multi-hop routing & composed transformers

Edges compose. Register `path -> text` and `text -> record`, and you can go
`path -> record` directly — `castgraph` finds the route:

```python
g = TransformationGraph()
g.add_node('path', isa=lambda x: isinstance(x, str) and x.startswith('/'))
g.add_node('text', isa=lambda x: isinstance(x, str))
g.add_node('record', isa=lambda x: isinstance(x, dict))

@g.register_edge('path', 'text')
def p2t(p, ctx): return (ctx or {}).get(p, '')

@g.register_edge('text', 'record', cost=0.5)
def t2r(t, ctx):
    import json
    return json.loads(t or '{}')

ctx = {'/a.json': '{"x": 1}'}
g.transform('/a.json', 'record', from_kind='path', context=ctx)   # {'x': 1}

# get a reusable composed transformer for a (from, to) pair:
to_record = g.get_transformer('path', 'record')
to_record('/a.json', ctx)                                          # {'x': 1}
```

The `context` dict is a side channel passed to every transformer on the route —
use it for I/O handles, flags, caches, filesystem mocks, etc.

## Inspecting the graph

```python
g.kinds()                 # all registered kinds
g.reachable_from('path')  # {'text', 'record'} — kinds you can route to
g.sources_for('record')   # kinds that can route to 'record'
g.detect_kind(obj)        # kind of obj via isa predicates / detector
```

## Design guidance

- One `TransformationGraph` per bounded context; keep edges local.
- Keep transformers small, pure-ish, and doctested.
- Use a **canonical hub kind** when many formats interoperate, to avoid an
  edge explosion.
- Tune `cost` to prefer fast/accurate routes; paths are `lru_cache`-d.
- Keep adapters at the boundary; the core domain should consume domain kinds.

> **Deprecated:** `ConversionRegistry` (with `.register()` / `.convert()`) is
> the legacy type-only interface and emits `DeprecationWarning`. Use
> `TransformationGraph` (`.register_edge()` / `.transform()`) for all new code.
> The module docstring has a migration guide.

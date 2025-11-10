# i2

Core tools for minting code.

[Documentation here.](https://i2mint.github.io/i2/)

## What's mint?

Mint stands for "Meta-INTerface".

Minting is core technique of i2i: It can be seen as the encapsulation of a construct’s interface into a (data) 
structure that contains everything one needs to know about the construct to perform a specific action 
with or on the construct.

A little note on the use of “encapsulation”. The term is widely used in computer science, 
and is typically tied to object oriented programming. Wikipedia provides two definitions:
* A language mechanism for restricting direct access to some of the object's components.
* A language construct that facilitates the bundling of data with the methods (or other functions) 
operating on that data.

Though both these definitions apply to minting, 
the original sense of the word “encapsulate” is even more relevant (from google definitions): 
* express the essential features of (something) succinctly
* enclose (something) in or as if in a capsule

Indeed, minting is the process of enclosing a construct into a “mint” (for “Meta INTerface”) 
that will express the features of the construct that are essential to the task at hand. 
The mint provides a declarative layer of the construct that allows one to write code that operates with this layer, 
which is designed to be (as) consistent (as possible) from one system/language to another.

For example, whether a (non-anonymous) function was written in C, Python, or JavaScript, 
it will at least have a name, and it's arguments will (most often) have names, and may have types. 
Similarly with "data objects": The data of both JavaScript and Python objects can be represented by a tree whose 
leaves are base types, which can in turn be represented by a C struct. 

## Key Modules Overview

### i2.castgraph - Type/Kind-Based Transformation Graphs

`castgraph` provides a graph-based system for organizing transformations between different data representations ("kinds"). It routes objects through multi-hop conversion paths, selecting the optimal route based on cost.

**Basic Usage (Type-Based):**

```python
from i2.castgraph import TransformationGraph

graph = TransformationGraph()

# Register transformations between types
@graph.register_edge(str, float)
def str_to_float(s, ctx):
    return float(s)

@graph.register_edge(float, int)
def float_to_int(f, ctx):
    return int(f)

# Automatically routes str -> float -> int
result = graph.transform("42.7", int)
assert result == 42
```

**Advanced Usage (Kind-Based):**

```python
# Define custom "kinds" (not just types)
graph.add_node('json_string', isa=lambda x: isinstance(x, str) and x.startswith('{'))
graph.add_node('config_dict', isa=lambda x: isinstance(x, dict))

@graph.register_edge('json_string', 'config_dict')
def parse_json(text, ctx):
    import json
    return json.loads(text)

# Transform with automatic kind detection
result = graph.transform('{"key": "value"}', 'config_dict', from_kind='json_string')
```

**Key Features:**
- Multi-hop routing with cost-based path selection
- Support for arbitrary hashable kinds (types, strings, custom markers)
- Pluggable kind detection with predicates
- Context propagation for dependency injection
- MRO-aware fallback for type hierarchies

### i2.signatures - Function Signature Manipulation

`signatures` provides a calculus for working with function signatures - introspecting, merging, and modifying them programmatically.

**Signature Introspection:**

```python
from i2.signatures import Sig

def func(z, a: float = 1.0, /, b=2, *, c: int = 3):
    pass

sig = Sig(func)
print(sig.names)        # ['z', 'a', 'b', 'c']
print(sig.defaults)     # {'a': 1.0, 'b': 2, 'c': 3}
print(sig.annotations)  # {'a': <class 'float'>, 'c': <class 'int'>}
```

**Signature Construction:**

```python
# From function
sig1 = Sig(lambda x, y: x + y)

# From list of names
sig2 = Sig(['a', 'b', 'c'])

# From string
sig3 = Sig('x y z')

# All create callable Signature objects
print(sig2)  # <Sig (a, b, c)>
```

**Signature Merging:**

```python
def foo(x, y=1): pass
def bar(z: int, *, w=2): pass

# Combine signatures
combined = Sig(foo) + Sig(bar)
print(combined)  # <Sig (x, y=1, z: int, w=2)>
```

**Decorating with Signatures:**

```python
# Give a function a specific signature
@Sig('a b c')
def func(*args, **kwargs):
    print(f"Called with: {args}, {kwargs}")

# Now func has signature (a, b, c)
func(1, 2, 3)  # Works as expected
```

**Key Features:**
- Extract parameter names, kinds, defaults, and annotations
- Merge multiple signatures flexibly
- Apply signatures as decorators
- Support for all parameter kinds (positional-only, keyword-only, VAR_POSITIONAL, VAR_KEYWORD)
- Signature algebra for composing function interfaces

### i2.wrapper - Ingress/Egress Function Wrapping

`wrapper` provides the `Wrap` class for transforming function inputs and outputs through composable ingress/egress layers.

**Basic Wrapping:**

```python
from i2.wrapper import Wrap

def add(x, y):
    return x + y

# Transform inputs before function, outputs after
wrapped = Wrap(
    add,
    ingress=lambda x, y: (x * 2, y * 2),  # Double inputs
    egress=lambda result: result / 2       # Halve output
)

result = wrapped(3, 4)  # (3*2 + 4*2) / 2 = 7
assert result == 7
```

**Signature Transformation:**

```python
from i2.wrapper import Ingress

def process(data: dict):
    return data['value']

# Change signature: accept 'x' instead of 'data'
ingress = Ingress(
    outer_sig='x',
    inner_sig='data',
    kwargs_trans=lambda x: {'data': {'value': x}}
)

new_func = ingress(process)
result = new_func(42)  # Calls process({'value': 42})
assert result == 42
```

**The Wrap Flow:**

```
*outer_args, **outer_kwargs
        ↓
    [ingress] - transform inputs
        ↓
*inner_args, **inner_kwargs
        ↓
     [func] - original function
        ↓
   func_output
        ↓
    [egress] - transform outputs
        ↓
  final_output
```

**Key Features:**
- Separate ingress (input transformation) and egress (output transformation)
- Signature-aware argument mapping
- Composable wrapper layers
- Supports partial application and argument reordering
- Clean separation of concerns for cross-cutting functionality

### i2.routing_forest - Conditional Logic as Data Structures

`routing_forest` lets you express nested if/then conditions as composable, reusable tree structures instead of tangled code.

**Basic Routing:**

```python
from i2.routing_forest import RoutingForest, CondNode, FinalNode

# Define routing logic as a forest
router = RoutingForest([
    CondNode(
        cond=lambda x: isinstance(x, int),
        then=FinalNode("It's an integer!")
    ),
    CondNode(
        cond=lambda x: isinstance(x, str),
        then=FinalNode("It's a string!")
    )
])

# Get first match
result = next(router(42))
assert result == "It's an integer!"
```

**Nested Conditions:**

```python
# Nested routing with multiple conditions
router = RoutingForest([
    CondNode(
        cond=lambda x: isinstance(x, (int, str)),
        then=RoutingForest([
            CondNode(
                cond=lambda x: int(x) >= 10,
                then=FinalNode("≥ 10")
            ),
            CondNode(
                cond=lambda x: int(x) % 2 == 1,
                then=FinalNode("Odd number")
            )
        ])
    )
])

# Can get all matches or just first
list(router(15))   # ['≥ 10', 'Odd number']
next(router(8))    # None (no matches)
```

**Pattern Matching Example:**

```python
# Router as pattern matcher
def route_value(value):
    router = RoutingForest([
        CondNode(
            cond=lambda x: x < 0,
            then=FinalNode("negative")
        ),
        CondNode(
            cond=lambda x: x == 0,
            then=FinalNode("zero")
        ),
        CondNode(
            cond=lambda x: x > 0,
            then=FinalNode("positive")
        )
    ])
    return next(router(value), "unknown")

assert route_value(-5) == "negative"
assert route_value(0) == "zero"
assert route_value(10) == "positive"
```

**Key Features:**
- Objectify nested if/then logic into composable components
- Both callable and iterable nodes
- Get first match, all matches, or default values
- Cleaner than nested if/elif/else chains for complex routing
- Reusable condition components

### i2.util - Utility Functions and Helpers

`util` provides miscellaneous utility functions for common patterns.

**Identity and Constant Functions:**

```python
from i2.util import asis, return_true, return_false, return_none

# Identity function
assert asis(42) == 42
assert asis([1, 2, 3]) == [1, 2, 3]

# Constant functions (useful as defaults)
assert return_true(anything, goes="here") is True
assert return_false("doesn't", "matter") is False
assert return_none(1, 2, 3) is None
```

**Object Naming:**

```python
from i2.util import name_of_obj

# Get name of various objects
assert name_of_obj(map) == 'map'
assert name_of_obj([1, 2, 3]) == 'list'
assert name_of_obj(lambda x: x) == '<lambda>'

from functools import partial
assert name_of_obj(partial(print, sep=",")) == 'print'
```

**Attribute/Item Access:**

```python
from i2.util import imdict

# Flexible dict-like access
data = imdict({'a': 1, 'b': 2})
assert data.a == 1  # Attribute access
assert data['b'] == 2  # Item access
```

**Laziness Utilities:**

```python
from i2.util import lazyprop

class DataLoader:
    @lazyprop
    def expensive_data(self):
        print("Loading...")
        return [1, 2, 3, 4, 5]

loader = DataLoader()
# First access computes and caches
data1 = loader.expensive_data  # Prints "Loading..."
# Subsequent accesses use cached value
data2 = loader.expensive_data  # No print
assert data1 is data2
```

**Key Features:**
- Common function patterns (identity, constants)
- Object introspection helpers
- Flexible attribute/item access wrappers
- Lazy evaluation utilities
- Deprecation helpers
- String manipulation tools

## Common Patterns

### Composing Transformations

```python
from i2.castgraph import TransformationGraph
from i2.wrapper import Wrap

# Define transformation graph
graph = TransformationGraph()

@graph.register_edge('csv', 'rows')
def parse_csv(text, ctx):
    return [line.split(',') for line in text.strip().split('\n')]

@graph.register_edge('rows', 'records')
def rows_to_records(rows, ctx):
    return [dict(zip(headers, row)) for row in rows[1:]]

# Use with wrapper for clean API
def process_csv(csv_text: str) -> list:
    return graph.transform(csv_text, 'records', from_kind='csv')

# Wrap to add validation
validated = Wrap(
    process_csv,
    ingress=lambda text: (text.strip(),),
    egress=lambda records: [r for r in records if r]  # Filter empties
)
```

### Dynamic Signature Manipulation

```python
from i2.signatures import Sig
from i2.wrapper import Ingress

# Start with a general function
def process(**kwargs):
    return sum(kwargs.values())

# Give it a specific signature
@Sig('a b c')
def typed_process(**kwargs):
    return process(**kwargs)

# Now can call with clear parameters
result = typed_process(1, 2, 3)
assert result == 6
```

### Routing with Validation

```python
from i2.routing_forest import RoutingForest, CondNode, FinalNode
from i2.util import return_none

def validate_input(value):
    """Route to appropriate validator."""
    router = RoutingForest([
        CondNode(
            cond=lambda x: isinstance(x, str),
            then=RoutingForest([
                CondNode(lambda x: len(x) > 0, FinalNode(True)),
                CondNode(lambda x: len(x) == 0, FinalNode(False))
            ])
        ),
        CondNode(
            cond=lambda x: isinstance(x, int),
            then=FinalNode(x >= 0)
        )
    ])
    return next(router(value), False)

assert validate_input("hello") is True
assert validate_input("") is False
assert validate_input(5) is True
assert validate_input(-1) is False
```



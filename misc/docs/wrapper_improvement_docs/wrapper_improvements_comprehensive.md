# Comprehensive Improvement Suggestions for i2.wrapper

This document analyzes `i2.wrapper.py` in comparison with `wrapt`, identifies gaps, and provides actionable improvement suggestions. The goal is to make it EASY to create CORRECT decorators for typical contexts while maintaining backward compatibility where possible.

---

## Executive Summary

**Core Philosophy:**
The correct thing should be easy, and the incorrect thing should be hard. Decorators should preserve signatures by default, and users should opt-in to changing them, not opt-in to preserving them.

**Key Insights from wrapt:**
1. **Object Proxy Pattern**: Transparent delegation of attribute access avoids expensive copying
2. **Descriptor Protocol**: Full descriptor support enables wrapping of classmethods, staticmethods, properties
3. **Universal Decorator Context**: Pass (wrapped, instance, args, kwargs) to give full calling context
4. **Performance-Critical Path**: C extension for hot paths with Python fallback
5. **Thread Safety**: Built-in synchronization primitives

**Priority Levels:**
- 🔴 **Critical**: Should be implemented soon (backward compatible)
- 🟡 **Important**: Consider for next version (may have minor BC issues)
- 🟢 **Enhancement**: Nice to have (future consideration)
- ⚠️ **Breaking**: Would require major version bump

---

## Part 1: Signature Preservation (Critical Issues)

### 1.1 Automatic Signature Preservation from Wrapped Function 🔴

**Current Problem:**
When creating a `Wrap` with an ingress function defined as `def ingress(*args, **kwargs)`, the wrapper loses the original function's signature. Users must manually preserve it:

```python
ingress.__signature__ = inspect.signature(func)
```

**Suggestion:**
Add `preserve_signature` parameter with intelligent detection:

```python
class Wrap:
    def __init__(
        self,
        func,
        *,
        ingress=None,
        egress=None,
        preserve_signature='auto',  # NEW: 'auto' | True | False
        # ... other params
    ):
```

**Implementation Strategy:**

```python
def _should_preserve_signature(ingress, func, preserve_signature):
    """Determine if signature should be auto-preserved."""
    if preserve_signature is False:
        return False
    if preserve_signature is True:
        return True
    
    # 'auto' mode: preserve if ingress has generic (*args, **kwargs)
    if ingress is None:
        return False
    
    ingress_sig = inspect.signature(ingress)
    params = list(ingress_sig.parameters.values())
    
    # Check if ingress has exactly (*args, **kwargs) signature
    is_generic = (
        len(params) == 2 and
        params[0].kind == Parameter.VAR_POSITIONAL and
        params[1].kind == Parameter.VAR_KEYWORD
    )
    
    return is_generic and not hasattr(ingress, '__signature__')


# In Wrap.__init__:
if _should_preserve_signature(ingress, func, preserve_signature):
    ingress.__signature__ = inspect.signature(func)
    # Also preserve annotations for better type checking
    ingress.__annotations__ = getattr(func, '__annotations__', {})
```

**Pros:**
- Eliminates boilerplate in 90% of decorator implementations
- Matches user expectations (decorators should preserve signatures)
- 'auto' mode is safe and smart

**Cons:**
- Adds magic/implicit behavior (mitigated by 'auto' default)
- May confuse users who intentionally want different signatures
- Backward compatibility: existing code might rely on current behavior

**Verdict:** ✅ Critical improvement. The 'auto' mode provides safety.

**Breaking Change Note:** 
Document in code comments that in a future major version (e.g., 3.0), `preserve_signature=True` might become the default. Current behavior preserved with `preserve_signature='auto'` as default.

---

### 1.2 Smart Return Annotation Handling 🔴

**Current Problem:**
Return annotations are lost when:
1. No egress is provided (defaults to `empty`)
2. Egress is provided but has no return annotation

**Suggestion:**
Implement fallback chain: `egress_annotation → func_annotation → empty`

```python
# In Wrap.__init__, enhance annotation handling:
def _get_return_annotation(func, egress):
    """Get return annotation with smart fallback."""
    func_return = Sig(func).return_annotation
    
    if egress is None:
        # No egress: preserve func's return annotation
        return func_return if func_return is not Parameter.empty else empty
    
    # Egress provided: check its annotation first
    egress_return = Sig(egress).return_annotation
    if egress_return is not Parameter.empty:
        return egress_return
    
    # Egress has no annotation: fall back to func's annotation
    # This assumes egress doesn't transform the type
    return func_return if func_return is not Parameter.empty else empty

# Usage:
return_annotation = _get_return_annotation(func, egress)
```

**Pros:**
- Intuitive behavior matching user expectations
- Better IDE/type checker support
- No boilerplate needed

**Cons:**
- Could mask intentional removal of annotations
- Dangerous if egress transforms type without annotation (e.g., int → str)

**Suggested Mitigation:**
Add optional warning mode:

```python
def _get_return_annotation(func, egress, warn_on_mismatch=False):
    """Get return annotation with smart fallback."""
    # ... existing logic ...
    
    if warn_on_mismatch and egress is not None:
        # Check if egress might transform the type
        # (This is a heuristic, not foolproof)
        if egress_return is Parameter.empty and func_return is not Parameter.empty:
            import warnings
            warnings.warn(
                f"Egress for {func.__name__} has no return annotation but "
                f"func returns {func_return}. If egress transforms the type, "
                f"consider adding a return annotation to egress.",
                UserWarning,
                stacklevel=3
            )
```

**Verdict:** ✅ Critical improvement with optional safety check.

---

### 1.3 Helper Factory: `ingress_wrapper()` 🟡

**Current Problem:**
Creating signature-preserving ingress functions requires boilerplate every time.

**Suggestion:**
Add helper function:

```python
def ingress_wrapper(func: Callable, transform: Callable[[tuple, dict], tuple[tuple, dict]]) -> Callable:
    """Create an ingress function that transforms arguments while preserving signature.
    
    This is a convenience wrapper that handles signature preservation automatically.
    
    Parameters
    ----------
    func : Callable
        The function whose signature to preserve
    transform : Callable[[tuple, dict], tuple[tuple, dict]]
        Function that takes (args, kwargs) and returns modified (args, kwargs)
    
    Returns
    -------
    Callable
        Ingress function with preserved signature and annotations
        
    Examples
    --------
    >>> def my_transform(args, kwargs):
    ...     # Convert all args to strings
    ...     return tuple(str(a) for a in args), kwargs
    >>> 
    >>> def add(x: int, y: int = 5) -> int:
    ...     return int(x) + int(y)
    >>> 
    >>> ingress = ingress_wrapper(add, my_transform)
    >>> wrapped = Wrap(add, ingress=ingress)
    >>> wrapped("1", "2")  # Now accepts strings
    3
    
    Note
    ----
    With the `preserve_signature='auto'` feature in Wrap, this helper
    becomes less necessary but is still useful for explicit control.
    """
    from functools import wraps
    
    @wraps(func)
    def _ingress(*args, **kwargs):
        return transform(args, kwargs)
    
    # Explicitly preserve signature and annotations
    _ingress.__signature__ = inspect.signature(func)
    _ingress.__annotations__ = getattr(func, '__annotations__', {})
    
    return _ingress
```

**Pros:**
- Encapsulates common pattern
- Reusable across decorators
- Clear separation of concerns

**Cons:**
- Adds API surface
- Less needed if suggestion #1.1 is implemented

**Verdict:** 🟢 Nice to have. Useful for explicit cases.

---

## Part 2: Learning from wrapt - Missing Features

### 2.1 Object Proxy Pattern for Performance ⚠️ (Breaking)

**What wrapt Does:**
Uses a transparent object proxy that:
- Dynamically delegates attribute access via `__getattr__`
- Avoids copying attributes from wrapped to wrapper
- Much faster for method binding scenarios

**Current i2 Approach:**
Copies attributes explicitly in `Wrap.__init__` and updates signature.

**Gap Analysis:**
i2's approach works but:
- Copies more attributes than necessary
- Slower for repeated method calls (though likely not a bottleneck in practice)
- Doesn't support dynamic attribute additions to wrapped function

**Suggestion - Non-Breaking:**
Add an `ObjectProxy` mixin class that Wrap can optionally use:

```python
class ObjectProxyMixin:
    """Mixin to add transparent proxy behavior to wrappers.
    
    This provides wrapt-style transparent delegation of attribute access.
    Can be used when wrapping objects that dynamically add attributes.
    """
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped object."""
        if name.startswith('_self_'):
            # Internal wrapt-style attributes
            raise AttributeError(name)
        return getattr(self.__wrapped__, name)
    
    def __setattr__(self, name, value):
        """Set attributes on wrapper or delegate to wrapped."""
        if name in ('__wrapped__', '_self_ingress', '_self_egress', '__signature__'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__wrapped__, value)
    
    # Note: More special methods would need to be implemented
    # for a full transparent proxy (see wrapt's implementation)
```

**Verdict:** ⚠️ Major undertaking. Document as potential v3.0 feature.

**Breaking Change Note:**
```python
# In code comments:
# TODO(v3.0): Consider refactoring Wrap to use ObjectProxyMixin for
# better attribute delegation performance, similar to wrapt's approach.
# This would be a breaking change as it changes how attributes are
# accessed and may affect user code that relies on specific attribute
# copying behavior.
```

---

### 2.2 Enhanced Descriptor Protocol Support 🟡

**What wrapt Does:**
Implements full descriptor protocol with `__get__`, `__set__`, `__delete__` support, enabling wrapping of properties and descriptors.

**Current i2 Approach:**
Wrap is a callable but not a full descriptor.

**Gap:**
Cannot cleanly wrap:
- Properties
- Custom descriptors
- Descriptor-based decorators (like `@cached_property`)

**Suggestion:**
Add descriptor protocol methods to Wrap:

```python
class Wrap:
    # ... existing code ...
    
    def __get__(self, instance, owner):
        """Support descriptor protocol for method wrapping.
        
        This enables Wrap to work correctly when wrapping methods,
        classmethods, staticmethods, and properties.
        """
        # If wrapped is not a descriptor, just return self
        if not hasattr(self.__wrapped__, '__get__'):
            return self
        
        # Get bound version of wrapped
        bound_wrapped = self.__wrapped__.__get__(instance, owner)
        
        # Return new Wrap with bound wrapped function
        # This preserves the ingress/egress transformations
        if instance is None:
            # Accessed via class, return self
            return self
        
        # Create bound wrapper
        bound_wrapper = self.__class__.__new__(self.__class__)
        bound_wrapper.__wrapped__ = bound_wrapped
        bound_wrapper._self_ingress = self._self_ingress
        bound_wrapper._self_egress = self._self_egress
        # Copy other necessary attributes...
        
        return bound_wrapper
```

**Pros:**
- Enables wrapping properties and descriptors
- More complete Python object model support
- Matches wrapt's capabilities

**Cons:**
- Adds complexity
- May have subtle interactions with existing code
- Need to carefully test with classmethod, staticmethod, property

**Verdict:** 🟡 Important for completeness. Needs careful implementation and testing.

---

### 2.3 Universal Decorator Context (Breaking but Valuable) ⚠️

**What wrapt Does:**
Wrapper functions receive `(wrapped, instance, args, kwargs)` where:
- `wrapped`: The original function
- `instance`: The instance for method calls (None for functions/classmethods get class)
- `args, kwargs`: The call arguments

This enables "universal decorators" that work correctly across all contexts.

**Current i2 Approach:**
Ingress/egress functions don't receive instance context.

**Gap:**
Cannot easily write decorators that need to know:
- Whether they're wrapping a function vs method
- What instance the method is bound to
- What class the classmethod belongs to

**Suggestion - Non-Breaking:**
Add optional `context_aware=False` parameter:

```python
class Wrap:
    def __init__(
        self,
        func,
        *,
        ingress=None,
        egress=None,
        context_aware=False,  # NEW
        # ... other params
    ):
        """
        Parameters
        ----------
        context_aware : bool, default False
            If True, ingress and egress receive an additional 'instance' parameter
            before args and kwargs, enabling universal decorator patterns.
            
            - For functions: instance = None
            - For methods: instance = the bound instance
            - For classmethods: instance = the class
            - For staticmethods: instance = None
            
            When True, signatures change to:
            - ingress(instance, *args, **kwargs) -> (args, kwargs)
            - egress(instance, output) -> output
        """
```

**Usage Example:**

```python
def context_aware_ingress(instance, *args, **kwargs):
    if instance is None:
        print("Called as function or staticmethod")
    elif inspect.isclass(instance):
        print(f"Called as classmethod of {instance}")
    else:
        print(f"Called as method of {type(instance)}")
    return args, kwargs

@Wrap(ingress=context_aware_ingress, context_aware=True)
def my_func(x):
    return x * 2
```

**Pros:**
- Enables universal decorator patterns
- Backward compatible (opt-in)
- Matches wrapt's powerful context awareness

**Cons:**
- Changes ingress/egress signature when enabled
- Adds complexity to Wrap implementation
- Need to handle binding in `__get__`

**Verdict:** 🟢 Powerful feature for advanced use cases. Consider for v2.x.

**Breaking Change Note:**
```python
# In code comments:
# NOTE(v3.0): The context_aware pattern is powerful but adds complexity.
# In v3.0, we might consider making context_aware the default behavior
# with a compatibility mode for old-style ingress/egress functions.
# See wrapt's decorator pattern for inspiration.
```

---

### 2.4 Synchronized Decorator Pattern 🟢

**What wrapt Provides:**
`@synchronized` decorator for automatic thread-safe locking.

**Current i2:**
No built-in synchronization primitives.

**Suggestion:**
Add as separate utility (not part of core Wrap):

```python
# In i2/wrapper.py or i2/decorators.py
import threading
from functools import wraps

def synchronized(func):
    """Thread-safe decorator using automatic locking.
    
    Automatically creates and manages a lock for the wrapped function.
    For instance methods, the lock is per-instance.
    For classmethods, the lock is per-class.
    For functions, the lock is per-function.
    
    Examples
    --------
    >>> @synchronized
    ... def critical_section():
    ...     # This code is thread-safe
    ...     pass
    
    >>> class Counter:
    ...     def __init__(self):
    ...         self.count = 0
    ...     
    ...     @synchronized
    ...     def increment(self):
    ...         self.count += 1
    
    Note
    ----
    This is a simple implementation. For production use, consider
    using wrapt.synchronized which handles more edge cases.
    """
    # Get or create a lock for this function
    if not hasattr(func, '_sync_lock'):
        func._sync_lock = threading.RLock()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        with func._sync_lock:
            return func(*args, **kwargs)
    
    return wrapper
```

**Pros:**
- Useful utility for concurrent code
- Simple to implement basic version
- Common enough pattern to warrant inclusion

**Cons:**
- Thread synchronization is complex
- Better to point users to wrapt for production use
- Not core to wrapping functionality

**Verdict:** 🟢 Nice utility but optional. Document as "see wrapt.synchronized for production use."

---

## Part 3: Documentation and Developer Experience

### 3.1 Enhanced Documentation with Common Patterns 🔴

**Current Problem:**
Docstring is comprehensive but doesn't show how to avoid common pitfalls.

**Suggestion:**
Add "Common Patterns" section to Wrap docstring:

```python
class Wrap:
    """
    ... existing docstring ...
    
    Common Patterns and Best Practices
    -----------------------------------
    
    **Pattern 1: Transform inputs while preserving signature**
    
    By default (with preserve_signature='auto'), Wrap automatically preserves
    signatures when your ingress uses (*args, **kwargs):
    
    >>> def uppercase_args(func):
    ...     def ingress(*args, **kwargs):
    ...         args = tuple(str(a).upper() for a in args)
    ...         return args, kwargs
    ...     return Wrap(func, ingress=ingress)
    >>> 
    >>> @uppercase_args
    ... def greet(name: str, greeting: str = "Hello") -> str:
    ...     return f"{greeting}, {name}!"
    >>> 
    >>> greet("alice")  # Signature preserved, input transformed
    'Hello, ALICE!'
    
    **Pattern 2: Keep return annotation with transparent egress**
    
    When using an egress that doesn't transform the type, return annotations
    are automatically preserved:
    
    >>> def add_logging(func):
    ...     def egress(output):
    ...         print(f"Result: {output}")
    ...         return output  # Type unchanged
    ...     return Wrap(func, egress=egress)
    >>> 
    >>> @add_logging
    ... def calculate(x: int) -> int:
    ...     return x * 2
    >>> 
    >>> result: int = calculate(5)  # Return type preserved
    Result: 10
    
    **Pattern 3: Using Sig for signature manipulation**
    
    For complex signature transformations, use Sig:
    
    >>> from i2.signatures import Sig
    >>> 
    >>> def add_context_param(func):
    ...     sig = Sig(func)
    ...     new_sig = Sig(['context']) + sig  # Add 'context' parameter
    ...     
    ...     def ingress(context, *args, **kwargs):
    ...         # Use context somehow
    ...         return args, kwargs
    ...     
    ...     return Wrap(func, ingress=ingress, outer_sig=new_sig)
    
    **Pattern 4: Validation without transformation**
    
    Use ingress for validation without modifying arguments:
    
    >>> def validate_positive(func):
    ...     def ingress(*args, **kwargs):
    ...         if any(a <= 0 for a in args if isinstance(a, (int, float))):
    ...             raise ValueError("All numeric arguments must be positive")
    ...         return args, kwargs
    ...     return Wrap(func, ingress=ingress)
    
    **Pattern 5: Error handling and logging**
    
    Wrap both ends for comprehensive error handling:
    
    >>> def safe_call(func):
    ...     def ingress(*args, **kwargs):
    ...         print(f"Calling {func.__name__} with {args}, {kwargs}")
    ...         return args, kwargs
    ...     
    ...     def egress(output):
    ...         print(f"Success: {output}")
    ...         return output
    ...     
    ...     return Wrap(func, ingress=ingress, egress=egress)
    
    Common Pitfalls
    ---------------
    
    **Pitfall 1: Losing signatures with explicit signatures**
    
    If you explicitly set outer_sig, make sure it matches your ingress:
    
    >>> # WRONG: outer_sig doesn't match ingress behavior
    >>> def broken(func):
    ...     def ingress(x):  # Takes one arg
    ...         return (x,), {}
    ...     return Wrap(func, ingress=ingress, outer_sig='x y')  # Declares two!
    
    **Pitfall 2: Type-changing egress without annotation**
    
    If your egress changes the type, annotate it:
    
    >>> # GOOD: Egress annotated
    >>> def stringify(func):
    ...     def egress(output) -> str:
    ...         return str(output)
    ...     return Wrap(func, egress=egress)
    
    **Pitfall 3: Forgetting to call wrapped function**
    
    Your ingress/egress MUST preserve the call path:
    
    >>> # WRONG: ingress doesn't return (args, kwargs)
    >>> def broken_ingress(*args, **kwargs):
    ...     print("called")
    ...     return None  # WRONG!
    >>> 
    >>> # RIGHT:
    >>> def correct_ingress(*args, **kwargs):
    ...     print("called")
    ...     return args, kwargs  # CORRECT
    
    See Also
    --------
    Ingress : For signature-transformation-focused wrapping
    Sig : For signature manipulation
    i2.signatures : Full signature algebra
    """
```

**Verdict:** ✅ Critical for usability. Good documentation prevents bugs.

---

### 3.2 Validation and Helpful Error Messages 🟡

**Current Problem:**
When signature preservation fails or is misconfigured, errors are cryptic.

**Suggestion:**
Add validation mode with clear errors:

```python
class Wrap:
    def __init__(
        self,
        func,
        *,
        ingress=None,
        egress=None,
        validate=None,  # NEW: None (auto) | True | False | 'strict'
        # ... other params
    ):
        """
        Parameters
        ----------
        validate : None | bool | 'strict', default None
            Validation level for wrapper configuration:
            - None: Auto (warn in development, silent in production)
            - False: No validation
            - True: Validate with warnings
            - 'strict': Validate and raise errors
        """
        
        # ... existing init code ...
        
        if validate is None:
            # Auto: Check if in development mode
            validate = __debug__  # False with python -O
        
        if validate:
            self._validate_configuration(func, ingress, egress, strict=validate=='strict')
    
    def _validate_configuration(self, func, ingress, egress, strict=False):
        """Validate wrapper configuration and warn/error on issues."""
        issues = []
        
        # Check 1: Generic ingress without signature preservation
        if ingress is not None:
            ingress_sig = inspect.signature(ingress)
            params = list(ingress_sig.parameters.values())
            
            is_generic = (
                len(params) == 2 and
                params[0].kind == Parameter.VAR_POSITIONAL and
                params[1].kind == Parameter.VAR_KEYWORD
            )
            
            if is_generic and not hasattr(ingress, '__signature__'):
                issues.append(
                    f"Ingress for {func.__name__} has generic (*args, **kwargs) "
                    f"signature but no __signature__ attribute. "
                    f"This will lose the original function's signature. "
                    f"Consider using preserve_signature=True or manually setting "
                    f"ingress.__signature__ = inspect.signature(func)"
                )
        
        # Check 2: Type-changing egress without annotation
        if egress is not None:
            func_return = Sig(func).return_annotation
            egress_return = Sig(egress).return_annotation
            
            if (func_return is not Parameter.empty and 
                egress_return is Parameter.empty):
                issues.append(
                    f"Egress for {func.__name__} has no return annotation, "
                    f"but wrapped function returns {func_return}. "
                    f"If egress transforms the type, add a return annotation to egress."
                )
        
        # Check 3: Ingress return value signature
        if ingress is not None:
            # Check if ingress returns (args, kwargs) tuple
            # This is hard to check statically, so just document it
            pass
        
        # Report issues
        if issues:
            message = "\n".join([f"  - {issue}" for issue in issues])
            full_message = f"Wrapper configuration issues for {func.__name__}:\n{message}"
            
            if strict:
                raise ValueError(full_message)
            else:
                import warnings
                warnings.warn(full_message, UserWarning, stacklevel=3)
```

**Pros:**
- Catches configuration errors early
- Educational value for users
- Configurable strictness

**Cons:**
- Adds runtime overhead (mitigated by validate=None)
- Can produce false positives
- Warnings can be noisy

**Verdict:** 🟡 Good addition with opt-in/auto mode.

---

### 3.3 Pre-configured Decorator Factories 🟢

**Current Problem:**
Users need to understand Wrap internals for common scenarios.

**Suggestion:**
Provide factory functions for common patterns:

```python
# In i2/wrapper.py or i2/decorators.py

def input_transformer(transform: Callable[[tuple, dict], tuple[tuple, dict]]):
    """Create a decorator that transforms inputs while preserving signature.
    
    This is a convenience factory for the common pattern of transforming
    function inputs without changing the signature.
    
    Parameters
    ----------
    transform : Callable[[tuple, dict], tuple[tuple, dict]]
        Function that takes (args, kwargs) and returns modified (args, kwargs)
    
    Returns
    -------
    Callable
        Decorator function
    
    Examples
    --------
    >>> @input_transformer(lambda args, kw: (tuple(str(a) for a in args), kw))
    ... def add(x: int, y: int = 5) -> int:
    ...     return int(x) + int(y)
    >>> 
    >>> add("10", "20")  # Accepts strings, converts to ints
    30
    """
    def decorator(func):
        def ingress(*args, **kwargs):
            return transform(args, kwargs)
        return Wrap(func, ingress=ingress, preserve_signature='auto')
    return decorator


def output_transformer(transform: Callable):
    """Create a decorator that transforms outputs while preserving return type.
    
    Parameters
    ----------
    transform : Callable
        Function that takes the output and returns transformed output.
        Should have matching return annotation if it changes the type.
    
    Returns
    -------
    Callable
        Decorator function
    
    Examples
    --------
    >>> @output_transformer(lambda x: str(x))
    ... def calculate(x: int) -> str:  # Note: return type updated
    ...     return x * 2
    >>> 
    >>> calculate(5)
    '10'
    """
    def decorator(func):
        return Wrap(func, egress=transform)
    return decorator


def validator(**param_validators):
    """Create a decorator that validates inputs before calling function.
    
    Parameters
    ----------
    **param_validators : dict[str, Callable[[Any], bool]]
        Validators for each parameter. Validator should return True if valid.
    
    Returns
    -------
    Callable
        Decorator function
    
    Examples
    --------
    >>> @validator(x=lambda x: x > 0, y=lambda y: y > 0)
    ... def add(x: int, y: int = 5) -> int:
    ...     return x + y
    >>> 
    >>> add(10, 20)
    30
    >>> add(-10, 20)
    Traceback (most recent call last):
        ...
    ValueError: Validation failed for parameter 'x'
    """
    def decorator(func):
        sig = Sig(func)
        
        def ingress(*args, **kwargs):
            # Bind args to parameters
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each parameter
            for param_name, validator_func in param_validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator_func(value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' "
                            f"with value {value!r}"
                        )
            
            return args, kwargs
        
        return Wrap(func, ingress=ingress, preserve_signature='auto')
    return decorator


def cached(maxsize=128):
    """Create a decorator that caches function results.
    
    Similar to functools.lru_cache but implemented using Wrap.
    
    Parameters
    ----------
    maxsize : int, default 128
        Maximum cache size
    
    Examples
    --------
    >>> @cached(maxsize=100)
    ... def fibonacci(n):
    ...     if n < 2:
    ...         return n
    ...     return fibonacci(n-1) + fibonacci(n-2)
    """
    from functools import lru_cache
    
    def decorator(func):
        # Use functools.lru_cache as the implementation
        cached_func = lru_cache(maxsize=maxsize)(func)
        # Wrap it to preserve Wrap interface
        return Wrap(cached_func, preserve_signature=True)
    return decorator
```

**Pros:**
- Makes common patterns trivial to use
- Reduces learning curve
- Demonstrates best practices

**Cons:**
- Adds more API surface
- Need to maintain these factories
- Documentation overhead

**Verdict:** 🟢 Great for usability. Start with 2-3 common patterns.

---

## Part 4: Advanced Patterns from wrapt

### 4.1 Monkey Patching Utilities 🟢

**What wrapt Provides:**
Safe utilities for runtime patching of modules and classes.

**Suggestion:**
Out of scope for wrapper.py core, but could be separate module:

```python
# Future: i2/patching.py
# Tools for safe monkey patching
# See wrapt.wrap_function_wrapper for inspiration
```

**Verdict:** 🟢 Useful but separate concern.

---

### 4.2 Post-Import Hooks 🟢

**What wrapt Provides:**
`@when_imported` decorator for applying patches after module import.

**Suggestion:**
Out of scope for wrapper.py.

**Verdict:** 🟢 Different problem domain.

---

### 4.3 LazyObjectProxy 🟢

**What wrapt Provides:**
Proxy that defers object creation until first access.

**Suggestion:**
Could be useful utility in separate module:

```python
# Future: i2/lazy.py
# See wrapt.LazyObjectProxy for inspiration
```

**Verdict:** 🟢 Useful pattern but separate concern.

---

## Part 5: Implementation Priorities

### Phase 1: Critical (v2.1 - Backward Compatible) 🔴

1. **Automatic Signature Preservation** (#1.1)
   - Add `preserve_signature='auto'` parameter
   - Implement smart detection
   - Add tests

2. **Smart Return Annotation Handling** (#1.2)
   - Implement fallback chain
   - Add optional warning mode
   - Add tests

3. **Enhanced Documentation** (#3.1)
   - Add "Common Patterns" section
   - Add "Common Pitfalls" section
   - Add more examples

### Phase 2: Important (v2.2 - Minor BC Risk) 🟡

4. **Validation and Error Messages** (#3.2)
   - Add validate parameter
   - Implement helpful warnings
   - Test with common mistakes

5. **Helper Factories** (#1.3 and #3.3)
   - Add `ingress_wrapper()`
   - Add common decorator factories
   - Document patterns

6. **Enhanced Descriptor Support** (#2.2)
   - Implement `__get__` method
   - Test with classmethods, staticmethods
   - Handle edge cases

### Phase 3: Enhancements (v2.3+) 🟢

7. **Context-Aware Decorators** (#2.3)
   - Add `context_aware` parameter
   - Implement instance passing
   - Add universal decorator examples

8. **Synchronization Utilities** (#2.4)
   - Add `@synchronized` decorator
   - Reference wrapt for production use
   - Basic thread-safety examples

9. **Additional Utilities** (#4.1-4.3)
   - Consider separate modules
   - Evaluate user demand
   - Reference existing solutions

### Phase 4: Breaking Changes (v3.0) ⚠️

10. **Object Proxy Refactor** (#2.1)
    - Major refactor of Wrap internals
    - Transparent attribute delegation
    - Performance optimization

11. **Default Behavior Changes**
    - Consider making `preserve_signature=True` default
    - Consider making `context_aware=True` default
    - Provide migration guide

---

## Part 6: Code Comments for Breaking Changes

For features that would benefit from breaking changes in v3.0, add these comments to the code:

```python
class Wrap:
    def __init__(
        self,
        func,
        *,
        ingress=None,
        egress=None,
        preserve_signature='auto',  # NOTE(v3.0): Consider making True the default
        context_aware=False,  # NOTE(v3.0): Consider making True the default
        # ...
    ):
        """
        ...existing docstring...
        
        Backward Compatibility Notes
        -----------------------------
        
        **Signature Preservation (v3.0):**
        Currently, preserve_signature defaults to 'auto' which only preserves
        signatures for generic (*args, **kwargs) ingress functions. In v3.0,
        we may change the default to True to always preserve signatures unless
        explicitly disabled. This matches user expectations that decorators
        should preserve signatures by default.
        
        To prepare for this change:
        - If you want current behavior: explicitly set preserve_signature='auto'
        - If you want v3.0 behavior: explicitly set preserve_signature=True
        
        **Context Awareness (v3.0):**
        Currently, ingress and egress don't receive instance information.
        This limits the ability to write universal decorators. In v3.0, we
        may change the ingress/egress signature to include instance context:
        
        Current: ingress(*args, **kwargs) -> (args, kwargs)
        Future:  ingress(instance, *args, **kwargs) -> (args, kwargs)
        
        The context_aware parameter allows opt-in to this behavior now.
        
        **Object Proxy Pattern (v3.0):**
        Current implementation copies attributes from wrapped to wrapper.
        In v3.0, we may switch to a transparent proxy pattern (like wrapt)
        for better performance and more complete delegation. This would be
        a breaking change if you rely on specific attribute access patterns.
        """
        # NOTE(v3.0): Consider refactoring to use ObjectProxyMixin
        # for transparent attribute delegation. This would provide:
        # - Better performance for method binding
        # - Support for dynamic attribute additions
        # - More complete Python object model support
        # See wrapt's implementation for reference.
        
        # ... existing implementation ...
```

---

## Part 7: Testing Strategy

### New Tests Required

1. **Signature Preservation Tests:**
```python
def test_auto_signature_preservation():
    """Test that generic ingress auto-preserves signature."""
    def my_func(x: int, y: int = 5) -> int:
        return x + y
    
    def ingress(*args, **kwargs):
        return args, kwargs
    
    wrapped = Wrap(my_func, ingress=ingress, preserve_signature='auto')
    
    # Signature should be preserved
    assert inspect.signature(wrapped) == inspect.signature(my_func)
    
def test_explicit_signature_preservation():
    """Test explicit preservation mode."""
    def my_func(x: int) -> int:
        return x * 2
    
    def ingress(x):  # Non-generic signature
        return (x,), {}
    
    wrapped = Wrap(my_func, ingress=ingress, preserve_signature=True)
    assert inspect.signature(wrapped) == inspect.signature(my_func)
```

2. **Return Annotation Tests:**
```python
def test_return_annotation_fallback():
    """Test return annotation preservation without egress."""
    def my_func(x: int) -> int:
        return x * 2
    
    wrapped = Wrap(my_func)
    
    sig = inspect.signature(wrapped)
    assert sig.return_annotation == int
```

3. **Validation Tests:**
```python
def test_validation_warning():
    """Test that validation mode warns on issues."""
    def my_func(x: int) -> int:
        return x * 2
    
    def ingress(*args, **kwargs):
        # Generic signature without __signature__
        return args, kwargs
    
    with pytest.warns(UserWarning):
        Wrap(my_func, ingress=ingress, validate=True, preserve_signature=False)
```

---

## Summary Table

| Feature | Priority | Breaking? | Effort | Impact |
|---------|----------|-----------|--------|--------|
| Auto signature preservation | 🔴 Critical | No | Medium | High |
| Smart return annotations | 🔴 Critical | No | Low | High |
| Enhanced documentation | 🔴 Critical | No | Medium | High |
| Validation & errors | 🟡 Important | No | Medium | Medium |
| ingress_wrapper helper | 🟡 Important | No | Low | Medium |
| Descriptor protocol | 🟡 Important | No | High | Medium |
| Decorator factories | 🟢 Enhancement | No | Medium | Medium |
| Context awareness | 🟢 Enhancement | No | Medium | Low |
| Synchronization utils | 🟢 Enhancement | No | Low | Low |
| Object proxy refactor | ⚠️ Breaking | Yes | Very High | High |

---

## Conclusion

The i2.wrapper module is already powerful, but incorporating these improvements (especially from Phase 1 and 2) will:

1. **Make correct decorators easy to write** by auto-preserving signatures
2. **Prevent common mistakes** through validation and good defaults
3. **Improve developer experience** with better documentation and error messages
4. **Enable advanced patterns** like universal decorators and thread-safety
5. **Maintain backward compatibility** while documenting future improvements

The suggested approach is incremental:
- Phase 1 fixes immediate pain points (backward compatible)
- Phase 2 adds nice-to-have features (minimal BC risk)
- Phase 3 explores advanced patterns (opt-in)
- Phase 4 considers breaking changes for v3.0

All suggestions are backed by real patterns from wrapt and motivated by actual use cases. The implementation can be done gradually without disrupting existing users.

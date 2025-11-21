# Implementation Guide for AI Agent: i2.wrapper Improvements

This document provides concrete, actionable tasks for implementing improvements to `i2.wrapper`. Each task includes clear acceptance criteria, code snippets, and testing requirements.

---

## Phase 1: Critical Improvements (v2.1)

Implement these tasks in order. Each is self-contained and can be tested independently.

---

### Task 1.1: Add `preserve_signature` Parameter ✅ CRITICAL

**Location:** `i2/wrapper.py`, `Wrap.__init__()` method (around line 280-300)

**Objective:** Add automatic signature preservation for generic ingress functions.

**Changes Required:**

1. **Add parameter to `__init__`:**
```python
def __init__(
    self,
    func,
    *,
    ingress=None,
    egress=None,
    name=None,
    doc=None,
    module=None,
    annotations=None,
    preserve_signature='auto',  # NEW PARAMETER
):
    """
    ... existing docstring ...
    
    preserve_signature : {'auto', True, False}, default 'auto'
        Controls signature preservation from the wrapped function:
        
        - 'auto': Automatically preserve if ingress has (*args, **kwargs) signature
        - True: Always preserve signature from func (copies __signature__)
        - False: Don't preserve (use ingress's natural signature)
        
        When signature is preserved, both __signature__ and __annotations__ are
        copied from func to the wrapper, ensuring type checkers and IDEs see
        the original signature.
        
        Examples
        --------
        >>> def my_func(x: int, y: int = 5) -> int:
        ...     return x + y
        >>> 
        >>> def generic_ingress(*args, **kwargs):
        ...     return args, kwargs
        >>> 
        >>> # Auto mode: signature preserved (generic ingress detected)
        >>> wrapped = Wrap(my_func, ingress=generic_ingress)
        >>> str(Sig(wrapped))
        '(x: int, y: int = 5) -> int'
        
        >>> # Explicit preservation even with non-generic ingress
        >>> def specific_ingress(x, y):
        ...     return (x,), {'y': y}
        >>> wrapped = Wrap(my_func, ingress=specific_ingress, preserve_signature=True)
        >>> str(Sig(wrapped))
        '(x: int, y: int = 5) -> int'
    
    .. note::
       In a future major version (v3.0), the default may change to True
       to match user expectations that decorators preserve signatures by default.
    """
```

2. **Add helper function before `__init__`:**
```python
def _should_preserve_signature(ingress, func, preserve_mode):
    """Determine if signature should be auto-preserved from func to ingress.
    
    Parameters
    ----------
    ingress : callable or None
        The ingress function
    func : callable
        The wrapped function
    preserve_mode : 'auto' | True | False
        The preservation mode
    
    Returns
    -------
    bool
        True if signature should be preserved
    """
    if preserve_mode is False:
        return False
    if preserve_mode is True:
        return True
    
    # 'auto' mode: preserve if ingress has generic (*args, **kwargs)
    if ingress is None:
        return False
    
    try:
        ingress_sig = inspect.signature(ingress)
    except (ValueError, TypeError):
        # Can't get signature, don't preserve
        return False
    
    params = list(ingress_sig.parameters.values())
    
    # Check if ingress has exactly (*args, **kwargs) signature
    if len(params) != 2:
        return False
    
    is_generic = (
        params[0].kind == Parameter.VAR_POSITIONAL and
        params[1].kind == Parameter.VAR_KEYWORD
    )
    
    # Only preserve if generic and doesn't already have __signature__
    return is_generic and not hasattr(ingress, '__signature__')
```

3. **Apply preservation in `__init__` (before signature creation):**
```python
# In Wrap.__init__, after setting self.ingress but before creating outer_sig:

if _should_preserve_signature(ingress, func, preserve_signature):
    # Preserve signature and annotations from func to ingress
    self.ingress.__signature__ = inspect.signature(func)
    self.ingress.__annotations__ = getattr(func, '__annotations__', {})
```

**Acceptance Criteria:**
- [ ] `preserve_signature` parameter accepts 'auto', True, False
- [ ] 'auto' mode correctly detects `(*args, **kwargs)` signatures
- [ ] True mode always preserves regardless of ingress signature
- [ ] False mode never preserves
- [ ] Annotations are preserved along with signature
- [ ] Existing behavior unchanged when parameter not specified
- [ ] All existing tests pass

**Tests to Add:**
```python
def test_preserve_signature_auto_generic():
    """Test auto preservation with generic (*args, **kwargs) ingress."""
    def my_func(x: int, y: int = 5) -> int:
        return x + y
    
    def ingress(*args, **kwargs):
        return args, kwargs
    
    wrapped = Wrap(my_func, ingress=ingress)  # Default: preserve_signature='auto'
    
    # Signature should be preserved
    assert str(Sig(wrapped)) == '(x: int, y: int = 5) -> int'
    assert wrapped.__annotations__ == my_func.__annotations__


def test_preserve_signature_auto_non_generic():
    """Test auto mode doesn't preserve non-generic ingress."""
    def my_func(x: int, y: int = 5) -> int:
        return x + y
    
    def ingress(x, y):  # Not generic
        return (x,), {'y': y}
    
    wrapped = Wrap(my_func, ingress=ingress)  # Auto mode
    
    # Should use ingress's signature
    assert str(Sig(wrapped)) == '(x, y)'


def test_preserve_signature_explicit_true():
    """Test explicit True mode always preserves."""
    def my_func(x: int, y: int = 5) -> int:
        return x + y
    
    def ingress(x, y):  # Not generic
        return (x,), {'y': y}
    
    wrapped = Wrap(my_func, ingress=ingress, preserve_signature=True)
    
    # Should preserve despite non-generic ingress
    assert str(Sig(wrapped)) == '(x: int, y: int = 5) -> int'


def test_preserve_signature_explicit_false():
    """Test explicit False mode never preserves."""
    def my_func(x: int, y: int = 5) -> int:
        return x + y
    
    def ingress(*args, **kwargs):  # Generic
        return args, kwargs
    
    wrapped = Wrap(my_func, ingress=ingress, preserve_signature=False)
    
    # Should not preserve despite generic ingress
    assert str(Sig(wrapped)) == '(*args, **kwargs)'
```

---

### Task 1.2: Smart Return Annotation Handling ✅ CRITICAL

**Location:** `i2/wrapper.py`, `Wrap.__init__()` method (around line 336-342)

**Objective:** Preserve return annotations with smart fallback logic.

**Changes Required:**

1. **Add helper function:**
```python
def _get_return_annotation(func, egress):
    """Get return annotation with smart fallback logic.
    
    Fallback chain: egress annotation → func annotation → empty
    
    Parameters
    ----------
    func : callable
        The wrapped function
    egress : callable or None
        The egress function
    
    Returns
    -------
    annotation
        The return annotation to use, or Parameter.empty
    """
    from inspect import Parameter
    
    func_sig = Sig(func)
    func_return = func_sig.return_annotation
    
    if egress is None:
        # No egress: use func's return annotation
        return func_return if func_return is not Parameter.empty else empty
    
    # Egress provided: check its annotation first
    try:
        egress_sig = Sig(egress)
        egress_return = egress_sig.return_annotation
    except (ValueError, TypeError):
        # Can't get egress signature, fall back to func
        return func_return if func_return is not Parameter.empty else empty
    
    if egress_return is not Parameter.empty:
        # Egress has annotation, use it
        return egress_return
    
    # Egress has no annotation: fall back to func's annotation
    # Assumption: egress doesn't transform the type
    return func_return if func_return is not Parameter.empty else empty
```

2. **Replace return annotation logic in `__init__`:**

Replace this section (around line 336-342):
```python
# OLD CODE:
return_annotation = empty

if egress is None:
    self.egress = transparent_egress
else:
    self.egress = egress
    egress_return_annotation = Sig(egress).return_annotation
    if egress_return_annotation is not Parameter.empty:
        return_annotation = egress_return_annotation
```

With this:
```python
# NEW CODE:
if egress is None:
    self.egress = transparent_egress
else:
    self.egress = egress

# Smart return annotation fallback
return_annotation = _get_return_annotation(func, egress)
```

**Acceptance Criteria:**
- [ ] No egress → uses func's return annotation
- [ ] Egress with annotation → uses egress's annotation
- [ ] Egress without annotation → falls back to func's annotation
- [ ] empty return annotation → returns empty (not None)
- [ ] All existing tests pass

**Tests to Add:**
```python
def test_return_annotation_no_egress():
    """Test return annotation preserved when no egress."""
    def my_func(x: int) -> int:
        return x * 2
    
    wrapped = Wrap(my_func)  # No egress
    
    sig = Sig(wrapped)
    assert sig.return_annotation == int


def test_return_annotation_egress_with_annotation():
    """Test egress annotation takes precedence."""
    def my_func(x: int) -> int:
        return x * 2
    
    def egress(output) -> str:  # Different return type
        return str(output)
    
    wrapped = Wrap(my_func, egress=egress)
    
    sig = Sig(wrapped)
    assert sig.return_annotation == str


def test_return_annotation_egress_without_annotation():
    """Test fallback to func annotation when egress lacks one."""
    def my_func(x: int) -> int:
        return x * 2
    
    def egress(output):  # No annotation
        return output  # Doesn't transform
    
    wrapped = Wrap(my_func, egress=egress)
    
    sig = Sig(wrapped)
    assert sig.return_annotation == int  # Falls back to func's


def test_return_annotation_none_anywhere():
    """Test empty when no annotations exist."""
    def my_func(x):  # No annotation
        return x * 2
    
    def egress(output):  # No annotation
        return output
    
    wrapped = Wrap(my_func, egress=egress)
    
    sig = Sig(wrapped)
    from inspect import Parameter
    assert sig.return_annotation is Parameter.empty
```

---

### Task 1.3: Enhanced Documentation ✅ CRITICAL

**Location:** `i2/wrapper.py`, `Wrap` class docstring

**Objective:** Add comprehensive "Common Patterns" and "Common Pitfalls" sections.

**Changes Required:**

Add this section to the end of the Wrap docstring (before "See Also"):

```python
"""
... existing docstring ...

Common Patterns and Best Practices
-----------------------------------

Pattern 1: Transform inputs while preserving signature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Pattern 2: Keep return annotation with transparent egress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Pattern 3: Using Sig for signature manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
...     ingress.__signature__ = new_sig  # Explicit signature
...     return Wrap(func, ingress=ingress)

Pattern 4: Validation without transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ingress for validation without modifying arguments:

>>> def validate_positive(func):
...     def ingress(*args, **kwargs):
...         if any(a <= 0 for a in args if isinstance(a, (int, float))):
...             raise ValueError("All numeric arguments must be positive")
...         return args, kwargs
...     return Wrap(func, ingress=ingress)
>>> 
>>> @validate_positive
... def multiply(x: int, y: int) -> int:
...     return x * y
>>> 
>>> multiply(2, 3)
6
>>> multiply(-1, 3)
Traceback (most recent call last):
    ...
ValueError: All numeric arguments must be positive

Pattern 5: Error handling and logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Common Pitfalls and Solutions
------------------------------

Pitfall 1: Losing signatures with explicit non-generic ingress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your ingress doesn't use (*args, **kwargs), the auto mode won't preserve
the signature. Use preserve_signature=True explicitly:

>>> # WRONG: Signature lost with non-generic ingress
>>> def broken(func):
...     def ingress(x):  # Specific signature
...         return (x,), {}
...     return Wrap(func, ingress=ingress)  # Auto mode won't preserve!
>>> 
>>> # RIGHT: Explicit preservation
>>> def fixed(func):
...     def ingress(x):
...         return (x,), {}
...     return Wrap(func, ingress=ingress, preserve_signature=True)

Pitfall 2: Type-changing egress without annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your egress changes the output type, annotate it. Otherwise, the function's
original return type will be preserved, creating incorrect type hints:

>>> # WRONG: Egress changes type but doesn't annotate
>>> def broken_stringify(func):
...     def egress(output):  # No annotation
...         return str(output)  # Changes int -> str
...     return Wrap(func, egress=egress)
>>> 
>>> @broken_stringify
... def calc(x: int) -> int:  # Says returns int
...     return x * 2
>>> # But actually returns str! Type checkers will be confused.
>>> 
>>> # RIGHT: Egress annotated with correct return type
>>> def fixed_stringify(func):
...     def egress(output) -> str:  # Annotated!
...         return str(output)
...     return Wrap(func, egress=egress)
>>> 
>>> @fixed_stringify
... def calc(x: int) -> str:  # Now correctly says returns str
...     return x * 2

Pitfall 3: Forgetting to return (args, kwargs) from ingress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ingress MUST return a tuple of (args, kwargs) for the wrapped function:

>>> # WRONG: ingress doesn't return (args, kwargs)
>>> def broken_ingress(*args, **kwargs):
...     print("called")
...     return None  # WRONG! Must return (args, kwargs)
>>> 
>>> # RIGHT: Always return (args, kwargs)
>>> def correct_ingress(*args, **kwargs):
...     print("called")
...     return args, kwargs  # CORRECT

Pitfall 4: Modifying mutable arguments in place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Be careful when modifying arguments - changes affect the original objects:

>>> # WRONG: Modifies original list
>>> def broken(func):
...     def ingress(*args, **kwargs):
...         if args and isinstance(args[0], list):
...             args[0].append(999)  # Modifies original!
...         return args, kwargs
...     return Wrap(func, ingress=ingress)
>>> 
>>> # RIGHT: Create new objects
>>> def fixed(func):
...     def ingress(*args, **kwargs):
...         if args and isinstance(args[0], list):
...             args = ([*args[0], 999],) + args[1:]  # New list
...         return args, kwargs
...     return Wrap(func, ingress=ingress)

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
- If you never want preservation: explicitly set preserve_signature=False

**Context Awareness (v3.0):**

Currently, ingress and egress don't receive instance information, limiting
the ability to write universal decorators that work differently for functions,
methods, classmethods, etc. In v3.0, we may add an optional context_aware
mode that changes the signature to include instance context.

This would be opt-in initially, potentially becoming default in v3.0.

See Also
--------
... existing See Also ...
"""
```

**Acceptance Criteria:**
- [ ] "Common Patterns" section with 5 complete examples
- [ ] "Common Pitfalls" section with 4 anti-patterns and fixes
- [ ] "Backward Compatibility Notes" for v3.0 changes
- [ ] All examples are runnable doctests
- [ ] Examples cover the most common use cases

**Tests:**
The examples themselves serve as doctests. Run:
```bash
python -m doctest i2/wrapper.py
```

---

## Phase 2: Important Enhancements (v2.2)

Implement after Phase 1 is complete and tested.

---

### Task 2.1: Add Validation Mode

**Location:** `i2/wrapper.py`, `Wrap.__init__()` method

**Objective:** Warn users about common configuration issues.

**Changes Required:**

1. **Add parameter:**
```python
def __init__(
    self,
    func,
    *,
    ingress=None,
    egress=None,
    preserve_signature='auto',
    validate=None,  # NEW: None (auto) | True | False | 'strict'
    # ... other params
):
    """
    ... existing docstring ...
    
    validate : None | bool | 'strict', default None
        Validation level for wrapper configuration:
        
        - None: Auto mode - validate with warnings in development (__debug__=True),
          silent in production (__debug__=False)
        - False: No validation
        - True: Validate with warnings
        - 'strict': Validate and raise errors on issues
        
        Checks for:
        - Generic ingress without signature preservation
        - Type-changing egress without annotation
        - Common configuration mistakes
        
        Examples
        --------
        >>> def my_func(x: int) -> int:
        ...     return x * 2
        >>> 
        >>> def ingress(*args, **kwargs):
        ...     return args, kwargs
        >>> 
        >>> # Auto mode: warns in development, silent in production
        >>> wrapped = Wrap(my_func, ingress=ingress, preserve_signature=False)
        >>> # May warn: "Generic ingress without signature preservation"
        
        >>> # Strict mode: raises error
        >>> wrapped = Wrap(my_func, ingress=ingress, 
        ...                preserve_signature=False, validate='strict')
        Traceback (most recent call last):
            ...
        ValueError: Wrapper configuration issues for my_func:
          - Generic ingress without signature preservation
    """
```

2. **Add validation method:**
```python
def _validate_configuration(self, func, ingress, egress, preserve_signature, strict=False):
    """Validate wrapper configuration and warn/error on issues.
    
    Parameters
    ----------
    func : callable
        The wrapped function
    ingress : callable or None
        The ingress function
    egress : callable or None
        The egress function
    preserve_signature : 'auto' | True | False
        Signature preservation mode
    strict : bool
        If True, raise error instead of warning
    """
    issues = []
    
    # Check 1: Generic ingress without signature preservation
    if ingress is not None and preserve_signature is False:
        try:
            ingress_sig = inspect.signature(ingress)
            params = list(ingress_sig.parameters.values())
            
            is_generic = (
                len(params) == 2 and
                params[0].kind == Parameter.VAR_POSITIONAL and
                params[1].kind == Parameter.VAR_KEYWORD
            )
            
            if is_generic:
                issues.append(
                    f"Ingress has generic (*args, **kwargs) signature but "
                    f"preserve_signature=False. This will lose the original "
                    f"function's signature. Consider preserve_signature='auto' or True."
                )
        except (ValueError, TypeError):
            pass  # Can't check, skip
    
    # Check 2: Type-changing egress without annotation
    if egress is not None:
        try:
            func_return = Sig(func).return_annotation
            egress_return = Sig(egress).return_annotation
            
            if (func_return is not Parameter.empty and 
                egress_return is Parameter.empty):
                # Heuristic: check if egress might be transforming
                # (This is imperfect but catches common cases)
                egress_source = inspect.getsource(egress)
                if 'str(' in egress_source or 'int(' in egress_source:
                    issues.append(
                        f"Egress has no return annotation, but appears to "
                        f"transform the output. If egress changes the type, "
                        f"add a return annotation to egress."
                    )
        except (ValueError, TypeError, OSError):
            pass  # Can't check, skip
    
    # Report issues
    if issues:
        message = "\n".join([f"  - {issue}" for issue in issues])
        full_message = f"Wrapper configuration issues for {func.__name__}:\n{message}"
        
        if strict:
            raise ValueError(full_message)
        else:
            import warnings
            warnings.warn(full_message, UserWarning, stacklevel=4)
```

3. **Call validation in `__init__`:**
```python
# At end of __init__, before return:
if validate is None:
    # Auto: Check if in development mode
    validate_mode = __debug__  # False with python -O
else:
    validate_mode = validate

if validate_mode:
    self._validate_configuration(
        func, ingress, egress, preserve_signature,
        strict=(validate_mode == 'strict')
    )
```

**Tests to Add:**
```python
import warnings

def test_validate_generic_ingress_warning():
    """Test warning for generic ingress without preservation."""
    def my_func(x: int) -> int:
        return x * 2
    
    def ingress(*args, **kwargs):
        return args, kwargs
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Wrap(my_func, ingress=ingress, preserve_signature=False, validate=True)
        assert len(w) == 1
        assert "Generic ingress" in str(w[0].message)


def test_validate_strict_mode_raises():
    """Test strict mode raises instead of warning."""
    def my_func(x: int) -> int:
        return x * 2
    
    def ingress(*args, **kwargs):
        return args, kwargs
    
    with pytest.raises(ValueError, match="Wrapper configuration issues"):
        Wrap(my_func, ingress=ingress, preserve_signature=False, validate='strict')


def test_validate_false_no_warning():
    """Test validate=False suppresses warnings."""
    def my_func(x: int) -> int:
        return x * 2
    
    def ingress(*args, **kwargs):
        return args, kwargs
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Wrap(my_func, ingress=ingress, preserve_signature=False, validate=False)
        assert len(w) == 0
```

---

### Task 2.2: Add `ingress_wrapper()` Helper

**Location:** `i2/wrapper.py`, after `Wrap` class definition

**Objective:** Provide convenience helper for common pattern.

**Implementation:**
```python
def ingress_wrapper(func: Callable, transform: Callable[[tuple, dict], tuple[tuple, dict]]) -> Callable:
    """Create an ingress function that transforms arguments while preserving signature.
    
    This is a convenience wrapper that handles signature preservation automatically.
    It's most useful when you want explicit control over signature preservation,
    though the `preserve_signature='auto'` feature in Wrap reduces the need for this.
    
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
    With the `preserve_signature='auto'` feature in Wrap (default behavior),
    this helper is less necessary since Wrap will automatically preserve
    signatures for generic (*args, **kwargs) ingress functions. However, it
    remains useful when you want explicit, guaranteed preservation or when
    creating reusable ingress functions.
    
    See Also
    --------
    Wrap : The main wrapper class that can auto-preserve signatures
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

**Tests to Add:**
```python
def test_ingress_wrapper_basic():
    """Test basic ingress_wrapper functionality."""
    def my_func(x: int, y: int = 5) -> int:
        return x + y
    
    def transform(args, kwargs):
        # Double all args
        return tuple(a * 2 for a in args), kwargs
    
    ingress = ingress_wrapper(my_func, transform)
    wrapped = Wrap(my_func, ingress=ingress)
    
    # Signature preserved
    assert str(Sig(wrapped)) == '(x: int, y: int = 5) -> int'
    
    # Transform applied
    assert wrapped(2, 3) == 10  # (2*2) + (3*2) = 10


def test_ingress_wrapper_preserves_annotations():
    """Test that annotations are preserved."""
    def my_func(x: int, y: str = "default") -> str:
        return f"{x}: {y}"
    
    ingress = ingress_wrapper(my_func, lambda a, k: (a, k))
    
    assert ingress.__annotations__ == my_func.__annotations__
```

---

## Testing Guidelines

### Running Tests

After implementing changes:

```bash
# Run all tests
pytest i2/tests/test_wrapper.py -v

# Run specific test
pytest i2/tests/test_wrapper.py::test_preserve_signature_auto_generic -v

# Run doctests
python -m doctest i2/wrapper.py -v
```

### Test Coverage

Aim for:
- Line coverage: >95% for modified code
- Branch coverage: >90% for new conditionals
- Edge cases: Test None, empty, invalid inputs

```bash
# Run with coverage
pytest --cov=i2.wrapper --cov-report=html
```

---

## Documentation Updates

After implementation:

1. **Update CHANGELOG.md:**
```markdown
## [2.1.0] - 2025-XX-XX

### Added
- Automatic signature preservation with `preserve_signature='auto'` parameter
- Smart return annotation fallback logic
- Comprehensive "Common Patterns" and "Common Pitfalls" documentation
- Validation mode with helpful warnings
- `ingress_wrapper()` helper function

### Changed
- Wrap now automatically preserves signatures for generic (*args, **kwargs) ingress
- Return annotations are preserved even when egress has no annotation

### Deprecated
- None

### Fixed
- Signature loss when using generic ingress functions
- Return annotation loss when using transparent egress
```

2. **Update README if needed**
3. **Create migration guide for users**

---

## Deployment Checklist

Before release:

- [ ] All Phase 1 tasks implemented
- [ ] All tests passing
- [ ] Doctest examples working
- [ ] CHANGELOG updated
- [ ] README updated (if needed)
- [ ] Documentation built without errors
- [ ] Version number bumped to 2.1.0
- [ ] Git tag created
- [ ] PyPI package built and tested locally
- [ ] Package uploaded to PyPI

---

## Notes for Implementation

### Backward Compatibility

All Phase 1 and Phase 2 changes are backward compatible:
- New parameters have safe defaults
- Existing code continues to work unchanged
- No breaking changes to public API

### Performance Considerations

The added logic is minimal:
- Signature checks are done once at decoration time (not per call)
- No performance impact on wrapped function calls
- Validation can be disabled in production (validate=False)

### Edge Cases to Test

- Functions with no signature (C functions, builtins)
- Functions with complex signatures (positional-only, keyword-only, etc.)
- Nested decorators
- Methods, classmethods, staticmethods
- Generators and coroutines
- Partial functions

---

## Questions/Issues?

If you encounter any issues during implementation:
1. Check existing tests for similar patterns
2. Review the wrapt documentation for inspiration
3. Document any deviations from this plan
4. Add comments explaining non-obvious decisions

**Remember:** The goal is to make the correct thing easy, not to add unnecessary complexity.

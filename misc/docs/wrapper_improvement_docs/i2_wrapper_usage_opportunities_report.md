# i2.wrapper Usage Opportunities Report

**Generated:** 2025-11-21  
**Scope:** Python files in `/Users/thorwhalen/Dropbox/py/proj`

## Executive Summary

This report identifies locations where functionality from `i2.wrapper` could be beneficial. The `i2.wrapper` module provides:
- `wrap`, `Wrap`, `Wrapx` - Function wrapping with ingress/egress transformations
- `Ingress` - Template for creating argument transformation decorators
- `include_exclude`, `rm_params` - Parameter filtering utilities
- `partialx` - Extended partial application
- `ch_names` - Parameter name changing
- `func_to_method_func` - Convert functions to method functions
- `bind_funcs_object_attrs` - Bind functions to object attributes
- `kwargs_trans` - Keyword argument transformation

---

## Current Direct Usage

### Files Already Using i2.wrapper

1. **`i/i2/i2/castgraph.py`** (lines 181, 1080)
   - Uses: `Wrap` for ingress/egress decorators in CastGraph
   - Context: Type conversion graph with automatic argument transformation
   ```python
   from i2.wrapper import Wrap
   # ...
   return Wrap(func, ingress=ingress_func, egress=egress_func)
   ```

2. **`c/http_services/http_services/dog/imbed_dog.py`** (line 14)
   - Uses: `ch_names` for parameter renaming
   ```python
   from i2.wrapper import ch_names
   ```

3. **`c/http_services/http_services/experiments/azure_deploy/azure_funcs_02.py`** (line 56)
   - Uses: `Wrap` for Azure function wrapping
   ```python
   from i2 import Wrap
   ```

4. **`c/cosmox/cosmox/ingress_framework.py`** (line 55)
   - Uses: `Ingress`, `kwargs_trans` for custom ingress framework
   ```python
   from i2.wrapper import Ingress as I2Ingress, kwargs_trans
   ```

5. **`t/allude/allude/base.py`** (line 3)
   - Uses: Multiple wrapper utilities
   ```python
   from i2.wrapper import transparent_ingress, transparent_egress, Wrap, wrap
   ```

6. **`t/oa/oa/vector_stores.py`** (lines 74, 104)
   - Uses: `Ingress`, `wrap` for OpenAI vector store wrappers
   ```python
   from i2.wrapper import Ingress, wrap
   ```

7. **`t/plunk/plunk/tw/things_decorators_break.py`** (line 43)
   - Uses: `wrap` for decorator testing
   ```python
   from i2.wrapper import wrap
   ```

8. **`t/plunk/plunk/tw/front_examples/crude_examples/take_09_model_pipe.py`** (line 43)
   - Uses: `rm_params` for parameter removal
   ```python
   from i2.wrapper import rm_params
   ```

9. **`t/theremin/theremin/util.py`** (line 150)
   - Uses: `partialx` for extended partial application
   ```python
   from i2 import partialx, Sig as Signature
   ```

10. **`i/front/front/py2pydantic.py`** (line 32)
    - Uses: `wrap`, `Ingress` for Pydantic model conversion
    ```python
    from i2.wrapper import wrap, Ingress
    ```

11. **`i/front/front/crude.py`** (line 48)
    - Uses: `Ingress`, `wrap` for UI generation
    ```python
    from i2.wrapper import Ingress, wrap
    ```

12. **`i/front/front/util.py`** (lines 11, 127)
    - Uses: `Ingress`, `wrap`, `Wrap` for front-end utilities
    ```python
    from i2.wrapper import Ingress, wrap
    from i2.wrapper import Wrap
    ```

13. **`i/front/front/dag.py`** (line 371)
    - Uses: `rm_params` for DAG parameter management
    ```python
    from i2.wrapper import rm_params
    ```

---

## Potential Opportunities

### Category 1: Functions That Could Benefit from `wrap`/`Ingress`

#### 1.1 Parameter Transformation Patterns

**Location:** `a/raglab/raglab/util.py` (line 207)
```python
def prompt_func_ingress(kwargs: dict) -> dict:
    """Ingress function for prompt functions."""
```
- **Opportunity:** This is already implementing an ingress pattern manually. Could use `Ingress` class from i2.wrapper for more robust implementation.
- **Benefit:** Type safety, automatic signature inference, standardized transformation pipeline

**Location:** `c/_cosmodata/cosmodata/util.py` (line 36)
```python
key_ingress=_graze.graze.key_ingress_print_downloading_message,
```
- **Opportunity:** Custom ingress for key transformation. Could standardize with `i2.wrapper.Ingress`
- **Benefit:** Consistency across codebase, easier testing

#### 1.2 Decorator Patterns

**Files with manual decorator implementations that could use `wrap`:**

1. **`i/uf/uf/decorators.py`** (lines 81-118)
   ```python
   def ui_config(...):
       def decorator(func: Callable) -> Callable:
           @wraps(func)
           def wrapper(*args, **kwargs):
               return func(*args, **kwargs)
   ```
   - **Opportunity:** Replace with `wrap(func, ingress=...)` pattern
   - **Benefit:** Signature preservation, cleaner code

2. **`i/i2/i2/castgraph.py`** (line 289)
   ```python
   def _normalize_converter(func):
       """Wrap a converter function to ensure it accepts (obj, context) signature."""
   ```
   - **Opportunity:** Already wrapping, could use `Ingress` for parameter normalization
   - **Benefit:** Standardization

#### 1.3 Argument Manipulation Patterns

**Files that manually transform arguments:**

1. **`c/dev_utils_for_cosmograph/dev_utils_for_cosmograph/_code_sync.py`** (line 195)
   - Context: Function definition replacement with decorator handling
   - **Opportunity:** Use `ch_names` or `Ingress` for parameter renaming in function transformations

2. **`t/priv/priv/git_ops.py`** (multiple locations: lines 59, 65, 90, 134, etc.)
   ```python
   @project_kinds.ingress.local_proj_folder
   def function_name(...):
       # The @ingress decorator already converted project_ref to local_proj_folder
   ```
   - **Opportunity:** Already using custom ingress pattern. Could be standardized with `i2.wrapper.Ingress`
   - **Benefit:** Better error handling, type hints, documentation

### Category 2: Functions That Could Use `rm_params`

**Pattern:** Functions that need to accept variable arguments but only use a subset

**Locations:**
1. `i/crude/crude/sb/flat_dispatch_app_5.py` (line 140) - Commented out usage
2. `t/plunk_copy/plunk_tw/front_examples/crude_examples/take_06_model_run.py` (line 64) - Commented out
3. `t/plunk_copy/plunk_tw/front_examples/crude_examples/take_05_model_run.py` (line 41) - Commented out

**Opportunity:** Re-enable these commented-out uses or find similar patterns where parameter filtering is needed

### Category 3: Functions That Could Use `partialx`

**Pattern:** Advanced partial application with position/keyword control

**Current Usage:** `t/theremin/theremin/util.py` (line 150)

**Potential Locations:**
- Any file using `functools.partial` could potentially benefit from `partialx`'s enhanced capabilities
- Search for `from functools import partial` to find candidates

### Category 4: Method Binding Opportunities

**Pattern:** Functions that need to be converted to methods or bound to objects

**Tool:** `func_to_method_func`, `bind_funcs_object_attrs`

**Potential Locations:**
1. `i/i2/i2/wrapper.py` (line 170) - Already handling method binding internally
2. Any dynamic class/method generation code

### Category 5: Store Wrapping (dol integration)

**Pattern:** Files using `dol.wrap_kvs` could potentially use i2.wrapper for additional transformations

**Locations:**
1. `c/_cosmodata/cosmodata/util.py` (lines 137, 183, 210, 222, 267)
2. `a/raglab/raglab/web_services/store_access.py` (lines 6, 17)
3. `a/raglab/raglab/retrieval/lib_alexis.py` (lines 6, 36, 369, 371)
4. `a/raglab/raglab/stores/stores_util.py` (lines 18, 51, 112, 125)

**Opportunity:** Combine `dol.wrap_kvs` with `i2.wrapper.Ingress` for more sophisticated store transformations

---

## Recommendations by Priority

### High Priority

1. **Standardize ingress patterns** in `a/raglab` and `c/cosmodata`
   - Replace manual ingress implementations with `i2.wrapper.Ingress`
   - Benefits: Type safety, consistency, easier maintenance

2. **Review commented-out `rm_params` usage** in plunk examples
   - Determine why it was commented out
   - Re-enable if appropriate or document why not

3. **Consolidate decorator patterns** in `i/uf/uf/decorators.py`
   - Replace manual wrapper functions with `i2.wrap`
   - Benefits: Signature preservation, less boilerplate

### Medium Priority

4. **Enhance store wrappers** with Ingress
   - Combine `dol.wrap_kvs` patterns with `i2.wrapper.Ingress`
   - Benefits: More powerful transformations, better error handling

5. **Standardize Git operations** in `t/priv/priv/git_ops.py`
   - Use `i2.wrapper.Ingress` instead of custom ingress decorators
   - Benefits: Consistency with rest of i2 ecosystem

6. **Add `partialx` where appropriate**
   - Find uses of `functools.partial` that could benefit from `partialx`
   - Benefits: More control over argument binding

### Low Priority

7. **Document i2.wrapper patterns** in project documentation
   - Create guidelines for when to use each wrapper tool
   - Benefits: Team education, consistency

8. **Create i2.wrapper examples** specific to common project patterns
   - Store wrapping examples
   - Ingress for type conversion examples
   - Benefits: Faster adoption, fewer bugs

---

## Code Smell Patterns Suggesting Wrapper Usage

Look for these patterns as indicators where `i2.wrapper` could help:

1. **Manual decorator with signature preservation:**
   ```python
   @wraps(func)
   def wrapper(*args, **kwargs):
       # transform args/kwargs
       return func(*transformed_args, **transformed_kwargs)
   ```
   → Use `wrap` or `Ingress`

2. **Parameter name changing:**
   ```python
   def outer_func(old_name):
       return inner_func(new_name=old_name)
   ```
   → Use `ch_names`

3. **Parameter subset extraction:**
   ```python
   def wrapper(**kwargs):
       relevant_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
       return func(**relevant_kwargs)
   ```
   → Use `rm_params` or `include_exclude`

4. **Custom ingress/egress patterns:**
   ```python
   def wrapper(*args):
       transformed_args = preprocess(args)
       result = func(*transformed_args)
       return postprocess(result)
   ```
   → Use `Wrap(func, ingress=..., egress=...)`

5. **Method binding in class definitions:**
   ```python
   class MyClass:
       def method(self):
           return some_function(self.attr)
   ```
   → Consider `bind_funcs_object_attrs`

---

## Statistics

- **Files already using i2.wrapper:** 13 unique files
- **Potential opportunity areas:** ~30+ locations
- **Most common usage:** `wrap` and `Ingress` (for argument transformation)
- **Underutilized tools:** `partialx`, `func_to_method_func`, `bind_funcs_object_attrs`

---

## Next Steps

1. **Audit commented-out wrapper usage** to understand removal reasons
2. **Create migration guide** for converting manual patterns to i2.wrapper
3. **Add linting rules** to detect manual wrapper patterns
4. **Conduct code review** of high-priority opportunities
5. **Create test suite** for wrapper migrations to ensure behavior preservation

---

## Appendix: Full i2.wrapper API

For reference, here are all the tools available in `i2.wrapper`:

```python
from i2.wrapper import (
    wrap,          # Main wrapping function (auto-selects Wrap or Wrapx)
    Wrap,          # Basic wrapper with ingress/egress
    Wrapx,         # Extended wrapper with custom caller
    Ingress,       # Ingress template class
    include_exclude,  # Include/exclude parameter filter
    rm_params,     # Remove specific parameters
    partialx,      # Extended partial application
    ch_names,      # Change parameter names
    func_to_method_func,      # Convert function to method
    bind_funcs_object_attrs,  # Bind functions to object attributes
    kwargs_trans,  # Keyword argument transformation function
)
```

Each tool has specific use cases documented in the module docstrings.

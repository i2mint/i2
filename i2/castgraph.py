"""
castgraph
=========

A lightweight transformation service for Python that solves the "stable role,
unstable representation" problem: a resource has a consistent semantic role
(e.g., configuration, text, structured record) but appears in many forms
(filepath, string, dict, custom class), while consumers expect specific
representations. castgraph organizes transformations as a graph of "kinds"
(data representations) and routes requests through the best available path.

Key concepts
------------
- **Kind**: Any hashable identifier for a data representation (type, string, custom marker)
- **Transformation**: An edge in the graph that converts one kind to another
- **Kind Predicate (isa)**: A function that determines if an object is of a kind
- **TransformationGraph**: The main registry with graph-oriented interface

Solution patterns
-----------------
- **Type Converter / Conversion Service**: central registry mapping (FromKind, ToKind) to transformer functions.
- **Adapter**: each edge adapts one representation to another.
- **Strategy**: routing/selection among multiple possible transformations via cost/priority.
- **(Optional) Canonical Data Model**: a hub kind to reduce pairwise conversions.
- **DDD Anti-Corruption Layer (ACL)**: keep external formats outside the core domain.
- **Typeclass / Multimethod idiom**: dispatch based on (source kind, target kind).

Minimal example (new kind-based interface)
-------------------------------------------
Use the new TransformationGraph with flexible kinds (not limited to types).

    >>> from i2.castgraph import TransformationGraph
    >>> graph = TransformationGraph()
    >>> # Add nodes with predicates
    >>> graph.add_node('text', isa=lambda x: isinstance(x, str))
    >>> graph.add_node('json_dict', isa=lambda x: isinstance(x, dict))
    >>> # Add transformation edges
    >>> @graph.register_edge('text', 'json_dict')
    ... def text_to_json(t, ctx):
    ...     import json
    ...     return json.loads(t or "{}")
    >>> # Transform using kinds (need explicit from_kind since 'text' != str)
    >>> result = graph.transform('{"x": 1}', 'json_dict', from_kind='text')
    >>> result["x"]
    1

Legacy example (type-based interface)
--------------------------------------
The old ConversionRegistry interface still works but is deprecated.

    >>> from i2.castgraph import ConversionRegistry
    >>> import warnings
    >>> class Path(str): ...
    >>> class Text(str): ...
    >>> class Record(dict): ...
    >>> reg = ConversionRegistry()
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     @reg.register(Path, Text)
    ...     def path_to_text(p, ctx):
    ...         fs = (ctx or {}).get("fs", {})
    ...         return Text(fs.get(str(p), ""))
    ...     @reg.register(Text, Record, cost=0.5)
    ...     def text_to_record(t, ctx):
    ...         import json
    ...         return Record(json.loads(t or "{}"))
    >>> ctx = {"fs": {"/app/data.json": '{"x": 1}'}}
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = reg.convert(Path("/app/data.json"), Record, context=ctx)
    >>> isinstance(out, Record) and out["x"] == 1
    True

Main tools
----------
- **TransformationGraph**: the main graph-based registry (recommended).
  - `.add_node(kind, isa=None)`: add a kind with optional predicate.
  - `.add_edge(src, dst, func, cost=1.0)`: add a transformation edge.
  - `.register_edge(src, dst, cost=1.0)`: decorator to add an edge.
  - `.transform(obj, to_kind, from_kind=None, context=None)`: transform with multi-hop routing.
  - `.transform_any(obj, to_kind, context=None)`: transform with automatic kind detection.
  - `.get_transformer(from_kind, to_kind)`: get a composed transformer function.
  - `.detect_kind(obj)`: detect the kind of an object.
  - `.reachable_from(kind)`: get all reachable kinds.
  - `.sources_for(kind)`: get all source kinds.
  - `.kinds()`: get all registered kinds.

- **ConversionRegistry**: DEPRECATED - use TransformationGraph instead.
  - `.register(From, To, cost=1.0)`: DEPRECATED - use `.register_edge()` instead.
  - `.convert(obj, ToType, context=None)`: DEPRECATED - use `.transform()` instead.

- **Kind**: Optional wrapper for explicit kind specification with predicates.
- **KindMatch**: Truthy result from kind predicates that can carry metadata.
- **ConversionError**: raised when no route exists between kinds.

Design guidelines
-----------------
- Define a single TransformationGraph per bounded context; keep edges local.
- Prefer small, testable transformer functions with explicit kinds.
- Use a canonical domain kind as a **hub** when many formats interoperate.
- Assign **costs** to prefer fast/accurate routes; tune with metrics.
- Pass a **context** dict for side-channel knobs (I/O, flags, cache handles).
- Cache paths (via lru_cache) and consider result caching for hot transformations.
- Keep adapters at the boundaries; the core domain should consume domain kinds.
- Add identity edges implicitly; avoid no-op boilerplate.
- Write doctests on each transformer to lock behavior and invariants.
- Use bare hashables (types, strings) as kinds; Kind class is optional.

Migration guide
---------------
Old code using ConversionRegistry::

    reg = ConversionRegistry()
    @reg.register(SrcType, DstType)
    def convert_func(obj, ctx): ...
    result = reg.convert(obj, DstType)

New code using TransformationGraph::

    graph = TransformationGraph()
    @graph.register_edge(SrcType, DstType)
    def transform_func(obj, ctx): ...
    result = graph.transform(obj, DstType)

Or with string kinds::

    graph = TransformationGraph()
    graph.add_node('src_format', isa=lambda x: ...)
    @graph.register_edge('src_format', 'dst_format')
    def transform_func(obj, ctx): ...
    result = graph.transform(obj, 'dst_format')

Design heritage
---------------
castgraph is a composition of well-known patterns centered on a **Type Converter /
Conversion Service**, with **Adapter** edges and **Strategy**-based route selection.
At system boundaries, it complements DDD’s **Anti-Corruption Layer** and can employ
an integration **Canonical Data Model** to curb O(n²) pairwise mappings.
Its (FromType, ToType) dispatch style mirrors **typeclass/multimethod** idioms.
For background reading, see:
- .NET TypeConverter: https://learn.microsoft.com/dotnet/api/system.componentmodel.typeconverter
- Spring ConversionService: https://docs.spring.io/spring-framework/reference/core/validation/convert.html
- Apache Camel Type Converter: https://camel.apache.org/manual/type-converter.html
- Adapter: https://refactoring.guru/design-patterns/adapter
- Strategy: https://refactoring.guru/design-patterns/strategy
- Anti-Corruption Layer: https://martinfowler.com/bliki/AntiCorruptionLayer.html
- Canonical Data Model: https://www.enterpriseintegrationpatterns.com/patterns/messaging/CanonicalDataModel.html
- PEP 443 singledispatch: https://peps.python.org/pep-0443/

Related
-------

- Issue that sparked this implementation: https://github.com/i2mint/i2/issues/79
- Computational path resolution: https://github.com/i2mint/meshed/discussions/71
- Subsuming concept - "routing": https://github.com/i2mint/i2/discussions/68

"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache, wraps
import inspect
import warnings
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    get_type_hints,
    get_origin,
    Type,
    TypeVar,
)
from collections.abc import Callable, Hashable, Iterable, MutableMapping


T = TypeVar("T")
U = TypeVar("U")
Converter = Callable[[Any, Optional[dict]], Any]


class KindMatch:
    """Result of a successful kind predicate match.

    Evaluates to True but can carry additional metadata about the match
    that downstream transformations might use.

    >>> match = KindMatch({'encoding': 'utf-8', 'analyzed': True})
    >>> bool(match)
    True
    >>> match.metadata
    {'encoding': 'utf-8', 'analyzed': True}
    """

    def __init__(self, metadata: dict | None = None):
        self.metadata = metadata or {}

    def __bool__(self):
        return True

    def __repr__(self):
        return f"KindMatch({self.metadata})"


class Kind:
    """Optional marker for explicit kind specification.

    A Kind wraps a hashable identifier and optionally an 'isa' predicate.
    Users are NOT required to use this class - bare hashables work fine.
    This class is for when you want to be explicit or bundle identifier + predicate.

    >>> text_kind = Kind('text', isa=lambda x: isinstance(x, str))
    >>> text_kind.identifier
    'text'
    >>> text_kind.isa("hello")
    True
    """

    def __init__(
        self,
        identifier: Hashable,
        isa: Callable[[Any], bool | KindMatch] | None = None,
    ):
        self.identifier = identifier
        self._isa = isa

    def isa(self, obj: Any) -> bool | KindMatch:
        """Check if obj is of this kind (predicate/recognizer function)."""
        if self._isa is None:
            # If identifier is a type, use isinstance
            if isinstance(self.identifier, type):
                return isinstance(obj, self.identifier)
            raise ValueError(f"Kind {self.identifier} has no 'isa' predicate")
        return self._isa(obj)

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if isinstance(other, Kind):
            return self.identifier == other.identifier
        return self.identifier == other  # Allow comparison with raw identifiers

    def __repr__(self):
        return f"Kind({self.identifier!r})"


@dataclass(frozen=True)
class Transformation:
    """An edge in the transformation graph.

    Represents a transformation function from one kind to another.
    """

    src: Hashable  # Source kind (any hashable)
    dst: Hashable  # Destination kind
    func: Callable[[Any, dict | None], Any]
    cost: float = 1.0  # lower = preferred


# Backward compatibility alias
@dataclass(frozen=True)
class Edge:
    """DEPRECATED: Use Transformation instead. Kept for backward compatibility."""

    src: type[Any]
    dst: type[Any]
    func: Converter
    cost: float = 1.0  # lower = preferred


class ConversionError(TypeError):
    pass


# -----------------------------
# Helpers for registration
# -----------------------------
def _normalize_converter(func: Converter) -> Converter:
    """
    Wrap a converter function to ensure it accepts (obj, context) signature.

    If func only takes one parameter, wrap it to accept and ignore context.
    If func already takes two parameters, return as-is.

    >>> def simple_converter(x): return x * 2
    >>> wrapped = _normalize_converter(simple_converter)
    >>> wrapped(5, None)
    10
    >>> wrapped(5, {'some': 'context'})
    10
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) == 1:
        # Function only takes the object parameter
        @wraps(func)
        def wrapper(obj, context):
            return func(obj)

        return wrapper
    elif len(params) >= 2:
        # Assume it already has (obj, context) or (obj, context, ...) signature
        return func
    else:
        raise ValueError(
            f"Converter function must have at least one parameter, got: {sig}"
        )


def _extract_types_from_annotations(
    func: Callable, provided_src: type | None, provided_dst: type | None
) -> tuple[type[Any], type[Any]]:
    """
    Extract src and dst types from function annotations if not provided.

    Raises
    ------
    ValueError
        If src or dst cannot be determined from either arguments or annotations.

    >>> def converter(x: int, ctx) -> str: return str(x)
    >>> _extract_types_from_annotations(converter, None, None)
    (<class 'int'>, <class 'str'>)
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Evaluate annotations (handles from __future__ annotations)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # Determine src type
    if provided_src is None:
        if not params:
            raise ValueError(
                f"Cannot infer src type: function {func.__name__} has no parameters"
            )
        first_param = params[0]
        annotated = hints.get(first_param.name, None)
        if annotated is None:
            raise ValueError(
                f"Cannot infer src type for {func.__name__}: "
                f"parameter '{first_param.name}' has no type annotation. "
                f"Either provide src explicitly or annotate the first parameter."
            )
        src = annotated
    else:
        src = provided_src
        # Check for conflicts
        if params:
            annotated_src = hints.get(params[0].name, None)
            if annotated_src is not None:
                # Normalize typing aliases (e.g., typing.Mapping) to their runtime
                # origins (e.g., collections.abc.Mapping) for an apples-to-apples
                # comparison. This avoids spurious warnings when the same
                # semantic type is referenced via different typing modules.
                def _canonical_type(t: type[Any]) -> type[Any]:
                    try:
                        origin = get_origin(t)
                        if origin:
                            return origin
                    except Exception:
                        pass
                    # Fallback: map typing.Mapping -> collections.abc.Mapping when
                    # typing.get_origin doesn't return an origin (older Python)
                    try:
                        import collections.abc as cabc

                        if (
                            getattr(t, "__name__", None) == "Mapping"
                            and getattr(t, "__module__", "") == "typing"
                        ):
                            return cabc.Mapping
                    except Exception:
                        pass
                    return t

                if _canonical_type(annotated_src) != _canonical_type(src):
                    warnings.warn(
                        f"Type mismatch in {func.__name__}: "
                        f"register() specifies src={getattr(src, '__name__', str(src))}, "
                        f"but first parameter is annotated as {getattr(annotated_src, '__name__', str(annotated_src))}. "
                        f"Using src={getattr(src, '__name__', str(src))} from register()."
                    )

    # Determine dst type
    if provided_dst is None:
        annotated_ret = hints.get("return", None)
        if annotated_ret is None:
            raise ValueError(
                f"Cannot infer dst type for {func.__name__}: "
                f"no return annotation found. "
                f"Either provide dst explicitly or add a return type annotation."
            )
        dst = annotated_ret
    else:
        dst = provided_dst
        annotated_dst = hints.get("return", None)
        if annotated_dst is not None:
            # Reuse the same canonicalization logic for dst comparison
            def _canonical_type(t: type[Any]) -> type[Any]:
                try:
                    origin = get_origin(t)
                    if origin:
                        return origin
                except Exception:
                    pass
                try:
                    import collections.abc as cabc

                    if (
                        getattr(t, "__name__", None) == "Mapping"
                        and getattr(t, "__module__", "") == "typing"
                    ):
                        return cabc.Mapping
                except Exception:
                    pass
                return t

            if _canonical_type(annotated_dst) != _canonical_type(dst):
                warnings.warn(
                    f"Type mismatch in {func.__name__}: "
                    f"register() specifies dst={getattr(dst, '__name__', str(dst))}, "
                    f"but return annotation is {getattr(annotated_dst, '__name__', str(annotated_dst))}. "
                    f"Using dst={getattr(dst, '__name__', str(dst))} from register()."
                )

    return src, dst


class TransformationGraph:
    """
    A graph-based registry of transformations between kinds (data representations).

    A "kind" is any hashable identifier for a data representation - it can be a type,
    a string, or any custom marker. The graph supports:
      - Flexible kind system (not limited to Python types)
      - Graph-oriented interface (add_node, add_edge)
      - Pluggable kind detection via predicates
      - Shortest-path (by total cost) routing
      - MRO-aware fallback for type-based kinds
      - Caching of paths and (optionally) results

    Design notes:
    - Each transformer has signature: func(obj, context) -> transformed_obj
    - Identity edges are implicit (K -> K) with cost 0
    - If multiple routes exist, the minimum total cost path is chosen
    - Kinds can be types, strings, or any hashable objects
    """

    def __init__(self) -> None:
        # Store transformations (edges) indexed by source kind
        self._transformations: dict[Hashable, list[Transformation]] = defaultdict(list)
        # Store kind predicates (isa functions) for automatic kind detection
        self._kind_predicates: dict[Hashable, Callable[[Any], bool | KindMatch]] = {}
        # Track all registered kinds (nodes in the graph)
        self._kinds: set[Hashable] = set()
        # Optional custom kind detector function
        self._kind_detector: Callable[[Any], Hashable | None] | None = None
        # Optional result cache (obj identity-sensitive)
        self._result_cache: dict[tuple[int, Hashable], Any] = {}

    # -----------------------------
    # Graph-oriented interface (new)
    # -----------------------------
    def add_node(
        self,
        kind: Hashable | Kind,
        isa: Callable[[Any], bool | KindMatch] | None = None,
    ) -> None:
        """Add a kind (node) to the graph with optional predicate.

        Parameters
        ----------
        kind : Hashable | Kind
            The kind identifier (can be a type, string, or Kind object)
        isa : Callable[[Any], bool | KindMatch] | None
            Optional predicate function to detect if an object is of this kind

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> graph.add_node('text', isa=lambda x: isinstance(x, str))
        >>> graph.add_node(int)  # Type implies isinstance check
        """
        if isinstance(kind, Kind):
            identifier = kind.identifier
            if isa is None:
                # Use the Kind's isa method
                isa = kind.isa
        else:
            identifier = kind

        # Track this kind
        self._kinds.add(identifier)

        # Auto-generate isa for types if not provided
        if isa is None and isinstance(identifier, type):
            isa = lambda obj, t=identifier: isinstance(obj, t)

        if isa is not None:
            self._kind_predicates[identifier] = isa

    def add_edge(
        self,
        src: Hashable | Kind,
        dst: Hashable | Kind,
        func: Callable,
        *,
        cost: float = 1.0,
    ) -> None:
        """Add a transformation (edge) between two kinds.

        Automatically adds nodes if they don't exist.

        Parameters
        ----------
        src : Hashable | Kind
            Source kind
        dst : Hashable | Kind
            Destination kind
        func : Callable
            Transformation function with signature func(obj, context) -> transformed_obj
            or func(obj) -> transformed_obj (will be wrapped)
        cost : float
            Cost of this transformation (lower is preferred)

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> def text_to_int(s, ctx): return int(s)
        >>> graph.add_edge('text', int, text_to_int)
        """
        src_id = src.identifier if isinstance(src, Kind) else src
        dst_id = dst.identifier if isinstance(dst, Kind) else dst

        # Auto-add nodes if they don't exist
        if src_id not in self._kind_predicates:
            self.add_node(src)
        if dst_id not in self._kind_predicates:
            self.add_node(dst)

        normalized_func = _normalize_converter(func)
        self._transformations[src_id].append(
            Transformation(src_id, dst_id, normalized_func, cost)
        )
        self._clear_cache()

    def register_edge(
        self,
        src: Hashable | Kind | None = None,
        dst: Hashable | Kind | None = None,
        *,
        cost: float = 1.0,
    ) -> Callable:
        """Decorator to register a transformation edge.

        Can infer src/dst from function annotations if not provided.

        Parameters
        ----------
        src : Hashable | Kind | None
            Source kind (inferred from annotations if None)
        dst : Hashable | Kind | None
            Destination kind (inferred from annotations if None)
        cost : float
            Cost of this transformation

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> @graph.register_edge('text', int)
        ... def text_to_int(s, ctx): return int(s)
        """

        def deco(func: Callable) -> Callable:
            nonlocal src, dst
            # Infer from annotations if not provided
            if src is None or dst is None:
                inferred_src, inferred_dst = _extract_types_from_annotations(
                    func, src, dst
                )
                src = src if src is not None else inferred_src
                dst = dst if dst is not None else inferred_dst

            self.add_edge(src, dst, func, cost=cost)
            return func

        return deco

    def _clear_cache(self):
        """Clear path cache when graph changes."""
        try:
            type(self)._find_path_cached.cache_clear()
        except Exception:
            pass

    # -----------------------------
    # Transformation methods (new)
    # -----------------------------
    def get_transformer(
        self,
        from_kind: Hashable | Kind,
        to_kind: Hashable | Kind,
        *,
        context: dict | None = None,
    ) -> Callable[[Any], Any]:
        """Get a function that transforms from_kind → to_kind.

        Returns a composed transformer function (Pipe-like).

        Parameters
        ----------
        from_kind : Hashable | Kind
            Source kind
        to_kind : Hashable | Kind
            Destination kind
        context : dict | None
            Optional context to bake into the transformer

        Returns
        -------
        Callable[[Any], Any]
            A function that transforms objects from from_kind to to_kind

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> @graph.register_edge(str, int)
        ... def str_to_int(s, ctx): return int(s)
        >>> transformer = graph.get_transformer(str, int)
        >>> transformer("42")
        42
        """
        from_id = from_kind.identifier if isinstance(from_kind, Kind) else from_kind
        to_id = to_kind.identifier if isinstance(to_kind, Kind) else to_kind

        path = self._find_path_cached(from_id, to_id)

        def transformer(obj: Any, ctx: dict | None = context) -> Any:
            return self._apply_path(obj, path, ctx)

        return transformer

    def transform(
        self,
        obj: Any,
        to_kind: Hashable | Kind,
        *,
        from_kind: Hashable | Kind | None = None,
        context: dict | None = None,
        use_result_cache: bool = False,
    ) -> Any:
        """Transform obj to to_kind.

        If from_kind not specified, uses type(obj) with MRO fallback.

        Parameters
        ----------
        obj : Any
            Object to transform
        to_kind : Hashable | Kind
            Destination kind
        from_kind : Hashable | Kind | None
            Source kind (inferred if None)
        context : dict | None
            Optional context passed to transformation functions
        use_result_cache : bool
            If True, cache results keyed by (id(obj), to_kind)

        Returns
        -------
        Any
            Transformed object

        Raises
        ------
        ConversionError
            If no transformation path is found

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> @graph.register_edge(str, int)
        ... def str_to_int(s, ctx): return int(s)
        >>> graph.transform("42", int)
        42
        """
        to_id = to_kind.identifier if isinstance(to_kind, Kind) else to_kind

        # Explicit from_kind specified
        if from_kind is not None:
            from_id = from_kind.identifier if isinstance(from_kind, Kind) else from_kind
            path = self._find_path_cached(from_id, to_id)
            return self._apply_path(obj, path, context)

        # Check identity first
        if isinstance(obj, to_id) if isinstance(to_id, type) else False:
            return obj

        # Check cache
        if use_result_cache:
            key = (id(obj), to_id)
            if key in self._result_cache:
                return self._result_cache[key]

        # Try type-based lookup with MRO
        from_types = tuple(type(obj).mro())
        for from_type in from_types:
            if from_type in self._transformations or from_type in self._kind_predicates:
                try:
                    path = self._find_path_cached(from_type, to_id)
                    result = self._apply_path(obj, path, context)
                    if use_result_cache:
                        self._result_cache[key] = result
                    return result
                except ConversionError:
                    continue

        raise ConversionError(
            f"No transformation path from {type(obj).__name__} to {to_id}"
        )

    def transform_any(
        self,
        obj: Any,
        to_kind: Hashable | Kind,
        *,
        context: dict | None = None,
        use_result_cache: bool = False,
    ) -> Any:
        """Transform obj to to_kind with automatic kind detection.

        Uses configured kind detector or fallback detection strategy.

        Parameters
        ----------
        obj : Any
            Object to transform
        to_kind : Hashable | Kind
            Destination kind
        context : dict | None
            Optional context passed to transformation functions
        use_result_cache : bool
            If True, cache results

        Returns
        -------
        Any
            Transformed object

        Raises
        ------
        ConversionError
            If no transformation path is found or kind cannot be detected

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> graph.add_node('text', isa=lambda x: isinstance(x, str))
        >>> @graph.register_edge('text', int)
        ... def text_to_int(s, ctx): return int(s)
        >>> graph.transform_any("42", int)  # doctest: +SKIP
        42
        """
        detected_kind = self.detect_kind(obj)
        if detected_kind is None:
            raise ConversionError(f"Could not detect kind of {obj}")

        return self.transform(
            obj,
            to_kind,
            from_kind=detected_kind,
            context=context,
            use_result_cache=use_result_cache,
        )

    # -----------------------------
    # Kind detection methods (new)
    # -----------------------------
    def set_kind_detector(self, detector: Callable[[Any], Hashable | None]) -> None:
        """Set a custom kind detector function.

        The detector receives an object and returns a kind identifier or None.

        Parameters
        ----------
        detector : Callable[[Any], Hashable | None]
            Function that takes an object and returns its kind or None

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> def my_detector(obj):
        ...     if isinstance(obj, str) and obj.startswith('{"'):
        ...         return 'json_string'
        ...     return None
        >>> graph.set_kind_detector(my_detector)
        """
        self._kind_detector = detector

    def detect_kind(self, obj: Any) -> Hashable | None:
        """Detect the kind of an object.

        Uses custom detector if set, otherwise tries registered predicates in order.
        Returns None if no kind matches.

        Parameters
        ----------
        obj : Any
            Object to classify

        Returns
        -------
        Hashable | None
            The detected kind identifier, or None if no match

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> graph.add_node('text', isa=lambda x: isinstance(x, str))
        >>> graph.detect_kind("hello")
        'text'
        """
        # Try custom detector first
        if self._kind_detector is not None:
            result = self._kind_detector(obj)
            if result is not None:
                return result

        # Fall back to trying all registered predicates
        # Dict maintains insertion order (Python 3.7+)
        for kind, predicate in self._kind_predicates.items():
            try:
                result = predicate(obj)
                if result:
                    # Result can be True or a truthy KindMatch
                    # Either way, we found our kind
                    return kind
            except Exception:
                # Silently skip predicates that raise
                continue

        # Last resort: try type(obj)
        obj_type = type(obj)
        if obj_type in self._kind_predicates:
            return obj_type

        return None

    # -----------------------------
    # Introspection methods (new)
    # -----------------------------
    def reachable_from(self, kind: Hashable | Kind) -> set[Hashable]:
        """Get all kinds reachable from this kind via transformations.

        Parameters
        ----------
        kind : Hashable | Kind
            The starting kind

        Returns
        -------
        set[Hashable]
            Set of all reachable kind identifiers

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> # ... register transformations ...
        >>> reachable = graph.reachable_from('text')
        """
        kind_id = kind.identifier if isinstance(kind, Kind) else kind
        reachable = set()
        visited = set()
        queue = [kind_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for trans in self._transformations.get(current, []):
                reachable.add(trans.dst)
                queue.append(trans.dst)

        return reachable

    def sources_for(self, kind: Hashable | Kind) -> set[Hashable]:
        """Get all kinds that can be transformed to this kind.

        Parameters
        ----------
        kind : Hashable | Kind
            The destination kind

        Returns
        -------
        set[Hashable]
            Set of all source kind identifiers

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> # ... register transformations ...
        >>> sources = graph.sources_for(int)
        """
        kind_id = kind.identifier if isinstance(kind, Kind) else kind
        sources = set()

        for src, trans_list in self._transformations.items():
            for trans in trans_list:
                if trans.dst == kind_id:
                    sources.add(src)

        return sources

    def kinds(self) -> set[Hashable]:
        """Get all registered kinds (nodes in the graph).

        Returns
        -------
        set[Hashable]
            Set of all registered kind identifiers

        Examples
        --------
        >>> graph = TransformationGraph()
        >>> graph.add_node('text')
        >>> graph.add_node(int)
        >>> 'text' in graph.kinds()
        True
        """
        return self._kinds.copy()

    # -----------------------------
    # Backward compatibility methods (deprecated)
    # -----------------------------
    def register(
        self,
        src: type[Any] | None = None,
        dst: type[Any] | None = None,
        *,
        cost: float = 1.0,
    ) -> Callable[[Converter], Converter]:
        """DEPRECATED: Use register_edge() instead.

        This method is kept for backward compatibility.

        Examples
        --------
        >>> import warnings
        >>> graph = TransformationGraph()
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     @graph.register(str, int)
        ...     def str_to_int(s, ctx): return int(s)
        """
        warnings.warn(
            "register() is deprecated and will be removed in a future version. "
            "Use register_edge() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.register_edge(src, dst, cost=cost)

    def convert(
        self,
        obj: Any,
        to_type: type[U],
        *,
        context: dict | None = None,
        use_result_cache: bool = False,
    ) -> U:
        """DEPRECATED: Use transform() instead.

        This method is kept for backward compatibility.

        Examples
        --------
        >>> import warnings
        >>> graph = TransformationGraph()
        >>> @graph.register_edge(str, int)
        ... def str_to_int(s, ctx): return int(s)
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     result = graph.convert("42", int)
        >>> result
        42
        """
        warnings.warn(
            "convert() is deprecated and will be removed in a future version. "
            "Use transform() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.transform(
            obj, to_type, context=context, use_result_cache=use_result_cache
        )

    # -----------------------------
    # Internals: path finding & application
    # -----------------------------
    def _apply_path(
        self, obj: Any, path: list[Transformation], context: dict | None
    ) -> Any:
        out = obj
        for trans in path:
            out = trans.func(out, context)
        return out

    @lru_cache(maxsize=4096)
    def _find_path_cached(self, src: Hashable, dst: Hashable) -> list[Transformation]:
        return self._find_min_cost_path(src, dst)

    def _neighbors(self, src: Hashable) -> Iterable[Transformation]:
        # Outgoing transformations from exact src
        yield from self._transformations.get(src, [])
        # Identity edge (src->src) for free; useful for uniform path logic.
        yield Transformation(src, src, lambda x, ctx: x, cost=0.0)

    def _find_min_cost_path(self, src: Hashable, dst: Hashable) -> list[Transformation]:
        if src is dst:
            return []  # already at target

        # Dijkstra over kind-nodes
        frontier: list[tuple[float, Hashable]] = [(0.0, src)]
        dist: dict[Hashable, float] = {src: 0.0}
        prev: dict[Hashable, tuple[Hashable | None, Transformation | None]] = {
            src: (None, None)
        }

        visited: set[Hashable] = set()

        while frontier:
            frontier.sort(key=lambda t: t[0])
            cost_so_far, node = frontier.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node is dst:
                break

            for trans in self._neighbors(node):
                new_cost = cost_so_far + trans.cost
                if trans.dst not in dist or new_cost < dist[trans.dst]:
                    dist[trans.dst] = new_cost
                    prev[trans.dst] = (node, trans)
                    frontier.append((new_cost, trans.dst))

        if dst not in prev:
            # Try MRO-based widening for dst (accept subclass→superclass matches)
            # Only if dst is a type
            if isinstance(dst, type):
                original_dst = dst
                for super_dst in dst.mro():
                    if super_dst in prev and super_dst is not src:
                        dst = super_dst  # type: ignore[assignment]
                        break

        if dst not in prev:
            raise ConversionError(f"No route from {src} to {dst}")

        # Reconstruct path
        path_edges: Deque[Transformation] = deque()
        cur: Hashable | None = dst
        while cur and prev[cur][0] is not None:
            _, trans = prev[cur]
            assert trans is not None
            path_edges.appendleft(trans)
            cur = prev[cur][0]

        # Strip leading/trailing identity edges
        final_path = [t for t in path_edges if t.src is not t.dst]

        # If MRO widening happened but no conversion steps, this is invalid
        if "original_dst" in locals() and dst is not original_dst and not final_path:
            raise ConversionError(f"No route from {src} to {original_dst}")

        return final_path


class ConversionRegistry:
    """
    DEPRECATED: Use TransformationGraph instead.

    A graph-based registry of converters between Python types with:
      - registration decorator
      - shortest-path (by total cost) routing
      - MRO-aware fallback for source types
      - caching of paths and (optionally) results

    Design notes:
    - Each converter has signature: func(obj, context) -> converted_obj
    - Identity edges are implicit (T -> T) with cost 0.
    - If multiple routes exist, the minimum total cost path is chosen.
    """

    def __init__(self) -> None:
        self._edges: dict[type[Any], list[Edge]] = defaultdict(list)
        # Optional result cache (obj identity-sensitive). Keep simple by default.
        self._result_cache: dict[tuple[int, type[Any]], Any] = {}

    # -----------------------------
    # Registration API
    # -----------------------------
    def register(
        self,
        src: type[Any] | None = None,
        dst: type[Any] | None = None,
        *,
        cost: float = 1.0,
    ) -> Callable[[Converter], Converter]:
        """
        Decorator to register a converter function.

        If src or dst are not provided, they will be inferred from the function's
        type annotations.

        Example
        -------
        >>> reg = ConversionRegistry()
        >>> class A: ...
        >>> class B: ...
        >>> @reg.register(A, B)
        ... def a_to_b(a, ctx): return B()
        ...
        >>> isinstance(reg.convert(A(), B), B)
        True

        # Can infer types from annotations:
        >>> class X: ...
        >>> class Y: ...
        >>> @reg.register()
        ... def x_to_y(x: X, ctx) -> Y:
        ...     return Y()
        >>> isinstance(reg.convert(X(), Y), Y)
        True
        """

        def deco(func: Converter) -> Converter:
            nonlocal src, dst
            # Infer types from annotations if not provided
            src, dst = _extract_types_from_annotations(func, src, dst)
            normalized_func = _normalize_converter(func)
            self._edges[src].append(Edge(src, dst, normalized_func, cost))
            # Invalidate path cache because the graph changed
            try:
                type(self)._find_path_cached.cache_clear()
            except Exception:
                pass
            return func

        return deco

    # -----------------------------
    # Public convert()
    # -----------------------------
    def convert(
        self,
        obj: Any,
        to_type: type[U],
        *,
        context: dict | None = None,
        use_result_cache: bool = False,
    ) -> U:
        """
        Convert `obj` to `to_type`, possibly via multi-hop.

        Parameters
        ----------
        obj : Any
            Source object to convert.
        to_type : Type[U]
            Desired target type.
        context : Optional[dict]
            Arbitrary context propagated through the chain (e.g., config, flags).
        use_result_cache : bool
            If True, cache results keyed by (id(obj), to_type).

        Returns
        -------
        U
            Converted object.

        Raises
        ------
        ConversionError
            If no conversion path is found.

        Examples
        --------
        >>> reg = ConversionRegistry()
        >>> class X: ...
        >>> class Y: ...
        >>> class Z: ...
        >>> @reg.register(X, Y)
        ... def x_to_y(x, ctx): return Y()
        ...
        >>> @reg.register(Y, Z)
        ... def y_to_z(y, ctx): return Z()
        ...
        >>> isinstance(reg.convert(X(), Z), Z)
        True

        MRO fallback: if a converter is registered for a base class, it applies to a subclass.
        >>> class Base: ...
        >>> class Sub(Base): ...
        >>> class Out: ...
        >>> reg2 = ConversionRegistry()
        >>> @reg2.register(Base, Out)
        ... def base_to_out(b, ctx): return Out()
        ...
        >>> isinstance(reg2.convert(Sub(), Out), Out)
        True
        """
        if isinstance(obj, to_type):
            return obj  # Identity

        if use_result_cache:
            key = (id(obj), to_type)
            if key in self._result_cache:
                return self._result_cache[key]

        from_types = tuple(type(obj).mro())  # MRO-aware source candidates

        # Try direct or multi-hop path for the first MRO type that can route.
        for from_type in from_types:
            try:
                path = self._find_path_cached(from_type, to_type)
            except ConversionError:
                continue
            result = self._apply_path(obj, path, context)
            if use_result_cache:
                self._result_cache[(id(obj), to_type)] = result
            return result

        raise ConversionError(
            f"No conversion path from {type(obj).__name__} to {to_type.__name__}"
        )

    # -----------------------------
    # Internals: path finding & application
    # -----------------------------
    def _apply_path(self, obj: Any, path: list[Edge], context: dict | None) -> Any:
        out = obj
        for edge in path:
            out = edge.func(out, context)
        return out

    @lru_cache(maxsize=4096)
    def _find_path_cached(self, src: type[Any], dst: type[Any]) -> list[Edge]:
        return self._find_min_cost_path(src, dst)

    def _neighbors(self, src: type[Any]) -> Iterable[Edge]:
        # Outgoing edges from exact src
        yield from self._edges.get(src, [])
        # Identity edge (src->src) for free; useful for uniform path logic.
        yield Edge(src, src, lambda x, ctx: x, cost=0.0)

    def _find_min_cost_path(self, src: type[Any], dst: type[Any]) -> list[Edge]:
        if src is dst:
            return []  # already at target

        # Dijkstra over type-nodes
        frontier: list[tuple[float, type[Any]]] = [(0.0, src)]
        dist: dict[type[Any], float] = {src: 0.0}
        prev: dict[type[Any], tuple[type[Any] | None, Edge | None]] = {
            src: (None, None)
        }

        visited: set[type[Any]] = set()

        while frontier:
            frontier.sort(key=lambda t: t[0])
            cost_so_far, node = frontier.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node is dst:
                break

            for edge in self._neighbors(node):
                new_cost = cost_so_far + edge.cost
                if edge.dst not in dist or new_cost < dist[edge.dst]:
                    dist[edge.dst] = new_cost
                    prev[edge.dst] = (node, edge)
                    frontier.append((new_cost, edge.dst))

        if dst not in prev:
            # Try MRO-based widening for dst (accept subclass→superclass matches)
            original_dst = dst
            for super_dst in dst.mro():
                if super_dst in prev and super_dst is not src:
                    dst = super_dst  # type: ignore[assignment]
                    break

        if dst not in prev:
            raise ConversionError(f"No route from {src.__name__} to {dst.__name__}")

        # Reconstruct path
        path_edges: Deque[Edge] = deque()
        cur: type[Any] | None = dst
        while cur and prev[cur][0] is not None:
            _, edge = prev[cur]
            assert edge is not None
            path_edges.appendleft(edge)
            cur = prev[cur][0]

        # Strip leading/trailing identity edges
        final_path = [e for e in path_edges if e.src is not e.dst]

        # If MRO widening happened but no conversion steps, this is invalid
        if "original_dst" in locals() and dst is not original_dst and not final_path:
            raise ConversionError(
                f"No route from {src.__name__} to {original_dst.__name__}"
            )

        return final_path

    # -----------------------------
    # Registration API
    # -----------------------------
    def register(
        self,
        src: type[Any] | None = None,
        dst: type[Any] | None = None,
        *,
        cost: float = 1.0,
    ) -> Callable[[Converter], Converter]:
        """
            Decorator to register a converter function.

            Example
            -------
        >>> reg = ConversionRegistry()
        >>> class A: ...
        >>> class B: ...
        >>> @reg.register(A, B)
        ... def a_to_b(a, ctx): return B()
        ...
        >>> isinstance(reg.convert(A(), B), B)
        True

        # Can infer types from annotations:
        >>> class X: ...
        >>> class Y: ...
        >>> @reg.register()
        ... def x_to_y(x: X, ctx) -> Y:
        ...     return Y()
        >>> isinstance(reg.convert(X(), Y), Y)
        True
        """

        def deco(func: Converter) -> Converter:
            nonlocal src, dst
            # Infer types from annotations if not provided
            src, dst = _extract_types_from_annotations(func, src, dst)
            normalized_func = _normalize_converter(func)
            self._edges[src].append(Edge(src, dst, normalized_func, cost))
            # Invalidate path cache because the graph changed
            try:
                type(self)._find_path_cached.cache_clear()
            except Exception:
                pass
            return func

        return deco

    # -----------------------------
    # Public convert()
    # -----------------------------
    def convert(
        self,
        obj: Any,
        to_type: type[U],
        *,
        context: dict | None = None,
        use_result_cache: bool = False,
    ) -> U:
        """
        Convert `obj` to `to_type`, possibly via multi-hop.

        Parameters
        ----------
        obj : Any
            Source object to convert.
        to_type : Type[U]
            Desired target type.
        context : Optional[dict]
            Arbitrary context propagated through the chain (e.g., config, flags).
        use_result_cache : bool
            If True, cache results keyed by (id(obj), to_type).

        Returns
        -------
        U
            Converted object.

        Raises
        ------
        ConversionError
            If no conversion path is found.

        Examples
        --------
        >>> reg = ConversionRegistry()
        >>> class X: ...
        >>> class Y: ...
        >>> class Z: ...
        >>> @reg.register(X, Y)
        ... def x_to_y(x, ctx): return Y()
        ...
        >>> @reg.register(Y, Z)
        ... def y_to_z(y, ctx): return Z()
        ...
        >>> isinstance(reg.convert(X(), Z), Z)
        True

        MRO fallback: if a converter is registered for a base class, it applies to a subclass.
        >>> class Base: ...
        >>> class Sub(Base): ...
        >>> class Out: ...
        >>> reg2 = ConversionRegistry()
        >>> @reg2.register(Base, Out)
        ... def base_to_out(b, ctx): return Out()
        ...
        >>> isinstance(reg2.convert(Sub(), Out), Out)
        True
        """
        if isinstance(obj, to_type):
            return obj  # Identity

        if use_result_cache:
            key = (id(obj), to_type)
            if key in self._result_cache:
                return self._result_cache[key]

        from_types = tuple(type(obj).mro())  # MRO-aware source candidates

        # Try direct or multi-hop path for the first MRO type that can route.
        for from_type in from_types:
            try:
                path = self._find_path_cached(from_type, to_type)
            except ConversionError:
                continue
            result = self._apply_path(obj, path, context)
            if use_result_cache:
                self._result_cache[(id(obj), to_type)] = result
            return result

        raise ConversionError(
            f"No conversion path from {type(obj).__name__} to {to_type.__name__}"
        )

    # -----------------------------
    # Internals: path finding & application
    # -----------------------------
    def _apply_path(self, obj: Any, path: list[Edge], context: dict | None) -> Any:
        out = obj
        for edge in path:
            out = edge.func(out, context)
        return out

    @lru_cache(maxsize=4096)
    def _find_path_cached(self, src: type[Any], dst: type[Any]) -> list[Edge]:
        return self._find_min_cost_path(src, dst)

    def _neighbors(self, src: type[Any]) -> Iterable[Edge]:
        # Outgoing edges from exact src
        yield from self._edges.get(src, [])
        # Identity edge (src->src) for free; useful for uniform path logic.
        yield Edge(src, src, lambda x, ctx: x, cost=0.0)

    def _find_min_cost_path(self, src: type[Any], dst: type[Any]) -> list[Edge]:
        if src is dst:
            return []  # already at target

        # Dijkstra over type-nodes
        frontier: list[tuple[float, type[Any]]] = [(0.0, src)]
        dist: dict[type[Any], float] = {src: 0.0}
        prev: dict[type[Any], tuple[type[Any] | None, Edge | None]] = {
            src: (None, None)
        }

        visited: set[type[Any]] = set()

        while frontier:
            frontier.sort(key=lambda t: t[0])
            cost_so_far, node = frontier.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node is dst:
                break

            for edge in self._neighbors(node):
                new_cost = cost_so_far + edge.cost
                if edge.dst not in dist or new_cost < dist[edge.dst]:
                    dist[edge.dst] = new_cost
                    prev[edge.dst] = (node, edge)
                    frontier.append((new_cost, edge.dst))

        if dst not in prev:
            # Try MRO-based widening for dst (accept subclass→superclass matches)
            original_dst = dst
            for super_dst in dst.mro():
                if super_dst in prev and super_dst is not src:
                    dst = super_dst  # type: ignore[assignment]
                    break

        if dst not in prev:
            raise ConversionError(f"No route from {src.__name__} to {dst.__name__}")

        # Reconstruct path
        path_edges: Deque[Edge] = deque()
        cur: type[Any] | None = dst
        while cur and prev[cur][0] is not None:
            _, edge = prev[cur]
            assert edge is not None
            path_edges.appendleft(edge)
            cur = prev[cur][0]

        # Strip leading/trailing identity edges
        final_path = [e for e in path_edges if e.src is not e.dst]

        # If MRO widening happened but no conversion steps, this is invalid
        if "original_dst" in locals() and dst is not original_dst and not final_path:
            raise ConversionError(
                f"No route from {src.__name__} to {original_dst.__name__}"
            )

        return final_path


# ----------------------------------------------------------------------
# Backward compatibility aliases
# ----------------------------------------------------------------------

# Alias ConversionRegistry to TransformationGraph for backward compatibility
# The old ConversionRegistry class is kept but deprecated
# Users should migrate to TransformationGraph

# ----------------------------------------------------------------------
# Best-practice guidance
# ----------------------------------------------------------------------


def design_guidelines() -> str:
    """
    Returns concise guidance for organizing casting in Python.

    >>> "registry" in design_guidelines().lower()
    True
    """
    return (
        "- Define a single ConversionRegistry per bounded context; keep edges local.\n"
        "- Prefer small, testable converter functions with explicit types.\n"
        "- Use a canonical domain type as a **hub** when many formats interoperate.\n"
        "- Assign **costs** to prefer fast/accurate routes; tune with metrics.\n"
        "- Pass a **context** dict for side-channel knobs (I/O, flags, cache handles).\n"
        "- Cache paths (via lru_cache) and consider result caching for hot conversions.\n"
        "- Keep adapters at the boundaries; the core domain should consume domain types.\n"
        "- Add identity edges implicitly; avoid no-op boilerplate.\n"
        "- Write doctests on each converter to lock behavior and invariants.\n"
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()

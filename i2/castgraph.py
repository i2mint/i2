"""
castgraph
=========

A lightweight conversion service for Python that solves the "stable role,
unstable representation" problem: a resource has a consistent semantic role
(e.g., configuration, text, structured record) but appears in many forms
(filepath, string, dict, custom class), while consumers expect specific
representations. castgraph organizes conversions as a graph of typed edges and
routes requests through the best available path.

Solution patterns
-----------------
- **Type Converter / Conversion Service**: central registry mapping (FromType, ToType) to converter functions.
- **Adapter**: each edge adapts one representation to another.
- **Strategy**: routing/selection among multiple possible conversions via cost/priority.
- **(Optional) Canonical Data Model**: a hub type to reduce pairwise conversions.
- **DDD Anti-Corruption Layer (ACL)**: keep external formats outside the core domain.
- **Typeclass / Multimethod idiom**: dispatch based on (source type, target type).

Minimal example (doctest)
-------------------------
Register tiny adapters and convert across types with optional multi-hop routing.

    >>> from i2.castgraph import ConversionRegistry
    >>> class Path(str): ...
    >>> class Text(str): ...
    >>> class Record(dict): ...
    >>> reg = ConversionRegistry()
    >>> @reg.register(Path, Text)
    ... def path_to_text(p, ctx):  # ctx is an optional dict
    ...     fs = (ctx or {}).get("fs", {})
    ...     return Text(fs.get(str(p), ""))
    ...
    >>> @reg.register(Text, Record, cost=0.5)  # cheaper direct edge
    ... def text_to_record(t, ctx):
    ...     import json
    ...     return Record(json.loads(t or "{}"))
    ...
    >>> ctx = {"fs": {"/app/data.json": '{"x": 1}'}}
    >>> out = reg.convert(Path("/app/data.json"), Record, context=ctx)
    >>> isinstance(out, Record) and out["x"] == 1
    True

Main tools
----------
- **ConversionRegistry**: the central graph-based registry.
  - `.register(From, To, cost=1.0)`: decorator to add an adapter (`func(obj, context) -> new_obj`).
  - `.convert(obj, ToType, context=None, use_result_cache=False)`: convert with multi-hop routing,
    MRO-aware fallbacks, and optional result caching.
- **ConversionError**: raised when no route exists between types.

Design guidelines
-----------------
- Define a single ConversionRegistry per bounded context; keep edges local.
- Prefer small, testable converter functions with explicit types.
- Use a canonical domain type as a **hub** when many formats interoperate.
- Assign **costs** to prefer fast/accurate routes; tune with metrics.
- Pass a **context** dict for side-channel knobs (I/O, flags, cache handles).
- Cache paths (via lru_cache) and consider result caching for hot conversions.
- Keep adapters at the boundaries; the core domain should consume domain types.
- Add identity edges implicitly; avoid no-op boilerplate.
- Write doctests on each converter to lock behavior and invariants.

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
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Hashable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
)


T = TypeVar("T")
U = TypeVar("U")
Converter = Callable[[Any, Optional[dict]], Any]


@dataclass(frozen=True)
class Edge:
    src: Type[Any]
    dst: Type[Any]
    func: Converter
    cost: float = 1.0  # lower = preferred


class ConversionError(TypeError):
    pass


class ConversionRegistry:
    """
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
        self._edges: Dict[Type[Any], List[Edge]] = defaultdict(list)
        # Optional result cache (obj identity-sensitive). Keep simple by default.
        self._result_cache: Dict[Tuple[int, Type[Any]], Any] = {}

    # -----------------------------
    # Registration API
    # -----------------------------
    def register(
        self, src: Type[Any], dst: Type[Any], *, cost: float = 1.0
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
        """

        def deco(func: Converter) -> Converter:
            self._edges[src].append(Edge(src, dst, func, cost))
            # Invalidate path cache because the graph changed
            self._find_path_cached.cache_clear()
            return func

        return deco

    # -----------------------------
    # Public convert()
    # -----------------------------
    def convert(
        self,
        obj: Any,
        to_type: Type[U],
        *,
        context: Optional[dict] = None,
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
    def _apply_path(self, obj: Any, path: List[Edge], context: Optional[dict]) -> Any:
        out = obj
        for edge in path:
            out = edge.func(out, context)
        return out

    @lru_cache(maxsize=4096)
    def _find_path_cached(self, src: Type[Any], dst: Type[Any]) -> List[Edge]:
        return self._find_min_cost_path(src, dst)

    def _neighbors(self, src: Type[Any]) -> Iterable[Edge]:
        # Outgoing edges from exact src
        yield from self._edges.get(src, [])
        # Identity edge (src->src) for free; useful for uniform path logic.
        yield Edge(src, src, lambda x, ctx: x, cost=0.0)

    def _find_min_cost_path(self, src: Type[Any], dst: Type[Any]) -> List[Edge]:
        if src is dst:
            return []  # already at target

        # Dijkstra over type-nodes
        frontier: List[Tuple[float, Type[Any]]] = [(0.0, src)]
        dist: Dict[Type[Any], float] = {src: 0.0}
        prev: Dict[Type[Any], Tuple[Optional[Type[Any]], Optional[Edge]]] = {
            src: (None, None)
        }

        visited: set[Type[Any]] = set()

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
        cur: Optional[Type[Any]] = dst
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

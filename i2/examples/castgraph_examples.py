"""Examples using castgraph."""

from typing import Optional, MutableMapping
from i2.castgraph import ConversionRegistry


# ----------------------------------------------------------------------
# Example usage and optional "canonical" hub
# ----------------------------------------------------------------------


class FilePath(str):
    """Marker type for file paths (to distinguish from arbitrary strings)."""


class Text(str):
    """Marker type for text payloads."""


class JSONDict(dict):
    """Marker type for dicts with JSON semantics."""


class CanonicalRecord(dict):
    """Canonical in-memory record format."""


def example_registry() -> ConversionRegistry:
    """
    Build a registry that can:
      - FilePath -> Text
      - Text -> JSONDict
      - JSONDict -> CanonicalRecord
      - Direct Text -> CanonicalRecord (cheaper path)

    The route chosen will be the minimum total cost.

    >>> reg = example_registry()
    >>> # Multi-hop via cheaper direct Text->CanonicalRecord
    >>> data = Text('{"a": 1, "b": 2}')
    >>> out = reg.convert(data, CanonicalRecord)
    >>> isinstance(out, CanonicalRecord) and out["a"] == 1
    True

    >>> # FilePath -> Text -> CanonicalRecord
    >>> fp = FilePath("/tmp/dummy.json")
    >>> out = reg.convert(fp, CanonicalRecord, context={"fs": {" /tmp/dummy.json": '{"x":42}'}})
    >>> isinstance(out, CanonicalRecord) and out["x"] == 42
    True
    """
    import json

    reg = ConversionRegistry()

    @reg.register(FilePath, Text, cost=1.0)
    def filepath_to_text(fp: FilePath, ctx: Optional[dict]) -> Text:
        # Example: pretend-read from an injected "fs" dict for testability
        fs: MutableMapping[str, str] = (ctx or {}).get("fs", {})
        content = fs.get(str(fp))
        if content is None:
            # Fallback to real file read (commented to keep doctests pure)
            # with open(fp, "r", encoding="utf-8") as f:
            #     content = f.read()
            # For doctest, simulate missing file:
            content = ""
        return Text(content)

    @reg.register(Text, JSONDict, cost=1.0)
    def text_to_json(t: Text, ctx: Optional[dict]) -> JSONDict:
        return JSONDict(json.loads(str(t) or "{}"))

    @reg.register(JSONDict, CanonicalRecord, cost=1.0)
    def json_to_canonical(d: JSONDict, ctx: Optional[dict]) -> CanonicalRecord:
        # Simple normalization step
        return CanonicalRecord(d)

    @reg.register(Text, CanonicalRecord, cost=0.5)  # cheaper direct route
    def text_to_canonical(t: Text, ctx: Optional[dict]) -> CanonicalRecord:
        return CanonicalRecord(json.loads(str(t) or "{}"))

    return reg

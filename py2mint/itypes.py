from typing import NewType, Optional, Iterable


def new_type(name, tp, doc: Optional[str] = None, aka: Optional[Iterable] = None):
    """Make a new type with (optional) doc and (optional) aka, set of var names it often appears as"""
    new_tp = NewType(name, tp)
    if doc is not None:
        setattr(new_tp, '__doc__', doc)
    if aka is not None:
        setattr(new_tp, '_aka', set(aka))
    globals()[name] = new_tp

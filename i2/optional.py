"""
Optional dependencies: import a module that an extra provides, and when it is
missing say which extra to install.

A package keeps heavy or rarely used dependencies out of its install requirements
and lists them under ``[project.optional-dependencies]``. The code that needs one
imports it where it uses it, with :func:`optional_import`, so that a bare install
still imports and the failure, when it comes, names the pip line that fixes it.
Bind the package's own name and its module-to-extra map once:

>>> from functools import partial
>>> my_import = partial(optional_import, package="mypkg", extras={"torch": "local"})
>>> my_import("math").pi
3.141592653589793
>>> my_import("torch_nope")                      # doctest: +ELLIPSIS
Traceback (most recent call last):
  ...
ImportError: This needs torch_nope: pip install torch_nope
>>> optional_import("torch_nope", package="mypkg", extra="local")  # doctest: +ELLIPSIS
Traceback (most recent call last):
  ...
ImportError: This needs torch_nope: pip install 'mypkg[local]'

Absent and broken are different failures. A module that is installed but fails on
one of *its own* imports raises that error unchanged: telling the user to install
what they already have sends them to pip, which reports success and changes
nothing.
"""

import importlib
from collections.abc import Mapping
from types import ModuleType


def _is_or_has_parent(missing: str, module: str) -> bool:
    return module == missing or module.startswith(missing + ".")


def install_command(
    module: str,
    *,
    package: str | None = None,
    extra: str | None = None,
    extras: Mapping[str, str] | None = None,
) -> str:
    """The pip line that provides ``module``.

    ``extra`` (else ``extras[module]``, else ``extras`` of its top-level package)
    names the extra of ``package`` that does; without one, the module's own name.

    >>> install_command("torch", package="mypkg", extras={"torch": "local"})
    "pip install 'mypkg[local]'"
    >>> install_command("torch.nn", package="mypkg", extras={"torch": "local"})
    "pip install 'mypkg[local]'"
    >>> install_command("PIL")
    'pip install PIL'
    """
    extras = extras or {}
    extra = extra or extras.get(module) or extras.get(module.split(".")[0])
    if package and extra:
        return f"pip install '{package}[{extra}]'"
    return f"pip install {module.split('.')[0]}"


def optional_import(
    module: str,
    *,
    package: str | None = None,
    extra: str | None = None,
    extras: Mapping[str, str] | None = None,
) -> ModuleType:
    """Import ``module``, or raise an ``ImportError`` saying how to install it.

    Args:
        module: The (dotted) module to import.
        package: The distribution whose extra provides it (for the pip line).
        extra: That extra's name.
        extras: A module-to-extra map, for a package that binds it once
            (``functools.partial``) and imports many modules.

    The error's ``name`` is ``module``. A module that is installed but whose own
    imports fail re-raises that error unchanged.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        if e.name and not _is_or_has_parent(e.name, module):
            raise
        hint = install_command(module, package=package, extra=extra, extras=extras)
        raise ImportError(f"This needs {module}: {hint}", name=module) from e

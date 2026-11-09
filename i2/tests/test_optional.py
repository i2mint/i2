"""``optional_import``: the pip line, and absent versus broken."""

import sys
from functools import partial

import pytest

from i2 import optional_import
from i2.optional import install_command


def test_present_module_is_imported():
    assert optional_import("json.tool").__name__ == "json.tool"


def test_missing_module_names_the_extra():
    with pytest.raises(ImportError, match=r"pip install 'mypkg\[local\]'") as err:
        optional_import("no_such_mod_xyz", package="mypkg", extra="local")
    assert err.value.name == "no_such_mod_xyz"


def test_bound_map_resolves_dotted_modules_by_top_level_package():
    imp = partial(optional_import, package="mypkg", extras={"no_such_mod_xyz": "x"})
    with pytest.raises(ImportError, match=r"mypkg\[x\]"):
        imp("no_such_mod_xyz.sub")


def test_explicit_extra_beats_the_map():
    assert install_command("m", package="p", extra="a", extras={"m": "b"}) == (
        "pip install 'p[a]'"
    )


def test_unknown_extra_falls_back_to_the_module_name():
    assert install_command("m.sub", package="p") == "pip install m"


def test_installed_module_failing_on_its_own_import_is_not_called_missing(
    tmp_path, monkeypatch
):
    (tmp_path / "broken_pkg_xyz.py").write_text("import its_missing_dep_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("broken_pkg_xyz", None)
    with pytest.raises(ImportError) as err:
        optional_import("broken_pkg_xyz", package="p", extra="e")
    assert err.value.name == "its_missing_dep_xyz"
    assert "pip install" not in str(err.value)

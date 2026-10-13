"""Test utils"""

import pickle
from i2.signatures import Sig
from i2.wrapper import wrap
from i2.util import copy_func
import tempfile
import os

from i2.util import FileLikeObject


def test_file_pointer():
    # Create a temporary file and write some content to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"Test content")
        temp_file_path = temp_file.name

    try:
        # Case 1: file is a string (file path)
        with FileLikeObject(temp_file_path) as f:
            assert f.read() == b"Test content"

        # Case 2: file is bytes
        with FileLikeObject(b"Test content") as f:
            assert f.read() == b"Test content"

        # Case 3: file is an open file pointer
        with open(temp_file_path, "rb") as file:
            with FileLikeObject(file) as f:
                assert f.read() == b"Test content"
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def test_copy_func_with_wrap(copy_func=copy_func):
    f = lambda x, *, y=2: x * y
    wrapped_f = wrap(f)  # empty wrap
    assert Sig(copy_func(wrapped_f)) == Sig(f)
    assert wrapped_f(3) == f(3) == 6


def is_pickable(obj):
    try:
        _ = pickle.loads(pickle.dumps(obj))
        return True
    except Exception:
        return False


def unpickled_func_still_works(func, *args, **kwargs):
    expected = func(*args, **kwargs)
    unpickled_func = pickle.loads(pickle.dumps(func))
    result = unpickled_func(*args, **kwargs)
    return expected == result


def test_get_app_folder_ignores_empty_and_relative_env_var(monkeypatch, tmp_path):
    """XDG spec: an empty or relative value is ignored (i2mint/i2#98)."""
    import os
    import pytest
    from i2.util import get_app_folder, APP_FOLDER_STANDARDS

    env_var, default = APP_FOLDER_STANDARDS["config"]
    if os.name == "nt":
        pytest.skip("XDG semantics are for POSIX")
    expected_default = os.path.expanduser(default)

    monkeypatch.delenv(env_var, raising=False)
    assert get_app_folder("config") == expected_default
    for bad in ("", "relative/dir", "."):
        monkeypatch.setenv(env_var, bad)
        assert get_app_folder("config") == expected_default
    monkeypatch.setenv(env_var, str(tmp_path))
    assert get_app_folder("config") == str(tmp_path)
    # ``~`` is still expanded (backward compatible)
    monkeypatch.setenv(env_var, "~/somewhere")
    assert get_app_folder("config") == os.path.expanduser("~/somewhere")

"""``i2.Pipe`` and ``dol.Pipe`` keep one call contract (i2mint/i2#97, option a).

Both packages have zero dependencies, so neither re-exports the other. Instead this
test pins the behaviour they share: plain composition gives the same result, the
same signature, the same keyword handling, the same error when empty, and both
pickle. ``i2.Pipe`` adds named, inspectable stages on top; ``dol.Pipe`` is the plain
callable. The test is skipped where dol is not installed.
"""

import inspect
import operator
import pickle

import pytest

from i2.multi_object import Pipe as I2Pipe

DolPipe = pytest.importorskip("dol.util").Pipe

PIPES = pytest.mark.parametrize("Pipe", [I2Pipe, DolPipe], ids=["i2", "dol"])


def _first(a, b=2, *, c=1):
    return a + b + c


@PIPES
def test_composition_result_and_signature(Pipe):
    pipe = Pipe(_first, str)
    assert pipe(1) == "4" and pipe(1, 3, c=0) == "4"
    assert inspect.signature(pipe) == inspect.signature(_first)


@PIPES
def test_a_single_function_and_keywords(Pipe):
    assert Pipe(abs)(-2) == 2
    assert Pipe(_first)(1, c=5) == 8


@PIPES
def test_an_empty_pipe_is_refused(Pipe):
    with pytest.raises(ValueError):
        Pipe()


@PIPES
def test_pickles(Pipe):
    assert pickle.loads(pickle.dumps(Pipe(operator.neg, abs)))(3) == 3


def test_both_agree_on_the_same_stages():
    stages = (_first, lambda x: x * 10, str)
    assert I2Pipe(*stages)(1, 2, c=3) == DolPipe(*stages)(1, 2, c=3) == "60"

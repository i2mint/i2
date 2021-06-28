"""Meta-programming tools to build declarative frameworks"""
from i2.deco import (
    preprocess,
    postprocess,
    preprocess_arguments,
    input_output_decorator,
)

from i2.deco import (
    preprocess,
    postprocess,
    preprocess_arguments,
    input_output_decorator,
)
from i2.deco import wrap_class_methods_input_and_output
from i2.signatures import (
    Sig,
    Command,
    extract_commands,
    call_forgivingly,
    call_somewhat_forgivingly,
)

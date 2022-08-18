"""Meta-programming tools to build declarative frameworks"""
from i2.deco import (
    preprocess,
    postprocess,
    preprocess_arguments,
    input_output_decorator,
    wrap_class_methods_input_and_output,
    double_up_as_factory,
)

from i2.signatures import (
    Sig,
    Param,
    sort_params,
    call_forgivingly,
    call_somewhat_forgivingly,
    name_of_obj,
    empty as empty_param_attr,
)

from i2.multi_object import (
    MultiObj,
    MultiFunc,
    Pipe,
    FuncFanout,
    FlexFuncFanout,
    ParallelFuncs,
    ContextFanout,
)

from i2.errors import (
    InterruptWithBlock,
    HandleExceptions,
)

from i2.wrapper import (
    wrap,
    Wrap,
    Wrapx,
    Ingress,
    include_exclude,
    rm_params,
    partialx,
    ch_names,
    func_to_method_func,
    bind_funcs_object_attrs,
    kwargs_trans,
)

from i2.util import (
    Literal,
    path_extractor,
    get_function_body,
    lazyprop,
    frozendict,
    inject_method,
    mk_sentinel,
    ensure_identifiers,
)

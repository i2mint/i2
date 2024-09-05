"""Meta-programming tools to build declarative frameworks"""

from i2.deco import (
    FuncFactory,
    preprocess,
    postprocess,
    preprocess_arguments,
    input_output_decorator,
    wrap_class_methods_input_and_output,
    double_up_as_factory,
)

from i2.signatures import (
    Sig,  # An extended Signature object
    Param,
    sort_params,
    call_forgivingly,  # Call a function extracting the arguments from a pool of arguments
    call_somewhat_forgivingly,  # call_forgivingly with a bit more control
    name_of_obj,  # Get the name of an object, and control how it's done
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
    asis,  # the identity function: f(x) := x (takes only one argument, and returns it)
    return_true,  # a function that returns True (takes any number of arguments)
    return_false,  # a function that returns False (takes any number of arguments)
    ConditionalExceptionCatcher,  # A context manager that catches exceptions based on a condition.
    Namespace,  # A namespace that is also a mutable mapping.
    copy_func,  # Copy a function.
    get_app_data_folder,  # Get the application data folder of the current system.
    LiteralVal,  # An object to indicate that the value should be considered literally.
    path_extractor,  # Get items from a tree-structured object from a sequence of tree-traversal indices.
    get_function_body,
    lazyprop,  # Like functools.cached_property, but with a bit more.
    frozendict,  # A hashable dictionary.
    inject_method,  # Inject a method into an object instance
    mk_sentinel,  # Make a sentinel instance.
    ensure_identifiers,  # Ensure that one or several strings are valid python identifiers.
)

identity = asis  # alias for asis, the identity function

from i2.footprints import MethodTrace

from i2.itypes import validate_literal, ObjectClassifier

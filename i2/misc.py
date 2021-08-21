import inspect
from inspect import Signature, Parameter

from i2.base import MintOfCallable
from functools import partial, wraps


def inject_signature(
    sig, *, return_annotation=inspect._empty, __validate_parameters__=True
):
    if not isinstance(sig, inspect.Signature):
        sig = mk_signature_from_dict_specs(
            sig,
            return_annotation=return_annotation,
            __validate_parameters__=__validate_parameters__,
        )
    assert isinstance(
        sig, inspect.Signature
    ), 'sig should be an inspect.Signature (or be resolved to one)'

    def wrapper(func):
        func.__signature__ = sig
        return func

    return wrapper


def dict_only_with_specific_keys(d, keys):
    new_d = dict()
    for k in keys:
        if k in d:
            new_d[k] = d[k]
    return new_d


def filter_by_value(d, valfunc):
    return {k: v for k, v in d.items() if valfunc(v)}


# def mk_arg_name_dflt_annot_dict_list_from_func(func):
#     func_mint = MintOfCallable(func)
#     extractor = partial(dict_only_with_specific_keys, keys=('name', 'kind', 'default', 'annotation'))
#     val_filt = partial(filter_by_value, valfunc=lambda x: x is not inspect._empty)
#     arg_name_default_annot = list(map(extractor, dict(func_mint.parameters).values()))
#     arg_name_default_annot = list(map(val_filt, arg_name_default_annot))
#     return arg_name_default_annot

parameter_props = ['name', 'kind', 'default', 'annotation']


def mk_arg_name_dflt_annot_dict_list_from_func(func):
    s = inspect.signature(func)
    params_dict_list = list()
    for p in s.parameters.values():
        d = dict()
        for prop in parameter_props:
            prop_val = getattr(p, prop)
            if prop_val is not inspect._empty:
                d[prop] = prop_val
        params_dict_list.append(d)
    return params_dict_list


dflt_params = dict(
    kind=Parameter.POSITIONAL_OR_KEYWORD,
    default=Parameter.empty,
    annotation=Parameter.empty,
)


def mk_signature_from_dict_specs(
    arg_name_default_annot=(),
    *,
    return_annotation=inspect._empty,
    __validate_parameters__=True
):
    """

    :param arg_name_default_annot:
    :param return_annotation:
    :param __validate_parameters__:
    :return:

    >>> def foo(a, b: int, c=0, d:float=1.0) -> float:
    ...     return a + (c * b) ** d
    >>> params = mk_arg_name_dflt_annot_dict_list_from_func(foo)
    >>> s = mk_signature_from_dict_specs(params, return_annotation=float)
    >>> print(s)
    (a, b: int, c=0, d: float = 1.0) -> float
    """
    parameters = list()
    for d in arg_name_default_annot:
        d = dict(dflt_params, **d)
        parameters.append(Parameter(**d))
    return Signature(
        parameters=parameters,
        return_annotation=return_annotation,
        __validate_parameters__=__validate_parameters__,
    )


class SignatureFactory:
    dflt_params = dict(
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=Parameter.empty,
    )

    def __init__(self, **dflt_arg_specs):
        self.dflt_arg_specs = dflt_arg_specs

    def __call__(
        self, params, *, return_annotation=inspect._empty, __validate_parameters__=True
    ):
        parameters = list()
        for param in params:
            if not isinstance(param, Parameter):
                if isinstance(
                    param, str
                ):  # then assume param is the name of an argument
                    name = param
                    param_dict = dict(
                        self.dflt_params, **self.dflt_arg_specs.get(name, {})
                    )
                    if 'name' not in param_dict:
                        param_dict['name'] = name
                elif isinstance(param, dict):
                    param_dict = dict(
                        self.dflt_params, **dict(self.dflt_arg_specs, **param)
                    )
                else:
                    raise ValueError()
                param = Parameter(**param_dict)
            parameters.append(param)
        return Signature(
            parameters=parameters,
            return_annotation=return_annotation,
            __validate_parameters__=__validate_parameters__,
        )

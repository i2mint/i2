from inspect import Signature, Parameter, _empty, _ParameterKind

dflt_name_for_kind = {
    Parameter.VAR_POSITIONAL: 'args',
    Parameter.VAR_KEYWORD: 'kwargs',
}

arg_order_for_param_tuple = ('name', 'default', 'annotation', 'kind')


def mk_param(param, dflt_kind=Parameter.POSITIONAL_OR_KEYWORD):
    """
    Make an inspect.Parameter object with less boilerplate verbosity.
    Args:
        param: Can be an inspect.Parameter itself, or something resolving to it:
            - a string: Will be considered to be the argument name
            - an inspect._ParameterKind type: Can only be Parameter.VAR_KEYWORD or Parameter.VAR_POSITIONAL,
                will be taken as the kind, and the name will be decided according to the dflt_name_for_kind mapping.
            - a tuple: Will use the arg_order_for_param_tuple to resolve what Parameter constructor arguments
                the elements of the tuple correspond to.
        dflt_kind: Default kind (of type _ParameterKind) to use if no other kind is specified.

    Returns:
        An inspect.Parameter object

    See mk_signature for uses
    """
    if isinstance(param, str):  # then consider param to be the argument name
        param = dict(name=param)
    elif isinstance(param, _ParameterKind):  # then consider param to be the argument kind
        name = dflt_name_for_kind.get(param, None)
        if name is not None:
            param = dict(name=name, kind=param)
        else:
            raise ValueError("If param is an inspect._ParameterKind, is must be VAR_POSITIONAL or VAR_KEYWORD")
    elif isinstance(param, tuple):
        param = dict(zip(arg_order_for_param_tuple, param))

    if isinstance(param, dict):
        param = dict({'kind': dflt_kind}, **param)
        param = Parameter(**param)

    assert isinstance(param, Parameter), "param should be an inspect.Parameter at this point!"
    return param


def mk_signature(signature, *, return_annotation=_empty, __validate_parameters__=True):
    """
    Make an inspect.Signature object with less boilerplate verbosity.
    Args:
        signature: A list of parameter specifications. This could be an inspect.Parameter object or anything that
            the mk_param function can resolve into an inspect.Parameter object.
        return_annotation: Passed on to inspect.Signature.
        __validate_parameters__: Passed on to inspect.Signature.

    Returns:
        An inspect.Signature object

    >>> mk_signature(['a', 'b', 'c'])
    <Signature (a, b, c)>
    >>> mk_signature(['a', ('b', None), ('c', 42, int)])  # specifying defaults and annotations
    <Signature (a, b=None, c: int = 42)>
    >>> mk_signature(['a', 'b', 'c'], return_annotation=str)  # specifying return annotation
    <Signature (a, b, c) -> str>
    >>>
    >>> # But you can always specify parameters the "long" way
    >>> import inspect
    >>> mk_signature([inspect.Parameter(name='kws', kind=inspect.Parameter.VAR_KEYWORD)], return_annotation=str)
    <Signature (**kws) -> str>
    """
    if not isinstance(signature, Signature):
        signature = Signature(parameters=list(map(mk_param, signature)),
                              return_annotation=return_annotation,
                              __validate_parameters__=__validate_parameters__)

    assert isinstance(signature, Signature)
    return signature


def set_signature_of_func(func, signature, *, return_annotation=_empty, __validate_parameters__=True):
    """
    Set the signature of a function, with sugar.

    Args:
        func: Function whose signature you want to set
        signature: A list of parameter specifications. This could be an inspect.Parameter object or anything that
            the mk_param function can resolve into an inspect.Parameter object.
        return_annotation: Passed on to inspect.Signature.
        __validate_parameters__: Passed on to inspect.Signature.

    Returns:
        None (but sets the signature of the input function)

    >>> import inspect
    >>> def foo(*args, **kwargs):
    ...     pass
    ...
    >>> inspect.signature(foo)
    <Signature (*args, **kwargs)>
    >>> set_signature_of_func(foo, ['a', 'b', 'c'])
    >>> inspect.signature(foo)
    <Signature (a, b, c)>
    >>> set_signature_of_func(foo, ['a', ('b', None), ('c', 42, int)])  # specifying defaults and annotations
    >>> inspect.signature(foo)
    <Signature (a, b=None, c: int = 42)>
    >>> set_signature_of_func(foo, ['a', 'b', 'c'], return_annotation=str)  # specifying return annotation
    >>> inspect.signature(foo)
    <Signature (a, b, c) -> str>
    >>> # But you can always specify parameters the "long" way
    >>> set_signature_of_func(
    ...  foo,
    ...  [inspect.Parameter(name='kws', kind=inspect.Parameter.VAR_KEYWORD)], return_annotation=str)
    >>> inspect.signature(foo)
    <Signature (**kws) -> str>

    """
    func.__signature__ = mk_signature(signature,
                                      return_annotation=return_annotation,
                                      __validate_parameters__=__validate_parameters__)
    # Not returning func so it's clear(er) that the function is transformed in place

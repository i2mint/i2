"""Scrap"""
from functools import wraps
from typing import Iterable, Callable, Tuple
from i2.wrapper import Wrap
from i2 import call_forgivingly, Sig

# NOTE: Wrapx is now in i2.wrapper and test in i2.tests.wrapper_test


def default_caller(
    func: Callable, func_args: tuple, func_kwargs: dict, **caller_params
):
    """Function is called normally. caller_kwargs ignored"""
    return func(*func_args, **func_kwargs)


def _extract_params(ingress_output) -> Tuple[tuple, dict, dict, dict]:
    """Pads tuple with empty dicts"""
    if isinstance(ingress_output, tuple) and len(ingress_output) == 2:
        # This is the "normal" ingress protocol, so transform to the extended
        return ingress_output[0], ingress_output[1], {}, {}
    elif isinstance(ingress_output, dict):
        # This is the extended ingress protocol
        return (
            ingress_output.get('inner_args', ()),
            ingress_output.get('inner_kwargs', {}),
            ingress_output.get('caller_kwargs', {}),
            ingress_output.get('egress_kwargs', {}),
        )
        # TODO: Consider making a ingress_output object to avoid key-typo errors
        # TODO: Should we assert those were the only keys to mitigate key-typo errors?
    else:
        raise ValueError(
            f'ingress_output should be a 2-tuple or a dict. Was {ingress_output}'
        )


# TODO: Create a few higher-level constructors
# TODO: Add static analysis to ascertain protocols
class Wrapx1(Wrap):
    """An extended ``Wrap`` object that enables control over ``caller`` & ``egress``.

    To get the extended functionality, the ``ingress`` function must return a dict with
    keys ``inner_args, inner_kwargs, caller_kwargs`` and/or  ``egress_kwargs``.
    If not specified, the ``inner_args`` value will default to the empty tuple ``()``,
    and all others to the empty dictionary ``{}``.

    The additional control arguments be keyword-only in ``ingress`` as well as in the
    ``caller`` and/or ``egress`` they control.

    >>> from inspect import signature
    >>> def func(x, y):
    ...     return x + y

    Let's wrap the function endowing the wrapped function with an extra argument, ``z``,
    which can be used to multiply the output.

    >>> def ingress(x, y, *, z):
    ...     return dict(inner_args=(x, y), egress_kwargs=dict(z=z))
    >>> def egress(v, *, z):
    ...     return v * z
    ...
    >>> wrapped_func = Wrapx1(func, ingress=ingress, egress=egress)
    >>>
    >>> func(1, 2)
    3
    >>> str(signature(wrapped_func))
    '(x, y, *, z)'
    >>> assert wrapped_func(1, 2, z=3) == 9 == func(1, 2) * 3

    Here's a bit more realistic application. Say we want to add two arguments,
    ``s`` (a mapping) and ``k`` (a key) that will have the effect of writing outputs
    of the function in ``s[k]``.

    >>> def _saving_ingress(x, y, *, k, s):
    ...     return dict(inner_args=(x, y), egress_kwargs=dict(k=k, s=s))
    >>> def _saving_egress(v, *, k, s):
    ...     s[k] = v
    ...     return v
    >>> save_on_output = Wrapx1(func, ingress=_saving_ingress, egress=_saving_egress)
    >>> str(signature(save_on_output))
    '(x, y, *, k, s)'
    >>>
    >>> store = dict()
    >>> assert save_on_output(1, 2, k='save_here', s=store) == 3 == func(1, 2)
    >>> store
    {'save_here': 3}

    An graph exhibiting how it works:

    .. image:: https://user-images.githubusercontent.com/1906276/159377430-e67f4fb7-22f2-44cd-97f8-6ddbbe53f66d.png


    """

    def __init__(
        self, func, ingress=None, egress=None, *, name=None, caller=default_caller
    ):
        super().__init__(func, ingress=ingress, egress=egress, name=name)
        self.caller = caller

    def __call__(self, *outer_args, **outer_kwargs):
        func_args, func_kwargs, caller_params, egress_params = _extract_params(
            self.ingress(*outer_args, **outer_kwargs)
        )
        # call the function
        func_output = self.caller(self.func, func_args, func_kwargs, **caller_params)
        # process output with egress
        return self.egress(func_output, **egress_params)


def _pad_with_empty_dicts(tup, target_length=4):
    """Pads tuple with empty dicts"""
    return tup + tuple([{}] * max(0, target_length - len(tup)))


# TODO: Add static analysis to ascertain dynamic _pad_with_empty_dicts won't lead to
#  trouble
class Wrapx2(Wrap):
    """
    >>> from inspect import signature
    >>> def func(x, y):
    ...     return x + y
    >>> def ingress(x, y, *, z):
    ...     return (x, y), {}, dict(z=z), {}
    >>> def egress(v, *, z):
    ...     return v * z
    >>> wrapped_func = Wrapx2(func, ingress=ingress, egress=egress)
    >>>
    >>> assert func(1, 2) == 3
    >>> assert str(signature(wrapped_func)) == '(x, y, *, z)'
    >>> assert wrapped_func(1, 2, z=3) == 9 == func(1, 2) * 3
    """

    def __init__(
        self, func, ingress=None, egress=None, *, name=None, caller=default_caller
    ):
        super().__init__(func, ingress=ingress, egress=egress, name=name)
        self.caller = caller

    def __call__(self, *outer_args, **outer_kwargs):
        func_args, func_kwargs, egress_params, caller_params = _pad_with_empty_dicts(
            self.ingress(*outer_args, **outer_kwargs)
        )
        # call the function
        func_output = self.caller(self.func, func_args, func_kwargs, **caller_params)
        # process output with egress
        return self.egress(func_output, **egress_params)


# from i2.deco import double_up_as_factory
#
# # TODO: Other design options? Example making an Egress class that has extract_kwargs meth
# # TODO: Make "copy" of function before adding extract_kwargs attr, or let user do it
# #  if it matters? (I vote let user)
# # TODO: Add ability to not pop extracted
# # TODO: Postelize extract_kwargs further?
# @double_up_as_factory
# def add_extract_kwargs(func=None, *, extract_kwargs):
#     if isinstance(extract_kwargs, Iterable):
#         keys_to_extract = tuple(extract_kwargs)
#         extract_kwargs = partial(_extract_kwargs, keys_to_extract=keys_to_extract)
#     if callable(extract_kwargs) and Sig(extract_kwargs).n_required <= 1:
#         setattr(func, 'extract_kwargs', extract_kwargs)
#     else:
#         setattr(func, 'extract_kwargs', _return_empty_dict)
#
#     return func
#
#
# def _return_empty_dict(kwargs):
#     """Used when we want to have an 'empty' extract_kwargs on an egress or caller"""
#     return dict()
#
#
# def _extract_kwargs(kwargs, keys_to_extract):
#     return {k: kwargs.pop(k) for k in keys_to_extract}
#
#
# @add_extract_kwargs(extract_kwargs=_return_empty_dict)
# def default_caller(
#     func: Callable, func_args: tuple, func_kwargs: dict, caller_kwargs
# ):
#     """Function is called normally. caller_kwargs ignored"""
#     return func(*func_args, **func_kwargs)
#
#
# # TODO: Better have an egress that always has a (potentially empty) extract_kwargs or
# #  check for existence at runtime?
# class Wrapxx(Wrap):
#     def __init__(
#             self, func, ingress=None, egress=None, *, name=None, caller=default_caller
#     ):
#         super().__init__(func, ingress=ingress, egress=egress, name=name)
#         self.caller = caller
#         # current_sig = Sig(self)
#         # # extended_sig = Sig(self) +
#         # self.__signature__ = Sig(
#         #     extended_sig, return_annotation=current_sig.return_annotation
#         # )
#
#     def __call__(self, *user_args, **user_kwargs):
#         # extract the kwargs the egress needs from user input
#         egress_kwargs = self.egress.extract_kwargs(user_kwargs)
#         # extract the kwargs the caller needs from user input
#         caller_kwargs = self.caller.extract_kwargs(user_kwargs)
#         # prepare function inputs
#         func_args, func_kwargs = self.ingress(*user_args, **user_kwargs)
#         # call the function
#         func_output = self.caller(self.func, func_args, func_kwargs, caller_kwargs)
#         # process output with egress
#         return self.egress(func_output, **egress_kwargs)
#

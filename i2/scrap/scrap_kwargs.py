from i2 import Sig
from i2.signatures import _remove_variadics_from_sig, ch_variadics_to_non_variadic_kind
from i2.wrapper import InnerMapIngress
from i2.wrapper import wrap
from inspect import Parameter, Signature


# def bar(w, /, x: float, y=1, *args, z: int = 1, **rest):
#     return ((w + x) * y) ** z


# bar_sig = Sig(bar)


def foo(po, *vp, ko, **vk):
    return f'{po=}, {vp=}, {ko=}, {vk=}'


new_sig = _remove_variadics_from_sig(Sig(foo))
assert str(new_sig) == '(po, vp=(), *, ko, vk={})'

ingress = InnerMapIngress.from_signature(foo, outer_sig=new_sig)

if __name__ == '__main__':
    new_foo = wrap(foo, ingress=ingress)
    # ingress_args = (1, (2, 3))
    # ingress_kwargs = {"ko": 4, "vk": {"hello": "world"}}
    # print(ingress(*ingress_args, **ingress_kwargs))

    assert (
        foo(1, 2, 3, ko=4, hello='world')
        == "po=1, vp=(2, 3), ko=4, vk={'hello': 'world'}"
    )
    new_foo = wrap(foo, ingress=ingress)
    assert str(Sig(new_foo)) == '(po, vp=(), *, ko, vk={})'
    # assert (
    #     new_foo(1, (2, 3), ko=4, vk=dict(hello="world"))
    #     == "po=1, vp=(2, 3), ko=4, vk={'hello': 'world'}"
    # )
    # args, kwargs = foo_sig.args_and_kwargs_from_kwargs(dict(w=4, x=3, y=2, z=1, t=12))

    # print(f"args:{args}, kwargs: {kwargs}")
    # print(new_foo(1, (2, 3), ko=4, vk=dict(hello="world")))

from collections import ChainMap
from collections.abc import Mapping
import operator
import inspect


class AsIsMap:
    def __init__(self, is_valid_key=None):
        if is_valid_key is None:

            def is_valid_key(x):
                return True

        self._is_valid_key = is_valid_key

    def _validate_key(self, k):
        if not self._is_valid_key(k):
            raise KeyError(f"{k} wasn't a valid key")

    def __getitem__(self, k):
        self._validate_key(k)
        return k


class AttrMap(Mapping):
    def __init__(self, obj, is_valid_val=None):
        self._obj = obj
        if is_valid_val is None:

            def is_valid_val(x):
                return True

        self._is_valid_val = is_valid_val

    @classmethod
    def _validate_key(cls, k):
        if not isinstance(k, str):
            raise KeyError('key should be a string')

    def _validate_val(self, v):
        if not self._is_valid_val(v):
            raise ValueError("key was valid and value found, but value wasn't valid")

    def _getitem(self, k):
        return getattr(self._obj, k)

    def _val_of_key_is_valid(self, k):
        return self._is_valid_val(self._getitem(k))

    def __getitem__(self, k):
        self._validate_key(k)
        v = self._getitem(k)
        self._validate_val(v)
        return v

    def __contains__(self, k):
        self._validate_key(k)
        return hasattr(self._obj, k) and self._val_of_key_is_valid(k)

    def __iter__(self):
        return filter(self._val_of_key_is_valid, dir(self._obj))

    def __len__(self):
        c = 0
        for _ in self.__iter__():
            c += 1
        return c


def n_args_and_n_args_with_no_defaults(func):
    try:
        args = inspect.signature(func).parameters.values()
    except ValueError:  # happens because some builtins don't have signatures (!?!?)
        return 0, 0
    return len(args), len(list(filter(lambda x: x.default == x.empty, args)))


def is_valid_featurizer(func):
    if not callable(func):
        return False
    n_args, n_no_dflt_args = n_args_and_n_args_with_no_defaults(func)
    return (n_args >= 1) and (n_no_dflt_args <= 1)


def is_valid_comparision(func):
    if not callable(func):
        return False
    n_args, n_no_dflt_args = n_args_and_n_args_with_no_defaults(func)
    return (n_args >= 2) and (n_no_dflt_args <= 2)


def is_valid_feat_and_comp(feat, comp):
    return is_valid_featurizer(feat) and is_valid_comparision(comp)


def values_are_valid_feat_and_comp(d):
    return all(is_valid_feat_and_comp(feat, comp) for feat, comp in d.values())


def mk_filt(obj, featurizer, comparison):
    feature = featurizer(obj)

    def filt(x):
        return comparison(feature, x)

    return filt


##########################################################################################################
# from anytree import Node, AnyNode, RenderTree, ContStyle, NodeMixin
#
# NodeMixin.separator = '.'
#
#
# def get_attr_and_vals(obj):
#     for k in dir(obj):
#         v = getattr(obj, k)
#         yield k, v
#
#
# def attr_and_vals_dict(obj, filt=None):
#     return {k: v for k, v in filter(filt, get_attr_and_vals(obj))}
#
#
# # Filters
# def value_not_callable(kv):
#     return not callable(kv[1])
#
#
# class AttrTreeImporter:
#     _base_types = (int, float, str, bytes, bool, list)
#
#     def __init__(self, val_types=()):
#         self.val_types = tuple(set(val_types).union(self._base_types))
#
#     def dict_of_obj(self, obj):
#         d = dict()
#         for k, v in obj.__dict__.items():
#             if isinstance(v, self.val_types):
#                 d[k] = v
#             else:
#                 d[k] = self.dict_of_obj(v)
#         return d
#
#     def tree_of_dict(self, d, parent=None, root_name='root'):
#         if parent is None:
#             parent = Node(root_name)
#         for k, v in d.items():
#             if isinstance(v, dict):
#                 n = Node(k, parent=parent)
#                 self.tree_of_dict(v, parent=n)
#             else:
#                 n = Node(k, parent=parent, val=v)
#         return parent
#
#     def tree_of_obj(self, obj, root_name=None):
#         if root_name is None:
#             if hasattr(obj, '__name__'):
#                 root_name = obj.__name__
#             elif hasattr(obj, '__class__'):
#                 root_name = obj.__class__.__name__
#             else:
#                 root_name = 'root'
#         return self.tree_of_dict(self.dict_of_obj(obj), root_name=root_name)


if __name__ == '__main__':
    from collections import Counter

    print(
        '##########################################################################################################'
    )
    special_featurizer = {
        'len': len,
        'cols': lambda df: df.columns,
        'sum': lambda df: df.sum().sum(),
    }

    some_local_func = lambda x: list(map(str, x))

    featurizer = ChainMap(
        special_featurizer,
        {k: v for k, v in locals().items() if is_valid_featurizer(v)},
    )

    special_comparison = {
        'alleq': lambda x, y: all(x == y),
        'isin': lambda x, y: x in y,
        'eq': operator.eq,
    }

    comparison = ChainMap(
        special_comparison, AttrMap(operator, is_valid_val=is_valid_comparision),
    )

    assert comparison['contains'] == operator.contains

    print(Counter(map(is_valid_featurizer, featurizer.values())))
    print(Counter(map(is_valid_comparision, featurizer.values())))
    print(Counter(map(is_valid_featurizer, comparison.values())))
    print(Counter(map(is_valid_comparision, comparison.values())))

    print(f'featurizer: {featurizer}')

    from contextlib import suppress

    with suppress(ModuleNotFoundError, ImportError):
        import pandas as pd
        from collections import namedtuple

        print(
            '###########################################################################'
        )

        Condition = namedtuple('Condition', ['feat', 'comp'])
        condition = {
            feat + '_' + comp: Condition(featurizer[feat], comparison[comp])
            for feat, comp in [('len', 'lt'), ('cols', 'isin'), ('cols', 'contains'),]
        }
        assert all(
            is_valid_feat_and_comp(feat, comp) for feat, comp in condition.values()
        )

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        filt = mk_filt(df, *condition['len_lt'])
        print(list(filter(filt, [2, 3, 4, 5])))

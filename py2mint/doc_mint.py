import doctest

MAX_LINE_LENGTH = 72  # https://en.wikipedia.org/wiki/Characters_per_line


def _prefix_lines(s: str, prefix: str = '# ', even_if_empty: bool = False) -> str:
    r"""
    Prefix every line of s with given prefix.

    :param s: String whose lines you want to prefix.
    :param prefix: Desired prefix string. Default is '# ', to have the effect of "commenting out" line
    :param even_if_empty: Whether to prefix empty strings or not.
    :return: A string whose lines have been prefixed.
    >>> _prefix_lines('something to comment out')
    '# something to comment out'
    >>> _prefix_lines('A line you want to prefix', prefix='PREFIX: ')
    'PREFIX: A line you want to prefix'
    >>> print(_prefix_lines('What happens\nif the thing to comment out\nhas multiple lines?'))
    # What happens
    # if the thing to comment out
    # has multiple lines?
    >>> _prefix_lines('')  # see that an empty string is returned as is
    ''
    >>> _prefix_lines('', even_if_empty=True)  # unless you ask for it
    '# '
    """
    if not even_if_empty:
        if len(s) == 0:
            return s
    return '\n'.join(map(lambda x: prefix + x, s.split('\n')))


def doctest_string(obj, output_prefix='# OUTPUT: ', recurse=True):
    """
    Extract the doctests found in given object.
    :param obj: Object (module, class, function, etc.) you want to extract doctests from.
    :params output_prefix:
    :param recurse: Whether the process should find doctests in the attributes of the object, recursively.
    :return: A string containing the doctests, with output lines prefixed by '# Output:'
    """
    doctest_finder = doctest.DocTestFinder(verbose=False, recurse=recurse)
    r = doctest_finder.find(obj)
    s = ''
    for rr in r:
        ss = f'# {rr.name} '
        ss += '#' * max(0, MAX_LINE_LENGTH - len(ss)) + '\n'
        for example in rr.examples:
            want = example.want
            if want.endswith('\n'):
                want = want[:-1]
            ss += '\n' + example.source + _prefix_lines(want, prefix=output_prefix)
        s += ss
    return s



if __name__ == '__main__':
    print(doctest_string(_prefix_lines))
# # _prefix_lines ########################################################
#
# _prefix_lines('something to comment out')
# # OUTPUT: '# something to comment out'
# _prefix_lines('A line you want to prefix', prefix='PREFIX: ')
# # OUTPUT: 'PREFIX: A line you want to prefix'
# print(_prefix_lines('What happens\nif the thing to comment out\nhas multiple lines?'))
# # OUTPUT: # What happens
# # OUTPUT: # if the thing to comment out
# # OUTPUT: # has multiple lines?
# _prefix_lines('')  # see that an empty string is returned as is
# # OUTPUT: ''
# _prefix_lines('', even_if_empty=True)  # unless you ask for it
# # OUTPUT: '# '
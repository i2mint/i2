"""Meta-interfaces """

from typing import Literal, List, Dict, Tuple
import re


def params_to_docstring(
    params: list[dict],
    *,
    doc_style: Literal["numpy", "google", "rest"] = "numpy",
) -> str:
    """
    Generate a docstring from a list of parameter specifications.

    :param params: A list of dictionaries where each dictionary specifies a parameter.
        Each dictionary should contain:
          - name: The name of the parameter (str).
          - default: The default value of the parameter (any, optional).
          - annotation: The type annotation for the parameter (str, optional).
          - description: A description of the parameter (str).
    :param doc_style: The style of the docstring to generate. One of 'numpy', 'google', or 'rest'.

    :return: A formatted docstring (str).

    Examples:
    >>> params = [
    ...     {"name": "x", "default": 1, "annotation": "int", "description": "An integer value."},
    ...     {"name": "y", "default": None, "annotation": "str", "description": "An optional string."},
    ... ]
    >>> print(params_to_docstring(params))  # doctest: +NORMALIZE_WHITESPACE
    Parameters
    ----------
    x : int, default=1
        An integer value.
    y : str, default=None
        An optional string.
    <BLANKLINE>
    >>> print(params_to_docstring(params, doc_style='google'))  # doctest: +NORMALIZE_WHITESPACE
    Args:
        x (int, optional): An integer value. Defaults to 1.
        y (str, optional): An optional string. Defaults to None.
    <BLANKLINE>
    >>> print(params_to_docstring(params, doc_style='rest'))  # doctest: +NORMALIZE_WHITESPACE
    :param x: An integer value. (Default: 1)
    :type x: int
    :param y: An optional string. (Default: None)
    :type y: str
    <BLANKLINE>
    """
    if doc_style == "numpy":
        docstring_lines = ["Parameters", "----------"]
        for param in params:
            name = param["name"]
            annotation = param.get("annotation", "")
            description = param.get("description", "")
            default = param.get("default")

            param_line = name
            if annotation:
                param_line += f" : {annotation}"
            if "default" in param:
                param_line += f", default={default}"

            docstring_lines.append(param_line)
            docstring_lines.append(f"    {description}")

        docstring = "\n".join(docstring_lines)
        return docstring + "\n"

    elif doc_style == "google":
        docstring_lines = ["Args:"]
        for param in params:
            name = param["name"]
            annotation = param.get("annotation", "")
            description = param.get("description", "")
            default = param.get("default")

            param_line = f"    {name}"
            if annotation:
                param_line += f" ({annotation}"
                if "default" in param:
                    param_line += ", optional"
                param_line += ")"
            elif "default" in param:
                param_line += " (optional)"

            if "default" in param:
                description += f" Defaults to {default}."

            param_line += f": {description}"
            docstring_lines.append(param_line)

        docstring = "\n".join(docstring_lines)
        return docstring + "\n"

    elif doc_style == "rest":
        docstring_lines = []
        for param in params:
            name = param["name"]
            annotation = param.get("annotation", "")
            description = param.get("description", "")
            default = param.get("default")

            if "default" in param:
                description += f" (Default: {default})"

            docstring_lines.append(f":param {name}: {description}")
            if annotation:
                docstring_lines.append(f":type {name}: {annotation}")

        docstring = "\n".join(docstring_lines)
        return docstring + "\n"

    else:
        raise ValueError(f"Unsupported doc_style: {doc_style}")


def docstring_to_params(
    docstring: str,
    *,
    doc_style: Literal["numpy", "google", "rest"] = "numpy",
) -> List[Dict]:
    """
    Parse a docstring and extract parameter specifications.

    :param docstring: The docstring to parse.
    :param doc_style: The style of the docstring to parse. One of 'numpy', 'google', or 'rest'.

    :return: A list of parameter specifications, where each specification is a dictionary containing:
        - name: The name of the parameter (str).
        - default: The default value of the parameter (str, optional).
        - annotation: The type annotation for the parameter (str, optional).
        - description: A description of the parameter (str).

    Examples:
    >>> docstring = '''
    ... Parameters
    ... ----------
    ... x : int, default=1
    ...     An integer value.
    ... y : str, default=None
    ...     An optional string.
    ... '''
    >>> params = docstring_to_params(docstring)
    >>> params == [
    ...     {"name": "x", "default": "1", "annotation": "int", "description": "An integer value."},
    ...     {"name": "y", "default": "None", "annotation": "str", "description": "An optional string."},
    ... ]
    True

    >>> docstring = '''
    ... Args:
    ...     x (int, optional): An integer value. Defaults to 1.
    ...     y (str, optional): An optional string. Defaults to None.
    ... '''
    >>> params = docstring_to_params(docstring, doc_style='google')
    >>> params == [
    ...     {"name": "x", "default": "1", "annotation": "int", "description": "An integer value."},
    ...     {"name": "y", "default": "None", "annotation": "str", "description": "An optional string."},
    ... ]
    True

    >>> docstring = '''
    ... :param x: An integer value. (Default: 1)
    ... :type x: int
    ... :param y: An optional string. (Default: None)
    ... :type y: str
    ... '''
    >>> params = docstring_to_params(docstring, doc_style='rest')
    >>> params == [
    ...     {"name": "x", "default": "1", "annotation": "int", "description": "An integer value."},
    ...     {"name": "y", "default": "None", "annotation": "str", "description": "An optional string."},
    ... ]
    True
    """
    # Extract parameter lines
    param_lines = _extract_params_section(docstring, doc_style)

    # Parse parameter lines
    if doc_style == "numpy":
        params = _parse_numpy_param_lines(param_lines)
    elif doc_style == "google":
        params = _parse_google_param_lines(param_lines)
    elif doc_style == "rest":
        params = _parse_rest_param_lines(param_lines)
    else:
        raise ValueError(f"Unsupported doc_style: {doc_style}")

    return params


def _extract_params_section(docstring: str, doc_style: str) -> List[str]:
    lines = docstring.strip().split("\n")
    if doc_style == "numpy":
        # Find "Parameters" section
        try:
            param_index = lines.index("Parameters")
            # Next line should be a separator like '----------'
            separator_line = lines[param_index + 1]
            if not re.match(r"-{3,}", separator_line.strip()):
                raise ValueError(
                    "Expected a separator after 'Parameters' in Numpy docstring"
                )
            param_lines = lines[param_index + 2 :]
        except ValueError:
            # "Parameters" section not found
            return []
    elif doc_style == "google":
        # Find "Args:" section
        try:
            param_index = lines.index("Args:")
            param_lines = lines[param_index + 1 :]
        except ValueError:
            # "Args:" section not found
            return []
    elif doc_style == "rest":
        param_lines = lines
    else:
        raise ValueError(f"Unsupported doc_style: {doc_style}")
    return param_lines


def _collect_indented_lines(
    param_lines: List[str], start_index: int, indent_level: int
) -> Tuple[str, int]:
    description_lines = []
    i = start_index
    while i < len(param_lines) and (
        param_lines[i].startswith(" " * indent_level) or param_lines[i].strip() == ""
    ):
        description_lines.append(param_lines[i].strip())
        i += 1
    description = " ".join(description_lines).strip()
    return description, i


def _parse_numpy_param_lines(param_lines: List[str]) -> List[Dict]:
    params = []
    i = 0
    while i < len(param_lines):
        line = param_lines[i]
        if not line.strip():
            i += 1
            continue
        # Match parameter line: name : type, default=value
        param_match = re.match(
            r"^(\w+)\s*:\s*([^,]+)(?:,\s*default=(.+))?", line.strip()
        )
        if param_match:
            name = param_match.group(1)
            annotation = param_match.group(2).strip()
            default = param_match.group(3).strip() if param_match.group(3) else None
            i += 1
            # Collect description lines
            description, i = _collect_indented_lines(param_lines, i, indent_level=4)
            param = {
                "name": name,
                "default": default,
                "annotation": annotation,
                "description": description,
            }
            params.append(param)
        else:
            i += 1
    return params


def _parse_google_param_lines(param_lines: List[str]) -> List[Dict]:
    params = []
    i = 0
    while i < len(param_lines):
        line = param_lines[i]
        if not line.strip():
            i += 1
            continue
        # Match parameter line: name (type, optional): description
        param_match = re.match(r"^ {4}(\w+)\s*(?:\(([^)]+)\))?:\s*(.+)", line)
        if param_match:
            name = param_match.group(1)
            type_default = param_match.group(2)
            description = param_match.group(3)
            annotation = None
            default = None
            if type_default:
                # Parse type and optional
                type_parts = type_default.split(",")
                annotation = type_parts[0].strip()
                if "optional" in [part.strip() for part in type_parts[1:]]:
                    # Extract default from description if present
                    default_match = re.search(r"Defaults to (.+)\.", description)
                    if default_match:
                        default = default_match.group(1).strip()
                        description = re.sub(
                            r"Defaults to .+\.", "", description
                        ).strip()
            i += 1
            # Collect additional description lines
            extra_description, i = _collect_indented_lines(
                param_lines, i, indent_level=8
            )
            if extra_description:
                description += " " + extra_description
            param = {
                "name": name,
                "default": default,
                "annotation": annotation,
                "description": description.strip(),
            }
            params.append(param)
        else:
            i += 1
    return params


def _parse_rest_param_lines(param_lines: List[str]) -> List[Dict]:
    params_dict = {}
    i = 0
    while i < len(param_lines):
        line = param_lines[i].strip()
        # Match ':param name: description'
        param_match = re.match(r"^:param (\w+):\s*(.+)", line)
        if param_match:
            name = param_match.group(1)
            description = param_match.group(2)
            # Check for default value in description
            default_match = re.search(r"\(Default:\s*(.+)\)", description)
            default = default_match.group(1).strip() if default_match else None
            if default:
                description = re.sub(r"\(Default:\s*.+\)", "", description).strip()
            # Initialize parameter entry
            params_dict[name] = {
                "name": name,
                "default": default,
                "annotation": None,
                "description": description,
            }
            i += 1
        # Match ':type name: type'
        elif line.startswith(":type"):
            type_match = re.match(r"^:type (\w+):\s*(.+)", line)
            if type_match:
                name = type_match.group(1)
                annotation = type_match.group(2).strip()
                if name in params_dict:
                    params_dict[name]["annotation"] = annotation
                else:
                    # Type without param, create entry
                    params_dict[name] = {
                        "name": name,
                        "default": None,
                        "annotation": annotation,
                        "description": "",
                    }
                i += 1
            else:
                i += 1
        else:
            i += 1
    params = list(params_dict.values())
    return params


# --------------------------------------------------------------------------------------
# Older stuff. TODO: Clean up and remove what's not needed

import doctest
from typing import Callable
import re
from itertools import groupby
from inspect import getdoc

MAX_LINE_LENGTH = 72  # https://en.wikipedia.org/wiki/Characters_per_line


def _prefix_lines(s: str, prefix: str = "# ", even_if_empty: bool = False) -> str:
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
    return "\n".join(map(lambda x: prefix + x, s.split("\n")))


class ExampleX(doctest.Example):
    """doctest.Example eXtended to have more convenient methods"""

    source_prefix = ">>> "
    source_continuation = "... "

    def __init__(
        self,
        source,
        want=None,
        exc_msg=None,
        lineno=0,
        indent=0,
        options=None,
    ):
        # if source is already a doctest.Example instance, use its properties as args
        if isinstance(source, doctest.Example):
            o = source
            source, want, exc_msg, lineno, indent, options = (
                o.source,
                o.want,
                o.exc_msg,
                o.lineno,
                o.indent,
                o.options,
            )
        super().__init__(source, want, exc_msg, lineno, indent, options)

    @property
    def indent_str(self):
        return self.indent * " "

    @property
    def _source_str(self):
        indented_continuation = f"{self.indent_str}{self.source_continuation}"
        t = self.source.replace("\n", f"\n{indented_continuation}")
        # remove the trailing '...' if present
        if t.endswith(indented_continuation):
            t = t[: -len(indented_continuation)]
        return t

    def __str__(self):
        want = self.want
        if self.want:
            want = self.indent_str + want
        return self.indent_str + self.source_prefix + self._source_str + want

    def __repr__(self):
        return str(self)


from typing import Sequence


class DoctestBlock(list):
    """A list that (should) contain doctest Example instances"""

    def __init__(self, seq=()):
        super().__init__(map(ExampleX, seq))

    def __str__(self):
        return "".join(map(str, self))

    def __repr__(self):
        return f"<DoctestBlock with {len(self)} examples>\n" + str(self)


parse_doctest = doctest.DocTestParser().parse


# TODO: Newer, using doctest parser. See what can be merged
#  with non_doctest_lines etc.
def split_text_and_doctests(doc_string: str):
    r"""
    Generates alternating blocks of "text" (string) and "doctest blocks"
    (``DoctestBlock`` instances, which are essentially a list of ``ExampleX`` instances).

    >>> example = '''
    ...     This is to test the doctest splitter.
    ...     Until now, we're in a text block.
    ...     The following is a doctest block:
    ...
    ...     >>> 2 + 3
    ...     5
    ...     >>> t = 5
    ...     >>> tt = 10
    ...
    ...     This is another text block, followed with another doctest block:
    ...
    ...     >>> def foo():
    ...     ...     return 42
    ...     >>> foo()
    ...     42
    ...
    ... '''
    >>>
    >>> blocks = list(split_text_and_doctests(example))

    There are 5 blocks:

    >>> len(blocks)
    5

    The first block is a string, corresponding to explanatory text of the doc string:

    >>> isinstance(blocks[0], str)
    True
    >>> print(blocks[0])
    <BLANKLINE>
    This is to test the doctest splitter.
    Until now, we're in a text block.
    The following is a doctest block:
    <BLANKLINE>
    <BLANKLINE>

    The next block is a ``DoctestBlock`` instance.

    >>> block = blocks[1]
    >>> isinstance(block, DoctestBlock)
    True

    This block has 3 elements (``ExampleX`` instances)

    >>> len(block)
    3

    If you ask for the string representation of this block, you'll get a doctest string:

    >>> str(block)
    '    >>> 2 + 3\n    5\n    >>> t = 5\n    >>> tt = 10\n'

    """

    def is_string(item):
        if isinstance(item, str):
            if item == "":
                return False
            else:
                return True
        else:
            return False

    t = parse_doctest(doc_string)
    g = groupby(t, key=is_string)
    for _is_string, group_data in g:
        if _is_string:
            yield "\n".join(list(group_data))
        else:
            yield DoctestBlock(filter(None, group_data))


comment_strip_p = re.compile(r"(?m)^ *#.*\n?")
doctest_line_p = re.compile("\s*>>>")
empty_line = re.compile("\s*$")


def non_doctest_lines(doc):
    r"""Generator of lines of the doc string that are not in a doctest scope.

    >>> def _test_func():
    ...     '''Line 1
    ...     Another
    ...     >>> doctest_1
    ...     >>> doctest_2
    ...     line_after_a_doc_test
    ...     another_line_that_is_in_the_doc_test scope
    ...
    ...     But now we're out of a doctest's scope
    ...
    ...     >>> Oh no, another doctest!
    ...     '''
    >>> from inspect import getdoc
    >>>
    >>> list(non_doctest_lines(getdoc(_test_func)))
    ['Line 1', 'Another', "But now we're out of a doctest's scope", '']

    :param doc:
    :return:
    """
    last_line_was_a_doc_test = False
    for line in doc.splitlines():
        if not doctest_line_p.match(line):
            if not last_line_was_a_doc_test:
                yield line
                last_line_was_a_doc_test = False
            else:
                if empty_line.match(line):
                    last_line_was_a_doc_test = False
        else:
            last_line_was_a_doc_test = True


def strip_comments(code):
    code = str(code)
    return comment_strip_p.sub("", code)


def mk_example_wants_callback(source_want_func: Callable[[str, str], Callable]):
    def example_wants_callback(example, *args, **kwargs):
        want = example.want.strip()
        if want:
            source = example.source.strip()
            return source_want_func(source, want, *args, **kwargs)
        else:
            return example.source

    return example_wants_callback


def split_line_comments(s):
    t = s.split("#")
    if len(t) == 1:
        comment = ""
    else:
        s, comment = t
    return s, comment


def _assert_wants(source, want, wrap_func_name=None):
    is_a_multiline = len(source.split("\n")) > 1

    if not is_a_multiline:
        source, comment = split_line_comments(source)
        if wrap_func_name is None:
            t = f"({source}) == {want} #{comment}"
        else:
            t = f"{wrap_func_name}({source}) == {wrap_func_name}({want}) #{comment}"
        if "'" in t and not '"' in t:
            strchr = '"'
            return "assert {t}, {strchr}{t}{strchr}".format(t=t, strchr=strchr)
        elif '"' in t and not "'" in t:
            strchr = "'"
            return "assert {t}, {strchr}{t}{strchr}".format(t=t, strchr=strchr)
        else:
            return "assert {t}".format(t=t)
    else:  # if you didn't return before
        if wrap_func_name is None:
            return f"actual = {source}\nexpected = {want}\nassert actual == expected"
        else:
            return (
                f"actual = {wrap_func_name}({source})\nexpected = {wrap_func_name}({want})\n"
                "assert actual == expected"
            )


def _output_prefix(source, want, prefix="# OUTPUT: "):
    return source + "\n" + prefix + want + "\n"


output_prefix = mk_example_wants_callback(_output_prefix)
assert_wants = mk_example_wants_callback(_assert_wants)

# def example_to_doctest_string(source, want):
#     want.replace()
#     return source +


def doctest_string_trans_lines(
    doctest_obj: doctest.DocTest, example_callback=assert_wants
):
    for example in doctest_obj.examples:
        yield example_callback(example)


def _doctest_string_gen(obj, example_callback, recurse=True):
    doctest_finder = doctest.DocTestFinder(verbose=False, recurse=recurse)
    doctest_objs = doctest_finder.find(obj)
    for doctest_obj in doctest_objs:
        yield from doctest_string_trans_lines(doctest_obj, example_callback)


def doctest_string(obj, example_callback=assert_wants, recurse=True):
    """
    Extract the doctests found in given object.

    :param obj: Object (module, class, function, etc.) you want to extract doctests from.
    :params output_prefix:
    :param recurse: Whether the process should find doctests in the attributes of the object, recursively.
    :return: A string containing the doctests, with output lines prefixed by '# Output:'
    """
    return "\n".join(_doctest_string_gen(obj, example_callback, recurse=recurse))


from functools import partial

doctest_string.for_output_prefix = partial(
    doctest_string, example_callback=output_prefix
)
doctest_string.for_assert_wants = partial(doctest_string, example_callback=assert_wants)


def doctest_string_print(obj, example_callback=assert_wants, recurse=True):
    """
    Extract the doctests found in given object.
    :param obj: Object (module, class, function, etc.) you want to extract doctests from.
    :param recurse: Whether the process should find doctests in the attributes of the object, recursively.
    :return: A string containing the doctests, with output lines prefixed by '# Output:'
    """
    return print(doctest_string(obj, example_callback, recurse=recurse))


def old_doctest_string(
    obj,
    output_prefix="# OUTPUT: ",
    include_attr_without_doctests=False,
    recurse=True,
):
    """
    Extract the doctests found in given object.
    :param obj: Object (module, class, function, etc.) you want to extract doctests from.
    :param output_prefix:
    :param recurse: Whether the process should find doctests in the attributes of the object, recursively.
    :return: A string containing the doctests, with output lines prefixed by '# Output:'
    """
    doctest_finder = doctest.DocTestFinder(verbose=False, recurse=recurse)
    r = doctest_finder.find(obj)
    s = ""
    for rr in r:
        header = f"# {rr.name} "
        header += "#" * max(0, MAX_LINE_LENGTH - len(header)) + "\n"
        ss = ""
        for example in rr.examples:
            want = example.want
            want = want.strip()
            ss += "\n" + example.source + _prefix_lines(want, prefix=output_prefix)
        if include_attr_without_doctests:
            s += header + ss
        elif len(ss) > 0:  # only append this attr if ss is non-empty
            s += header + ss
    return s


# import sphinx

if __name__ == "__main__":
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

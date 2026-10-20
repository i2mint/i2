"""Meta-interfaces"""

from typing import Literal, List, Dict, Tuple, Union
from collections.abc import Callable
from functools import partial
import re
import ast
from inspect import cleandoc

nan = float("nan")

# --------------------------------------------------------------------------------------


def find_in_params(
    query: str, params: Callable | str, *, search_in=("name", "description")
) -> list[dict]:
    """
    Find parameters in a list of parameter specifications.

    :param query: The query to search for.
    :param params: The list of parameter specifications.
        Params can be provided in two formats:

        - A function, from which the params will be extracted from the docstring.
        - A list of dictionaries where each dictionary specifies a parameter, containing:

            - name: The name of the parameter (str).
            - default: The default value of the parameter (any, optional).
            - annotation: The type annotation for the parameter (str, optional).
            - description: A description of the parameter (str).

        If a callable is provided, it will be used to generate the list of parameter specifications.
    :param search_in: The fields to search in each parameter specification.
    :return: A list of parameter specifications that match the query.

    Examples:

        >>> params = [
        ...     {"name": "x", "default": 1, "annotation": "int", "description": "An integer value."},
        ...     {"name": "y", "default": None, "annotation": "str", "description": "An optional string."},
        ... ]
        >>> find_in_params('int', params)
        [{'name': 'x', 'default': 1, 'annotation': 'int', 'description': 'An integer value.'}]

    """
    if isinstance(search_in, str):
        search_in = (search_in,)
    if isinstance(params, str) or callable(params):
        params = docstring_to_params(params)
    if isinstance(query, str):
        query = re.compile(query, re.IGNORECASE)
    return list(
        filter(lambda x: any(query.search(x.get(key, "")) for key in search_in), params)
    )


def indent_lines(string: str, indent: str) -> str:
    r"""
    Indent each line of a string.

    :param string: The string to indent.
    :param indent: The string to use for indentation.
    :return: The indented string.

    **Examples**

    >>> print(indent_lines('This is a test.\nAnother line.', ' ' * 8))
            This is a test.
            Another line.
    """
    return "\n".join(indent + line for line in string.split("\n"))


def most_common_indent(string: str, ignore_first_line=True) -> str:
    r"""
    Find the most common indentation in a string.

    :param string: The string to analyze.
    :param ignore_first_line: Whether to ignore the first line when determining the
        indentation. Default is True since the first line often has no indentation
        because of the way python strings appear in code.
    :return: The most common indentation string.

    **Examples**

    >>> most_common_indent('    This is a test.\n    Another line.')
    '    '
    """
    indents = re.findall(r"^( *)\S", string, re.MULTILINE)
    n_lines = len(indents)
    if ignore_first_line and n_lines > 1:
        # if there's more than one line, ignore the indent of the first
        # (This is is because of the way docstrings are often formatted)
        indents = indents[1:]
    return max(indents, key=indents.count)


def inject_docstring_content(to_inject, *, position=-1, indent=True):
    r"""
    Inject content into the docstring of a function.

    Note:
        If you use the decorator on a string, it will assume that string is the doc
        string you want to transform and return the modified string directly.

    :param to_inject: The content to inject.
    :param position: The position in the docstring to inject the content.
        If an integer, the content is injected at that line number (pushing the rest down).
        If a string, will consider it as a regex pattern to match the line to inject after.
        Default is -1, to inject at the end.
    :param indent: Control on indent.
        If True, will use the most common indent of the input docstrings.
        If a string, it will use that specific string.
    :return: A decorator that injects the content into the docstring of the decorated function.

    **Examples**

    >>> @inject_docstring_content('This is a test.')
    ... def test_func():
    ...     '''This is the docstring.'''
    ...     pass
    >>> test_func.__doc__
    'This is the docstring.\nThis is a test.'
    >>> @inject_docstring_content('This is a test.', position='###INSERT HERE###')
    ... def test_func():
    ...     '''This is the docstring.
    ...     ###INSERT HERE###
    ...     More blah.
    ...     '''
    ...     pass
    >>> test_func.__doc__
    'This is the docstring.\n    ###INSERT HERE###\n    More blah.\n    '
    """

    if indent is True:
        indent = most_common_indent(to_inject)
    else:
        indent = indent or ""

    def decorator(func):
        input_was_docstr_itself = False

        if isinstance(func, str):
            doc = func
            input_was_docstr_itself = True
        else:
            doc = func.__doc__ or ""

        lines = doc.split("\n")
        # TODO: Could figure out indent (ignoring first line), to inform the injected content indent

        if isinstance(position, int):
            if position == -1:
                lines.append(to_inject)
            else:
                lines.insert(position, to_inject)
        elif isinstance(position, str):
            for i, line in enumerate(lines):
                if re.match(position, line):
                    lines.insert(i + 1, to_inject)
                    break

        new_doc = "\n".join(lines)

        if input_was_docstr_itself:
            return new_doc
        else:
            func.__doc__ = new_doc
            return func

    return decorator


# TODO: params_to_docstring and docstring_to_params are a parse/generate pair, with echoes of embody and routing techniques.
def params_to_docstring(
    params: list[dict],
    *,
    doc_style: str = "numpy",
    take_name_of_types: bool = False,
    quote_string_defaults: bool = True,
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
    :param take_name_of_types: Whether to use the name of the type as the annotation (bool).
    :param quote_string_defaults: Whether to quote string defaults (bool).

    :return: A formatted docstring (str).

    **Examples**

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

    # Preprocess parameters to a uniform structure
    def processed_params():
        for p in params:
            name = p["name"]
            annotation = p.get("annotation", "")
            if (
                take_name_of_types
                and isinstance(annotation, type)
                and hasattr(annotation, "__name__")
            ):
                annotation = annotation.__name__
            description = p.get("description", "")
            if not description or isinstance(description, float):  # float to catch nans
                description = ""
            default = p.get("default", None)
            if quote_string_defaults and isinstance(default, str):
                default = f'"{default}"'
            optional = "default" in p

            yield {
                "name": name,
                "annotation": annotation,
                "description": description,
                "default": default,
                "optional": optional,
            }

    def numpy_lines(params):
        yield "Parameters"
        yield "----------"
        for param in params:
            line = param["name"]
            if param["annotation"]:
                line += f" : {param['annotation']}"
            if param["optional"]:
                line += f", default={param['default']}"
            yield line
            yield f"    {param['description']}"

    def google_lines(params):
        yield "Args:"
        for param in params:
            line = f"    {param['name']}"
            if param["annotation"]:
                line += f" ({param['annotation']}"
                if param["optional"]:
                    line += ", optional"
                line += ")"
            elif param["optional"]:
                line += " (optional)"
            desc = param["description"]
            if param["optional"]:
                desc += f" Defaults to {param['default']}."
            line += f": {desc}"
            yield line

    def rest_lines(params):
        for param in params:
            desc = param["description"]
            if param["optional"]:
                desc += f" (Default: {param['default']})"
            yield f":param {param['name']}: {desc}"
            if param["annotation"]:
                yield f":type {param['name']}: {param['annotation']}"

    formatters = {
        "numpy": numpy_lines,
        "google": google_lines,
        "rest": rest_lines,
    }

    if doc_style not in formatters:
        raise ValueError(f"Unsupported doc_style: {doc_style}")

    lines = formatters[doc_style](processed_params())
    return "\n".join(lines) + "\n"


def string_param_to_obj(string_to_object_mapping: dict, string=None):
    """
    Convert a string representation of a parameter to an object.

    Use Case: When parsing a docstring, you get values as strings, but these values
    may need to be converted to their actual object types
    (in the case of default and annotatios for example).
    This is a helper function to do that conversion.

    :param string: The string representation of the parameter.
    :param string_to_object_mapping: A mapping from string representations to objects.
    :return: The object corresponding to the string representation.

    **Examples**

    >>> string_to_object_mapping = {
    ...     'None': None,
    ...     'list': list,
    ...     'int': int,
    ... }
    >>> string_to_obj = string_param_to_obj(string_to_object_mapping)
    >>> string_to_obj('None')
    >>> string_to_obj('list')
    <class 'list'>
    >>> string_to_obj('int')
    <class 'int'>
    >>> string_to_obj('not something listed')
    'not something listed'
    """
    if string is None:
        return partial(string_param_to_obj, string_to_object_mapping)

    if string in string_to_object_mapping:
        return string_to_object_mapping[string]
    else:
        # TODO: Should handle strings that are numeric values...

        # if all else fails, return the string itself
        return string


dflt_str_to_obj_mapping = {
    "None": None,
    "True": True,
    "False": False,
    "list": list,
    "int": int,
    "float": float,
    "str": str,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "bool": bool,
    "complex": complex,
    "nan": nan,
    "inf": float("inf"),
    "-inf": float("-inf"),
}

_MAX_LENGTH_FOR_LITERAL_EVAL = 1000


def literal_eval_converter(s: str, max_length=_MAX_LENGTH_FOR_LITERAL_EVAL):
    """Evaluate ``s`` as a Python literal, or return None when it is not one (or is
    longer than ``max_length``, or contains a newline or ``;``).

    >>> literal_eval_converter("[1, 2]"), literal_eval_converter("foo")
    ([1, 2], None)
    """
    if len(s) > max_length:  # Restrict long strings for extra safety
        return None
    elif "\n" in s or "\r" in s or ";" in s:  # extra safety
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def register_converter(converter):
    """
    Register a new converter. A converter can be:

      - A dict: { "None": None, "int": int, ... }
      - A callable: lambda s: attempt to parse s and return object or None
    """
    dflt_str_to_obj_converters.append(converter)


dflt_str_to_obj_converters = []
register_converter(dflt_str_to_obj_mapping)
register_converter(literal_eval_converter)


def convert_string(s: str, converters: list[dict | Callable]) -> object:
    """Convert ``s`` with the first converter that matches (a dict containing ``s`` as a
    key, or a callable returning something other than None); return ``s`` if none does.

    >>> convert_string("None", dflt_str_to_obj_converters), convert_string("3.5", dflt_str_to_obj_converters)
    (None, 3.5)
    >>> convert_string("hello", dflt_str_to_obj_converters)
    'hello'
    """
    for converter in converters:
        # If converter is a dict
        if isinstance(converter, dict):
            if s in converter:
                return converter[s]

        # If converter is a callable
        else:
            try:
                result = converter(s)
                if result is not None:
                    return result
            except Exception:
                # If a callable fails, just skip it
                pass

    # If no converter matched
    return s


DocStyle = Literal["numpy", "google", "rest"]

# Each style's tell-tale marker: a numpy section header underlined with dashes, a reST
# field (``:param x:``), a google section header ending with a colon. The style whose
# marker comes first in the docstring wins (ties go to the earlier style here).
_doc_style_markers = (
    (
        "numpy",
        re.compile(
            r"^\s*(Parameters|Other Parameters|Returns|Yields|Raises|Attributes)"
            r"\s*\n\s*-{3,}\s*$",
            re.MULTILINE,
        ),
    ),
    (
        "rest",
        re.compile(
            r"^\s*:(param|parameter|arg|argument|key|keyword|type|returns?|rtype|raises?)\b",
            re.MULTILINE,
        ),
    ),
    (
        "google",
        re.compile(
            r"^\s*(Args|Arguments|Parameters|Keyword Args|Keyword Arguments|Returns"
            r"|Yields|Raises|Attributes):\s*$",
            re.MULTILINE,
        ),
    ),
)


def detect_doc_style(docstring: str | Callable) -> DocStyle | None:
    """Detect which parameter-description style a docstring uses, or ``None``.

    Returns ``'numpy'``, ``'rest'`` or ``'google'`` (the styles ``docstring_to_params``
    and ``params_to_docstring`` handle), based on each style's section markers. A
    callable's ``__doc__`` is used.

    >>> detect_doc_style('''
    ... Parameters
    ... ----------
    ... x : int
    ...     An integer.
    ... ''')
    'numpy'
    >>> detect_doc_style('''
    ... Args:
    ...     x (int): An integer.
    ... ''')
    'google'
    >>> detect_doc_style('''
    ... :param x: An integer.
    ... :type x: int
    ... ''')
    'rest'

    When several styles' markers appear (e.g. a google docstring whose example
    shows reST fields), the one that comes first wins:

    >>> detect_doc_style('''
    ... Args:
    ...     x (int): An integer.
    ...
    ... Example:
    ...     :param y: shown as text, not a field.
    ... ''')
    'google'
    >>> detect_doc_style('Just a summary line.') is None
    True
    >>> detect_doc_style(None) is None
    True

    It reads the markers of the docstring as written, so it works on the indented
    ``__doc__`` of a function too:

    >>> def f(x):
    ...     '''Do it.
    ...
    ...     :param x: An integer.
    ...     '''
    >>> detect_doc_style(f)
    'rest'
    """
    if not isinstance(docstring, str) and callable(docstring):
        docstring = docstring.__doc__ or ""
    if not isinstance(docstring, str):
        return None
    docstring = _normalize_newlines(docstring)
    first_match_positions = {}
    for style, marker in _doc_style_markers:
        if match := marker.search(docstring):
            first_match_positions[style] = match.start()
    if not first_match_positions:
        return None
    return min(first_match_positions, key=first_match_positions.get)


def _normalize_newlines(text: str) -> str:
    r"""Turn CRLF (and lone CR) line endings into LF.

    >>> _normalize_newlines('a\r\nb\rc')
    'a\nb\nc'
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def docstring_to_params(
    docstring: str,
    *,
    doc_style: Literal["numpy", "google", "rest", "auto"] = "numpy",
    converters: list = dflt_str_to_obj_converters,
) -> list[dict]:
    """
    Parse a docstring and extract parameter specifications.

    :param docstring: The docstring to parse.
    :param doc_style: The style of the docstring to parse. One of 'numpy', 'google',
        or 'rest', or 'auto' to use the style ``detect_doc_style`` finds (falling back
        to 'numpy').

    :return: A list of parameter specifications, where each specification is a dictionary containing:

        - name: The name of the parameter (str).
        - default: The default value of the parameter (str, optional).
        - annotation: The type annotation for the parameter (str, optional).
        - description: A description of the parameter (str).

    **Examples**

    >>> docstring = '''
    ... Parameters
    ... ----------
    ... x : int, default=1
    ...     An integer value.
    ... y : str, default=None
    ...     An optional string.
    ... '''
    >>> params = docstring_to_params(docstring)
    >>> params  == [
    ...     {'name': 'x', 'default': 1, 'annotation': int, 'description': 'An integer value.'},
    ...     {'name': 'y', 'default': None, 'annotation': str, 'description': 'An optional string.'}
    ... ]
    True


    >>> docstring = '''
    ... Args:
    ...     x (int, optional): An integer value. Defaults to 1.
    ...     y (str, optional): An optional string. Defaults to None.
    ... '''
    >>> params = docstring_to_params(docstring, doc_style='google')
    >>> params == [
    ...     {"name": "x", "default": 1, "annotation": int, "description": "An integer value."},
    ...     {"name": "y", "default": None, "annotation": str, "description": "An optional string."},
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
    ...     {"name": "x", "default": 1, "annotation": int, "description": "An integer value."},
    ...     {"name": "y", "default": None, "annotation": str, "description": "An optional string."},
    ... ]
    True

    With ``doc_style='auto'``, the style is detected (see ``detect_doc_style``).
    A function's (indented) ``__doc__`` works too:

    >>> docstring_to_params(docstring, doc_style='auto') == params
    True
    >>> def f(x=1):
    ...     '''Do it.
    ...
    ...     Args:
    ...         x (int, optional): An integer value. Defaults to 1.
    ...     '''
    >>> [p['name'] for p in docstring_to_params(f, doc_style='auto')]
    ['x']

    The Sphinx typed-field form (``:param int x:``) and CRLF line endings are handled:

    >>> docstring_to_params(':param int x: X.\\r\\n:param y: Note: Y.', doc_style='auto') == [
    ...     {"name": "x", "default": None, "annotation": int, "description": "X."},
    ...     {"name": "y", "default": None, "annotation": None, "description": "Note: Y."},
    ... ]
    True
    """
    if not isinstance(docstring, str) and callable(docstring):
        docstring = docstring.__doc__ or ""
    # dedent, so the section headers of an indented ``__doc__`` are found
    docstring = cleandoc(_normalize_newlines(docstring or ""))
    if doc_style == "auto":
        doc_style = detect_doc_style(docstring) or "numpy"

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

    def convert_param(param):
        param["default"] = convert_string(param["default"], converters)
        param["annotation"] = convert_string(param["annotation"], converters)
        return param

    params = list(map(convert_param, params))

    return params


_google_args_headers = {"Args:", "Arguments:", "Parameters:"}


def _extract_params_section(docstring: str, doc_style: str) -> list[str]:
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
        # Find the "Args:" (or "Arguments:", "Parameters:") section
        param_index = next(
            (i for i, line in enumerate(lines) if line.strip() in _google_args_headers),
            None,
        )
        if param_index is None:
            return []
        param_lines = lines[param_index + 1 :]
    elif doc_style == "rest":
        param_lines = lines
    else:
        raise ValueError(f"Unsupported doc_style: {doc_style}")
    return param_lines


def _collect_indented_lines(
    param_lines: list[str], start_index: int, indent_level: int
) -> tuple[str, int]:
    description_lines = []
    i = start_index
    while i < len(param_lines) and (
        param_lines[i].startswith(" " * indent_level) or param_lines[i].strip() == ""
    ):
        description_lines.append(param_lines[i].strip())
        i += 1
    description = " ".join(description_lines).strip()
    return description, i


def _parse_numpy_param_lines(param_lines: list[str]) -> list[dict]:
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


def _parse_google_param_lines(param_lines: list[str]) -> list[dict]:
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


def _parse_rest_param_lines(param_lines: list[str]) -> list[dict]:
    params_dict = {}
    i = 0
    while i < len(param_lines):
        line = param_lines[i].strip()
        # Match ':param name: description' or ':param type name: description'
        param_match = re.match(r"^:param\s+(?:([^:]+?)\s+)?(\w+):\s*(.*)", line)
        if param_match:
            annotation, name, description = param_match.groups()
            # Check for default value in description
            default_match = re.search(r"\(Default:\s*(.+)\)", description)
            default = default_match.group(1).strip() if default_match else None
            if default:
                description = re.sub(r"\(Default:\s*.+\)", "", description).strip()
            # Initialize parameter entry
            params_dict[name] = {
                "name": name,
                "default": default,
                "annotation": annotation,
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


# The doctest tools moved to i2.doctests; re-exported so existing imports keep working,
# along with the modules and helpers this module used to import for them.
import doctest  # noqa: E402, F401
from collections.abc import Sequence  # noqa: E402, F401
from inspect import getdoc  # noqa: E402, F401
from itertools import groupby  # noqa: E402, F401

from i2.doctests import (  # noqa: E402, F401
    MAX_LINE_LENGTH,
    _prefix_lines,
    ExampleX,
    DoctestBlock,
    parse_doctest,
    split_text_and_doctests,
    comment_strip_p,
    doctest_line_p,
    empty_line,
    non_doctest_lines,
    strip_comments,
    mk_example_wants_callback,
    split_line_comments,
    _assert_wants,
    _output_prefix,
    output_prefix,
    assert_wants,
    doctest_string_trans_lines,
    _doctest_string_gen,
    doctest_string,
    doctest_string_print,
    old_doctest_string,
)

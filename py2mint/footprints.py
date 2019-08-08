import ast
import inspect


def func(obj):
    return obj.a + obj.b


class A:
    e = 2

    def __init__(self, a=1, b=0, c=1, d=10):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def target_method(self, x):
        t = func(self)
        tt = self.other_method(t)
        return x * tt / self.e

    def other_method(self, x=1):
        return self.c * x


def attrs_used_by_method(cls, method_name):
    """
    Extracts a list of cls attributes which are used by a method or method_name function
    Args:
        cls: The class
        method_name: The method name

    Returns:
        A list of attribute names (of the class or instance thereof) that are accessed in the code of the said method.

    >>> def func(obj):
    ...     return obj.a + obj.b
    >>>
    >>> class A:
    ...     e = 2
    ...     def __init__(self, a, b=0, c=1, d=10):
    ...         self.a = a
    ...         self.b = b
    ...         self.c = c
    ...         self.d = d
    ...     def target_method(self, x):
    ...         t = func(self)
    ...         tt = self.other_method(t)
    ...         return x * tt / self.e
    ...     def other_method(self, x=1):
    ...         return self.c * x
    ...
    >>> attrs_used_by_method(A, 'target_method')
    ['a', 'b', 'c', 'e']
    """
    cls_source = inspect.getsource(cls)
    method_source = inspect.getsource(method_name)
    source = method_source + "\n" + cls_source
    root = ast.parse(source)
    functions = sorted({node.name for node in ast.walk(root) if isinstance(node, ast.FunctionDef)})
    names = sorted({node.attr for node in ast.walk(root) if isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load) and node.attr not in functions})
    return names


class Tracker:
    """
    Tracks the attribute access right after `start_track` is set to True.

    Add this to __metaclass__ for any class that you need to track attributes for given a
    target method.
    """
    attrs_to_ignore = []
    attr_used = []
    start_track = False

    def __getattribute__(self, item):
        """
        Inspect getter for tracking the attributes accessed.
        """
        if item not in ['on_access']:
            self.on_access(item)
        return super().__getattribute__(item)

    def __setattr__(self, key, value):
        """
        Inspect setter for tracking the attributes accessed.
        """
        if self.start_track:
            self.on_access(key)

        super().__setattr__(key, value)

    def on_access(self, key):
        """
        on attribute access, record attribute if and only if its not from
        core attribute or `attrs_to_ignore` set to class.
        """
        if key in ['start_track', 'attr_used', 'on_access', 'attrs_to_ignore'] or key in self.attrs_to_ignore:
            return
        if self.start_track:
            self.attr_used.append(key)


def attrs_used_by_method_computation(cls, method_name, method_params=None):
    """
    Tracks the access to attributes within an execution.
    Args:
        cls: class where method is defined
        method_name: method name
        method_params: kwargs dict to give to the method when run for analysis

    Returns:
         A list of attribute names (of the class or instance thereof) that are accessed in the code of the said method.
    """

    # create meta class with the above tracker and `cls` from parameters.
    if method_params is None:
        method_params = {}
    tracker = type('Tracker', (Tracker, cls), dict(cls.__dict__, **Tracker.__dict__))()
    tracker.attrs_to_ignore = [func for func in dir(tracker) if callable(getattr(tracker, func))]
    if hasattr(tracker, method_name):
        candidate_method = getattr(tracker, method_name)
        # Now, we want to track attributes.
        tracker.start_track = True
        candidate_method(**method_params)
        attr_used = tracker.attr_used.copy()
        tracker.attr_used = []
        return attr_used
    else:
        # Error class/obj do not have that method.
        return 1

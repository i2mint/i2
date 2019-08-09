import ast
import inspect
from contextlib import contextmanager


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
    start_track = False

    def __init__(self, *args, **kwargs):
        self.attr_used = []
        self.attrs_to_ignore = []
        super().__init__(*args, **kwargs)

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


def get_class_that_defined_method(method):
    """
    Get class for unbound/bound method.
    """
    if inspect.ismethod(method):
        for cls in inspect.getmro(method.__self__.__class__):
            if cls.__dict__.get(method.__name__) is method:
                return cls
        method = method.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(method):
        cls = getattr(inspect.getmodule(method),
                      method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return getattr(method, '__objclass__', None)


@contextmanager
def start_tracking(tracker_instance):
    """
    Ctx manager to gracefully start/stop tracking.
    """
    tracker_instance.start_track = True
    yield tracker_instance
    tracker_instance.start_track = False


def attrs_used_by_method_computation(cls_method, init_kwargs=None, method_kwargs=None):
    """
    Tracks the access to attributes within an execution.
    """
    if init_kwargs is None:
        init_kwargs = {}
    if method_kwargs is None:
        method_kwargs = {}

    method_class = get_class_that_defined_method(cls_method)
    method_name = cls_method.__name__
    tracker = type(
        'Tracker',
        (Tracker, method_class),
        dict(method_class.__dict__, **Tracker.__dict__)
    )(**init_kwargs)
    tracker.attrs_to_ignore = [func for func in dir(tracker) if callable(getattr(tracker, func))]
    if hasattr(tracker, method_name):
        candidate_method = getattr(tracker, method_name)
        # Now, we want to track attributes.
        with start_tracking(tracker):
            candidate_method(**method_kwargs)

        return tracker.attr_used
    else:
        # Error class/obj do not have that method.
        return 1

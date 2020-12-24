class DataError(Exception):
    pass


class DuplicateRecordError(DataError):
    pass


class NotFoundError(DataError):
    pass


class AuthorizationError(Exception):
    pass


class ForbiddenError(AuthorizationError):
    pass


class InputError(Exception):
    pass


class ModuleNotFoundIgnore:
    """Context manager meant to ignore import errors.
    The use case in mind is when we want to condition some code on the existence of some package.
    """

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True

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

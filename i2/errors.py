class DataError(Exception):
    pass


class DuplicateRecordError(Exception):
    pass


class NotFoundError(DataError):
    pass


class AuthorizationError(Exception):
    pass


class ForbiddenError(AuthorizationError):
    pass


class BusinessRuleError(Exception):
    pass


class InputError(Exception):
    pass

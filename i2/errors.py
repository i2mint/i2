class DataError(Exception):
	pass

class DuplicateRecordError(Exception):
	pass

class NotFoundError(DataError):
	pass

class ModelNotFoundError(Exception):
    pass


class ModelAlreadyUploadedError(Exception):
    pass


class ModelBadRequestError(Exception):
    pass


class ModelProcessingError(Exception):
    pass


class ModelStateError(Exception):
    pass


class InternalError(Exception):
    pass


class UnauthorizedError(Exception):
    pass


class ForbiddenError(Exception):
    pass


class ModelExceptionFactory:

    @classmethod
    def get_exception_by_code(cls, code: int, message: str):
        if code == 404:
            return ModelNotFoundError(message)
        elif code == 409:
            return ModelAlreadyUploadedError(message)
        elif code == 400:
            return ModelBadRequestError(message)
        elif code == 102:
            return ModelProcessingError(message)
        elif code == 412:
            return ModelStateError(message)
        elif code == 401:
            return UnauthorizedError(message)
        elif code == 403:
            return ForbiddenError(message)
        else:
            return InternalError(message)

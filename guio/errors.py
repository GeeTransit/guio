from curio.errors import CurioError


__all__ = ["GuioError", "EventResourceBusy"]


# Base exception class for errors in Guio
class GuioError(CurioError):
    pass


# Raised when multiple tasks try to wait on a single event queue
class EventResourceBusy(GuioError):
    pass

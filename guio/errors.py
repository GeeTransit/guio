from curio.errors import *


__all__ = ["GuioError", "NoEvent", "TaskNotEvent", "CloseWindow"]


# Base exception class for errors in this library
class GuioError(CurioError):
    pass


# Raised when there is no event to return
class NoEvent(GuioError):
    pass


# Raised when non-event task tries to use event traps
class TaskNotEvent(GuioError):
    pass


# Raised when the `X` button was pressed
# Note: This is only raised on the `_wait_event` trap.
class CloseWindow(GuioError):
    pass

from functools import wraps

from curio.task import current_task, Task as CurioTask

from .errors import *
from .traps import *


__all__ = [
    "Task", "pop_event", "aevents", "current_toplevel", "iseventtask",
    "seteventtask", "set_current_event", "unset_current_event",
]


class Task(CurioTask):

    @wraps(CurioTask.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_event = -1

    @property
    def is_event(self):
        return self._next_event >= 0

    @is_event.setter
    def is_event(self, value):
        if not isinstance(value, bool):
            raise TypeError(f"value must be bool, not {type(value).__name__}")
        if value != self.is_event:
            self._next_event *= -1
            self._next_event -= 1


async def pop_event(*, blocking=True):
    if not blocking:
        return await _pop_event()
    while True:
        try:
            return await _pop_event()
        except NoEvent:
            await _wait_event()


class aevents:

    def __init__(self, *, blocking=True):
        self.blocking = blocking

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.blocking:
            return await pop_event()
        try:
            return await pop_event(blocking=False)
        except NoEvent:
            raise StopAsyncIteration from None


async def current_toplevel():
    return await _get_toplevel()


def iseventtask(task):
    if isinstance(task, Task):
        return task.is_event
    elif hasattr(task, "_next_event"):
        return task._next_event >= 0
    else:
        return False


async def seteventtask(is_event):
    task = await current_task()
    if iseventtask(task) != is_event:
        if isinstance(task, Task):
            task.is_event = is_event
        elif hasattr(task, "_next_event"):
            task._next_event *= -1
            task._next_event -= 1
        else:
            task._next_event = is_event - 1  # (True == 1) and (False == 0)

set_current_event = lambda: seteventtask(True)
unset_current_event = lambda: seteventtask(False)

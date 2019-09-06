from curio.task import current_task

from .errors import *
from .traps import *


__all__ = [
    "pop_event", "aevents", "current_toplevel", "iseventtask",
    "set_current_event", "unset_current_event",
]


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
    return (getattr(task, "next_event", -1) >= 0)


async def seteventtask(isevent):
    task = await current_task()
    if iseventtask(task) != isevent:
        task.next_event = isevent - 1

set_current_event = lambda: seteventtask(True)
unset_current_event = lambda: seteventtask(False)

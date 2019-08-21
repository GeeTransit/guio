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

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await pop_event()


async def current_toplevel():
    return await _get_toplevel()


def iseventtask(task):
    return (getattr(task, "next_event", -1) >= 0)


async def set_current_event():
    task = await current_task()
    task.next_event = 0


async def unset_current_event():
    task = await current_task()
    task.next_event = -1

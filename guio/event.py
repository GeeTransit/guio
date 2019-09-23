from collections import deque
from contextlib import asynccontextmanager

from curio.task import spawn

from .task import current_toplevel
from .traps import *


__all__ = [
    "EVENT_ALL", "EVENT_KEY", "EVENT_BUTTON", "EVENT_MOTION",
    "PROT_DELETE", "Events",
]


EVENT_ALL = frozenset({
    '<Activate>',
    '<ButtonPress>',
    '<ButtonRelease>',
    '<Circulate>',
    '<Colormap>',
    '<Configure>',
    '<Deactivate>',
    '<Enter>',
    '<Expose>',
    '<FocusIn>',
    '<FocusOut>',
    '<Gravity>',
    '<KeyPress>',
    '<KeyRelease>',
    '<Leave>',
    '<Map>',
    '<Motion>',
    '<MouseWheel>',
    '<Property>',
    '<Reparent>',
    '<Unmap>',
    '<Visibility>',
    "WM_DELETE_WINDOW",
})

EVENT_KEY = frozenset({'<KeyPress>', '<KeyRelease>'})
EVENT_BUTTON = frozenset({'<ButtonPress>', '<ButtonRelease>'})
EVENT_MOTION = frozenset({'<Enter>', '<Motion>', '<Leave>'})
PROT_DELETE = frozenset({"WM_DELETE_WINDOW"})


class Events:

    def __init__(self, waiters, *, blocking=True):
        self.blocking = blocking
        self._waiters = waiters
        self._events = deque()
        self._waiting = None

    def __repr__(self):
        return f"<{type(self).__name__} waiting={self._waiting!r}>"

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.pop()
        except IndexError:
            raise StopAsyncIteration from None

    async def wait(self):
        await _event_wait(self)

    async def pop(self, *, blocking=None):
        if blocking is None:
            blocking = self.blocking
        if not blocking:
            return self._events.popleft()
        while True:
            try:
                return self._events.popleft()
            except IndexError:
                await self.wait()

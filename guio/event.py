from collections import deque

from curio.sched import SchedBarrier

from .errors import *
from .task import current_toplevel
from .traps import *


__all__ = ["ProtocolEvent", "Events"]


class ProtocolEvent:

    def __init__(self, type_, time, widget):
        self.type = type_
        self.time = time
        self.widget = widget


class Events:

    _names = (
        '<Activate>',
        '<Button>',
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
        '<Key>',
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
    )

    def __init__(self, waiters=(), *, blocking=True):
        self.blocking = blocking
        self._waiters = waiters
        self._events = deque()
        self._waiting = None

    @classmethod
    def from_events(cls, widget, names, **kwargs):
        waiters = ((widget, name) for name in names)
        return cls(tuple(waiters), **kwargs)

    @classmethod
    def from_pairs(cls, *pairs, **kwargs):
        waiters = ((widget, name) for widget, names in pairs for name in names)
        return cls(tuple(waiters), **kwargs)

    @classmethod
    def from_dict(cls, waiters, **kwargs):
        waiters = (
            (widget, name)
            for widget, names in waiters.items()
            for name in names
        )
        return cls(tuple(waiters), **kwargs)

    def __repr__(self):
        return f"<{type(self).__name__} waiting={self._waiting!r}>"

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.pop(blocking=self.blocking)
        except IndexError:
            raise StopAsyncIteration from None

    async def pop(self, *, blocking=True):
        if not blocking:
            return self._events.popleft()
        while True:
            try:
                return self._events.popleft()
            except IndexError:
                await self.wait()

    async def wait(self):
        await _event_wait(self)

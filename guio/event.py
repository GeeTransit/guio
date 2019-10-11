import threading
import tkinter
import weakref

from collections import Counter, defaultdict
from time import monotonic

from curio.queue import UniversalQueue
from curio.sync import UniversalEvent


__all__ = [
    "EVENT_ALL", "EVENT_KEY", "EVENT_BUTTON", "EVENT_MOTION",
    "PROT_DELETE", "EventQueue", "EventWaiter",
]


EVENT_ALL = frozenset({
    "<Activate>",
    "<ButtonPress>",
    "<ButtonRelease>",
    "<Circulate>",
    "<Colormap>",
    "<Configure>",
    "<Deactivate>",
    "<Enter>",
    "<Expose>",
    "<FocusIn>",
    "<FocusOut>",
    "<Gravity>",
    "<KeyPress>",
    "<KeyRelease>",
    "<Leave>",
    "<Map>",
    "<Motion>",
    "<MouseWheel>",
    "<Property>",
    "<Reparent>",
    "<Unmap>",
    "<Visibility>",
    "WM_DELETE_WINDOW",
})

EVENT_KEY = frozenset({"<KeyPress>", "<KeyRelease>"})
EVENT_BUTTON = frozenset({"<ButtonPress>", "<ButtonRelease>"})
EVENT_MOTION = frozenset({"<Enter>", "<Motion>", "<Leave>"})
PROT_DELETE = frozenset({"WM_DELETE_WINDOW"})


class _EventHandler:
    _event_queues = defaultdict(set)
    _watching = Counter()
    _handler_ids = {}
    _event_waiters = defaultdict(list)
    _lock = threading.Lock()

    def __init__(self):
        raise RuntimeError("Do not instantiate _EventHandler")

    # Create a new tkinter.Event instance but for a protocol event
    @staticmethod
    def _create_protocol_event(widget, name, tm):
        event = tkinter.Event()

        # Informational attributes
        event.time = int(tm*1000)
        event.type = name
        event.widget = widget

        # Default attributes
        event.char = "??"
        event.delta = 0
        event.height = "??"
        event.keycode = "??"
        event.keysym = "??"
        event.keysym_num = "??"
        event.num = "??"
        event.serial = "??"
        event.state = "??"
        event.width = "??"
        event.x = "??"
        event.y = "??"
        event.x_root = "??"
        event.y_root = "??"

        return event

    # Factory function for event callbacks
    @classmethod
    def _create_callback(cls, widget, name):
        key = (widget, name)

        # Callback function
        def _event_callback(event=None):
            if event is None:
                event = cls._create_protocol_event(widget, name, monotonic())
            elif str(event.type) is "VirtualEvent":
                event.type = name[1:-1]
            for q in list(cls._event_queues[key]):
                q.put(event)
                del q
            for eventref in list(cls._event_waiters[key]):
                event = eventref()
                if event is not None:
                    event.set()
                del event

        return _event_callback

    @classmethod
    def watch(cls, events, queue_or_event):
        """
        Attach a queue or event using a dict of widgets and events.
        """
        with cls._lock:
            for widget, names in events.items():
                for name in names:
                    is_event = name.startswith("<")
                    if not is_event:
                        while widget.master and not hasattr(widget, "protocol"):
                            widget = widget.master

                    key = (widget, name)
                    if cls._watching[key] == 0:
                        callback = cls._create_callback(widget, name)
                        if is_event:
                            cls._handler_ids[key] = widget.bind(name, callback, "+")
                        else:
                            widget.protocol(name, callback)
                            cls._handler_ids[key] = None
                    cls._watching[key] += 1

                    if isinstance(queue_or_event, UniversalQueue):
                        cls._event_queues[key].add(queue_or_event)
                    elif isinstance(queue_or_event, UniversalEvent):
                        cls._event_waiters[key].append(weakref.ref(queue_or_event))

    @classmethod
    def unwatch(cls, events, queue_or_event):
        """
        Detach a queue or event using a dict of widgets and events.
        """
        with cls._lock:
            for widget, names in events.items():
                for name in names:
                    is_event = name.startswith("<")
                    if not is_event:
                        while widget.master and not hasattr(widget, "protocol"):
                            widget = widget.master

                    key = (widget, name)
                    cls._watching[key] -= 1
                    if cls._watching[key] == 0:
                        if is_event:
                            widget.unbind(name, cls._handler_ids[key])
                        elif name == "WM_DELETE_WINDOW":
                            widget.protocol(name, widget.destroy)
                        else:
                            widget.protocol(name, lambda: None)

                    if isinstance(queue_or_event, UniversalQueue):
                        cls._event_queues[key].discard(queue_or_event)
                    elif isinstance(queue_or_event, UniversalEvent):
                        cls._event_waiters[key] = [
                            eventref
                            for eventref in cls._event_waiters[key]
                            if eventref() is not None
                        ]


class EventQueue(UniversalQueue):
    """
    A queue for watching a given dictionary of widgets and events. This
    is a subclass of UniversalQueue and is safe to use in Curio, Guio or
    threads.
    """

    def __init__(self, events, maxsize=0, **kwargs):
        assert maxsize == 0, "EventQueues must be unbounded"
        super().__init__(**kwargs)
        self._events = events
        self._watching = False

    def _get_noblock(self, default=None):
        """
        Convenience function for non-blocking gets. Returns an item from
        the queue if available, default otherwise.
        """
        with self._mutex:
            if not self._queue or self._getters:
                return default
            else:
                item = self._get_item()
                self._get_complete()
                return item

    def __enter__(self):
        assert not self._watching
        _EventHandler.watch(self._events, self)
        self._watching = True
        return self

    def __exit__(self, *args):
        _EventHandler.unwatch(self._events, self)
        self._watching = False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        return self.__exit__(*args)


class EventWaiter(UniversalEvent):

    def __init__(self, events):
        super().__init__()
        self._events = events
        _EventHandler.watch(events, self)

    def __del__(self):
        _EventHandler.unwatch(self._events, self)

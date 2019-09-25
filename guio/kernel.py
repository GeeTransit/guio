import errno
import logging
import os
import tkinter

from collections import deque
from contextlib import contextmanager, ExitStack
from functools import wraps
from selectors import EVENT_READ, EVENT_WRITE
from socket import socketpair
from time import monotonic

from curio.errors import *
from curio.kernel import run as _run, Kernel as _Kernel
from curio.task import Task
from curio.timequeue import TimeQueue
from curio.traps import _read_wait

from .errors import *
from .utilities import *


__all__ = ["Kernel", "run"]


logger = logging.getLogger(__name__)


class Kernel(_Kernel):

    def __init__(self, *, selector=None, debug=None, activations=None, toplevel=None):
        super().__init__(selector=selector, debug=debug, activations=activations)

        if toplevel is None:
            toplevel = tkinter.Tk()
        self._toplevel = toplevel
        self._call_at_shutdown(lambda: destroy(self._toplevel))

        def _close_window():
            self._shutdown_funcs = None
            toplevel.destroy()
        toplevel.protocol("WM_DELETE_WINDOW", _close_window)


    def _make_kernel_runtime(kernel):

        # --- Kernel state ---

        # Current task / state
        current = None
        running = False

        # Inner loop and dummy frame (tkinter loop)
        loop = None
        frame = None

        # Restore kernel attributes
        selector = kernel._selector
        tasks = kernel._tasks
        toplevel = kernel._toplevel

        # Internal kernel state
        event_waiters = {}
        ready = deque()
        sleepq = TimeQueue()
        wake_queue = deque()
        _activations = []


        # --- Bound methods ---

        selector_register = selector.register
        selector_unregister = selector.unregister
        selector_modify = selector.modify
        selector_select = selector.select
        selector_getkey = selector.get_key
        selector_getmap = selector.get_map

        ready_append = ready.append
        ready_popleft = ready.popleft


        # --- Future processing ---

        # Loopback sockets
        notify_sock = None
        wait_sock = None

        async def _kernel_task():
            wake_queue_popleft = wake_queue.popleft

            while True:
                await _read_wait(wait_sock)

                try:
                    wait_sock.recv(1000)
                except BlockingIOError:
                    # This may raise an error as the 1 ms delay in
                    # tkinter's loop could cause the read to fail.
                    pass

                while wake_queue:
                    task, future = wake_queue_popleft()
                    if future and task.future is not future:
                        continue
                    task.future = None
                    reschedule_task(task)

        def wake(task=None, future=None):
            if task:
                wake_queue.append((task, future))
            notify_sock.send(b"\x00")

        def init_loopback():
            nonlocal notify_sock, wait_sock
            notify_sock, wait_sock = socketpair()
            wait_sock.setblocking(False)
            notify_sock.setblocking(False)
            kernel._call_at_shutdown(notify_sock.close)
            kernel._call_at_shutdown(wait_sock.close)


        # --- Task helpers ---

        def reschedule_task(task, val=None):
            ready_append(task)
            task.state = "READY"
            task.cancel_func = None
            task._trap_result = val

        def suspend_task(state, cancel_func):
            nonlocal running
            current.state = state
            current.cancel_func = cancel_func
            if current._last_io:
                unregister_event(*current._last_io)
                current._last_io = None
            running = None

        def new_task(coro):
            task = Task(coro)
            tasks[task.id] = task
            reschedule_task(task)
            for a in _activations:
                a.created(task)
            return task

        def cancel_task(task, exc):
            if task.allow_cancel and task.cancel_func:
                task.cancel_func()
                reschedule_task(task, exc)
            else:
                task.cancel_pending = exc

        def set_timeout(tm, ty):
            if tm is None:
                sleepq.cancel((current.id, ty), getattr(current, ty))
            else:
                sleepq.push((current.id, ty), tm)
            setattr(current, ty, tm)


        # --- I/O helpers ---

        def register_event(fileobj, event, task):
            try:
                key = selector_getkey(fileobj)
            except KeyError:
                if event == EVENT_READ:
                    data = (task, None)
                else:
                    data = (None, task)
                selector_register(fileobj, event, data)
            else:
                mask = key.events
                rtask, wtask = key.data
                if event == EVENT_READ and rtask:
                    raise ReadResourceBusy(f"Multiple tasks can't wait to read on the same file descriptor {fileobj!r}")
                if event == EVENT_WRITE and wtask:
                    raise WriteResourceBusy(f"Multiple tasks can't wait to write on the same file descriptor {fileobj!r}")
                if event == EVENT_READ:
                    data = (task, wtask)
                else:
                    data = (rtask, task)
                selector_modify(fileobj, mask | event, data)

        def unregister_event(fileobj, event):
            key = selector_getkey(fileobj)
            mask = key.events
            rtask, wtask = key.data
            mask &= ~event
            if not mask:
                selector_unregister(fileobj)
            else:
                if event == EVENT_READ:
                    data = (None, wtask)
                else:
                    data = (rtask, None)
                selector_modify(fileobj, mask, data)


        # --- Event helpers ---

        # Create a new tkinter.Event instance but for a protocol event
        def new_protocol_event(widget, name, tm):
            event = tkinter.Event()
            event.time = int(tm*1000)
            event.type = name
            event.widget = widget
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

        # Add to event waiters and bind if new
        def add_waiters(events):
            if events._waiting:
                raise EventResourceBusy(f"Multiple tasks can't wait on the same event queue {events}")

            for widget, names in events._waiters.items():
                for name in names:
                    is_event = name.startswith("<")
                    if not is_event:
                        while widget.master and not hasattr(widget, "protocol"):
                            widget = widget.master

                    key = (widget, name)
                    if key in event_waiters:
                        events_set = event_waiters[key][1]
                        events_set.add(events)
                        continue

                    events_set = {events}
                    callback = send_event(widget, name, events_set)
                    if is_event:
                        bindid = widget.bind(name, callback, "+")
                    else:
                        widget.protocol(name, callback)
                        bindid = None
                    event_waiters[key] = (bindid, events_set)

        # Remove from events waiters and unbind if empty
        def remove_waiters(events):
            for widget, names in events._waiters.items():
                for name in names:
                    is_event = name.startswith("<")
                    if not is_event:
                        while widget.master and not hasattr(widget, "protocol"):
                            widget = widget.master

                    key = (widget, name)
                    assert key in event_waiters
                    bindid, events_set = event_waiters[key]
                    events_set.remove(events)

                    if events_set:
                        continue

                    if widget is toplevel and name == "WM_DELETE_WINDOW":
                        continue

                    if exists(widget):
                        if is_event:
                            widget.unbind(name, bindid)
                        else:
                            widget.protocol(name, lambda: None)
                    del event_waiters[key]


        # --- Tkinter callback decorator ---

        # Decorator to hide exceptions for tkinter callbacks
        def callback(func):
            @wraps(func)
            def _wrapper(*args):
                try:
                    func(*args)
                except BaseException as e:
                    loop.throw(e)
                else:
                    return # "break"
            return _wrapper


        # --- Tkinter callbacks ---

        # Factory for event callbacks
        def send_event(widget, name, events_set):
            @callback
            def _event_callback(event=None):
                if event is None:
                    event = new_protocol_event(widget, name, monotonic())
                for events in events_set:
                    events._events.append(event)

                if not loop.running:
                    for events in list(events_set):
                        if events._waiting:
                            reschedule_task(events._waiting)
                            events._waiting = None
                        remove_waiters(events)

                loop.send("EVENT_WAKE")

            return _event_callback

        # Factory for WM_DELETE_WINDOW protocol callbacks
        def close_window(widget, name, events_set):
            _last_close = None

            @callback
            def _event_callback():
                nonlocal _last_close
                if loop.closed:
                    kernel._shutdown_funcs = None
                    destroy(toplevel)
                    return

                if not events_set:
                    raise RuntimeError("Toplevel was closed")

                now = monotonic()
                if _last_close and _last_close + 0.5 > now:
                    raise RuntimeError("Kernel was force closed")
                else:
                    _last_close = now

                event = new_protocol_event(widget, name, monotonic())
                for events in events_set:
                    events._events.append(event)

                if not loop.running:
                    for events in list(events_set):
                        if events._waiting:
                            reschedule_task(events._waiting)
                            events._waiting = None
                        remove_waiters(events)

                loop.send("EVENT_WAKE")

            return _event_callback


        # --- Trap decorator ---

        def blocking(func):
            @wraps(func)
            def _wrapper(*args):
                if current.allow_cancel and current.cancel_pending:
                    exc = current.cancel_pending
                    current.cancel_pending = None
                    return exc
                else:
                    return func(*args)
            return _wrapper


        # --- Traps ---

        @blocking
        def trap_io(fileobj, event, state):
            if current._last_io != (fileobj, event):
                if current._last_io:
                    unregister_event(*current._last_io)
                try:
                    register_event(fileobj, event, current)
                except CurioError as e:
                    return e
            current._last_io = None
            suspend_task(state, lambda: unregister_event(fileobj, event))

        def trap_io_waiting(fileobj):
            try:
                key = selector_getkey(fileobj)
            except KeyError:
                return (None, None)
            else:
                rtask, wtask = key.data
                rtask = (rtask if rtask and rtask.cancel_func else None)
                wtask = (wtask if wtask and wtask.cancel_func else None)
                return (rtask, wtask)

        @blocking
        def trap_future_wait(future, event):
            current.future = future
            future.add_done_callback(lambda fut, task=current: wake(task, fut))
            if event:
                event.set()
            def _cancel(*, task=current):
                future.cancel()
                task.future = None
            suspend_task("FUTURE_WAIT", _cancel)

        def trap_spawn(coro):
            task = new_task(coro)
            task.parentid = current.id
            return task

        def trap_cancel_task(task, exc=TaskCancelled, val=None):
            if task.cancelled:
                return
            if not isinstance(exc, BaseException):
                exc = exc(exc.__name__ if val is None else val)
            task.cancelled = True
            task.timeout = None
            cancel_task(task, exc)

        @blocking
        def trap_sched_wait(sched, state):
            suspend_task(state, sched._kernel_suspend(current))

        def trap_sched_wake(sched, n):
            tasks = sched._kernel_wake(n)
            for task in tasks:
                reschedule_task(task)

        def trap_clock():
            return monotonic()

        @blocking
        def trap_sleep(clock, absolute):
            nonlocal running
            if clock == 0:
                reschedule_task(current)
                running = False
                return
            if not absolute:
                clock += monotonic()
            set_timeout(clock, "sleep")
            def _cancel(*, task=current):
                sleepq.cancel((task.id, "sleep"), task.sleep)
                task.sleep = None
            suspend_task("TIME_SLEEP", _cancel)

        def trap_set_timeout(timeout):
            old_timeout = current.timeout
            if timeout is not None:
                set_timeout(timeout, "timeout")
                if old_timeout and current.timeout > old_timeout:
                    current.timeout = old_timeout
            return old_timeout

        def trap_unset_timeout(previous):
            now = monotonic()
            set_timeout(None, "timeout")
            set_timeout(previous, "timeout")
            if not previous or previous >= now:
                current.timeout = previous
                if isinstance(current.cancel_pending, TaskTimeout):
                    current.cancel_pending = None
            return now

        @blocking
        def trap_event_wait(events):
            try:
                add_waiters(events)
            except GuioError as e:
                return e
            events._waiting = current
            def _cancel():
                events._waiting = None
                remove_waiters(events)
            suspend_task("EVENT_WAIT", _cancel)

        def trap_get_kernel():
            return kernel

        def trap_get_current():
            return current

        def trap_get_toplevel():
            return toplevel


        # --- Final setup ---

        # Trap table
        kernel._traps = traps = {
            key:value
            for key, value in locals().items()
            if key.startswith("trap_")
        }

        # Loopback sockets
        init_loopback()
        task = new_task(_kernel_task())
        task.daemon = True

        # Activations
        kernel._activations = _activations = [
            (act() if isinstance(act, type) and issubclass(act, Activation) else act)
            for act in kernel._activations
        ]
        for act in _activations:
            act.activate(kernel)

        # Bind toplevel "X" button
        events_set = set()
        toplevel.protocol("WM_DELETE_WINDOW", close_window(toplevel, "WM_DELETE_WINDOW", events_set))
        event_waiters[(toplevel, "WM_DELETE_WINDOW")] = (None, events_set)


        # --- Tkinter loop (run using tkinter's mainloop) ---
        # Note: A new inner loop is created for every piece of work that
        # gets submitted to the kernel. Shared state is stored outside
        # the inner loop to reduce slowdowns.

        def _inner_loop(coro):


            # --- Main loop preparation ---

            # Current task
            nonlocal current, running

            # Setup main task
            main_task = (new_task(coro) if coro else None)
            del coro


            # --- Main loop ---

            while True:


                # --- I/O event waiting ---

                try:
                    # Don't block here.
                    events = selector_select(0)
                except OSError as e:
                    # Windows throws an error if the selector is empty.
                    # Ignore it and set events to an empty tuple.
                    if e.errno != getattr(errno, "WSAEINVAL", None):
                        raise
                    events = ()

                # Reschedule I/O waiting tasks
                for key, mask in events:
                    rtask, wtask = key.data
                    intfd = isinstance(key.fileobj, int)

                    if mask & EVENT_READ:
                        rtask._last_io = (None if intfd else (key.fileobj, EVENT_READ))
                        reschedule_task(rtask)
                        mask &= ~EVENT_READ
                        rtask = None

                    if mask & EVENT_WRITE:
                        wtask._last_io = (None if intfd else (key.fileobj, EVENT_WRITE))
                        reschedule_task(wtask)
                        mask &= ~EVENT_WRITE
                        wtask = None

                    if intfd:
                        if mask:
                            selector_modify(key.fileobj, mask, (rtask, wtask))
                        else:
                            selector_unregister(key.fileobj)


                # --- Tkinter event waiting ---

                if ready or not main_task:
                    # A non-empty ready queue or an empty main task
                    # means that the waiting should be as close to
                    # non-blocking as possible.
                    timeout = 0
                    data = "NON_BLOCKING"

                else:
                    # Find the next deadline to wait for
                    now = monotonic()
                    timeout = sleepq.next_deadline(now)
                    data = "SLEEP_WAKE"

                    # Shorten timeouts if there is I/O that could
                    # complete in the future.
                    if (timeout is None) or (timeout > 0.1 and selector_getmap()):
                        timeout = 0.1
                        data = "SELECT"

                # Schedule after callback if required
                if timeout is not None:
                    id_ = frame.after(
                        max(int(timeout*1000), 1),
                        lambda data=data: loop.send(data)
                    )

                # Wait for callback
                try:
                    data = (yield)

                # Cancel after callback
                finally:
                    if timeout is not None:
                        frame.after_cancel(id_)


                # --- Timeout handling ---

                now = monotonic()
                for tm, (taskid, sleep_type) in sleepq.expired(now):
                    task = tasks.get(taskid)

                    if task is None:
                        continue
                    if tm != getattr(task, sleep_type):
                        continue

                    setattr(task, sleep_type, None)

                    if sleep_type == "sleep":
                        reschedule_task(task, now)
                    else:
                        cancel_task(task, TaskTimeout(now))


                # --- Running ready tasks ---

                for _ in range(len(ready)):
                    current = ready_popleft()
                    for a in _activations:
                        a.running(current)
                    current.state = "RUNNING"
                    current.cycles += 1
                    running = True

                    # Run task until it suspends or terminates
                    while running:

                        try:
                            # Send next value
                            trap = current._send(current._trap_result)

                        except BaseException as e:
                            # Wake joining tasks
                            for wtask in current.joining._kernel_wake(len(current.joining)):
                                reschedule_task(wtask)
                            current.terminated = True
                            current.state = "TERMINATED"
                            del tasks[current.id]
                            current.timeout = None

                            # Set task result / exception
                            if isinstance(e, StopIteration):
                                current.result = e.value
                            else:
                                current.exception = e
                                if (current != main_task and not isinstance(e, (CancelledError, SystemExit))):
                                    logger.error("Task Crash: %r", current, exc_info=True)
                                if not isinstance(e, Exception):
                                    raise
                            break

                        else:
                            # Find and run requested trap
                            current._trap_result = traps[trap[0]](*trap[1:])


                    # --- Task suspended / terminated ---

                    if current.suspend_func:
                        current.suspend_func()
                        current.suspend_func = None

                    if current._last_io:
                        unregister_event(*current._last_io)
                        current._last_io = None

                    for a in _activations:
                        a.suspended(current)
                        if current.terminated:
                            a.terminated(current)

                    current = None
                    running = False


                # --- Check if main task completed ---

                if main_task:
                    if main_task.terminated:
                        main_task.joined = True
                        return main_task
                else:
                    return None


        # --- Runner function ---

        def _runner(work):

            nonlocal loop, frame

            # Receive work and get result
            inner_loop = _inner_loop(work)
            del work

            # Wrap frame and loop in context managers
            with ExitStack() as inner_stack:
                frame = inner_stack.enter_context(destroying(tkinter.Frame(toplevel)))
                loop = inner_stack.enter_context(_GenWrapper(inner_loop, frame))
                del inner_loop

                # Run until frame is destroyed. Note that `wait_window`
                # will spawn in tkinter's event loop but will return
                # when the widget is destroyed. `frame` will be
                # destroyed when the loop ends or when an exception
                # happens while sending a value to the loop.
                loop.send(None)
                if not loop.closed:
                    frame.wait_window()

                # Check that the loop closed after `frame` was
                # destroyed.
                if not loop.closed:
                    raise RuntimeError("Frame closed before main task finished")

            try:
                # Check that toplevel still exists.
                if not exists(toplevel):
                    raise RuntimeError("Toplevel was destroyed")

                # Return end result
                return loop.result

            except BaseException:
                # If an exception happened in the loop, the kernel
                # "crashes" and stops any further attempt to use it.
                kernel._shutdown_funcs = None
                raise

        return _runner


@wraps(_run)
def run(
    corofunc, *args, with_monitor=False, selector=None, debug=None,
    activations=None, toplevel=None, **kernel_extra
):
    kernel = Kernel(
        selector=selector,
        debug=debug,
        activations=activations,
        toplevel=toplevel,
        **kernel_extra,
    )

    # Check if a monitor has been requested
    if with_monitor or "CURIOMONITOR" in os.environ:
        from curio.monitor import Monitor
        m = Monitor(kernel)
        kernel._call_at_shutdown(m.close)
        kernel.run(m.start)

    with kernel:
        return kernel.run(corofunc, *args)


# Wrapper class that hides away the implicit rescheduling when the
# generator is still running. Note that this doesn't attempt to fulfill
# the iterator protocol nor the coroutine methods as this is meant for
# use in tkinter callbacks. Exceptions that go unhandled in tkinter are
# hardcoded to be printed out; we prevent that from happening here.
class _GenWrapper:

    def __init__(self, gen, frame):
        self._gen = gen
        self._frame = frame
        self._final_val = None
        self._final_exc = None

    def __enter__(self):
        return self

    def __exit__(self, ty, val, tb):
        self.close()

    def __del__(self):
        self._gen.close()

    def _safe_call(func):
        @wraps(func)
        def _wrapper(self, *args, **kwargs):

            # Reschedule if called from within itself
            if self.running:
                self._frame.after(1, lambda: _wrapper(self, *args, **kwargs))
                return False

            # Only run if `gen` isn't closed
            if not self.closed:
                try:
                    func(self._gen, *args, **kwargs)
                except BaseException as e:
                    if isinstance(e, StopIteration):
                        self.result = e.value
                    else:
                        self.exception = e
                    destroy(self._frame)

            return True

        return _wrapper

    @property
    def result(self):
        if not self.closed:
            raise RuntimeError("Generator still running")
        if self._final_exc:
            raise self._final_exc
        return self._final_val

    @result.setter
    def result(self, value):
        self._final_val = value
        self._final_exc = None

    @property
    def exception(self):
        if not self.closed:
            raise RuntimeError("Generator still running")
        return self._final_exc

    @exception.setter
    def exception(self, value):
        self._final_val = None
        self._final_exc = value

    @_safe_call
    def send(gen, arg):
        return gen.send(arg)

    @_safe_call
    def throw(gen, ty, val=None, tb=None):
        return gen.throw(ty, val, tb)

    def close(self):
        self.throw(GeneratorExit)

    @property
    def closed(self):
        return self._gen.gi_frame is None

    @property
    def running(self):
        return bool(self._gen.gi_running)

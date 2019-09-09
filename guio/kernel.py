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

from curio import __version__ as CURIO_VERSION
from curio.errors import *
from curio.kernel import run as curio_run, Kernel as CurioKernel
from curio.sched import SchedBarrier
from curio.timequeue import TimeQueue
from curio.traps import _read_wait

from .errors import *
from .task import Task
from .utilities import *


__all__ = ["Kernel", "run"]


CURIO_VERSION = tuple(int(i) for i in CURIO_VERSION.split("."))


logger = logging.getLogger(__name__)


class Kernel(CurioKernel):

    _tk_events = (
        "<Activate>",
        "<Circulate>",
        "<Configure>",
        "<Colormap>",
        "<Deactivate>",
        "<FocusIn>",
        "<FocusOut>",
        "<Gravity>",
        "<Key>",
        "<KeyPress>",
        "<KeyRelease>",
        "<MouseWheel>",
        "<Property>",
    )

    _other_events = (
        "<Button>",
        "<ButtonPress>",
        "<ButtonRelease>",
        "<Enter>",
        "<Expose>",
        "<Leave>",
        "<Map>",
        "<Motion>",
        "<Reparent>",
        "<Unmap>",
        "<Visibility>",
    )


    def _make_kernel_runtime(kernel):

        # --- Kernel state ---

        # Current task / state
        current = None
        running = False

        # Toplevel and outer stack
        toplevel = None
        stack = ExitStack()
        kernel._call_at_shutdown(stack.close)

        # Inner loop and dummy frame (tkinter loop)
        loop = None
        frame = None

        # Restore kernel attributes
        selector = kernel._selector
        tasks = kernel._tasks

        # Internal kernel state
        event_queue = deque()
        event_wait = SchedBarrier()
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

        event_queue_append = event_queue.append
        event_queue_popleft = event_queue.popleft

        event_wait_add = event_wait.add
        event_wait_pop = event_wait.pop


        # --- Future processing ---

        # Loopback sockets
        notify_sock = None
        wait_sock = None

        async def kernel_task():
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
                    reschedule(task)

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

        def reschedule(task, val=None):
            ready_append(task)
            task.state = "READY"
            task.cancel_func = None
            task._trap_result = val

        def suspend(state, cancel_func):
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
            reschedule(task)
            for a in _activations:
                a.created(task)
            return task

        def cancel_task(task, exc):
            if task.allow_cancel and task.cancel_func:
                task.cancel_func()
                reschedule(task, exc)
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
                selector_register(
                    fileobj, event,
                    ((task, None) if event == EVENT_READ else (None, task)),
                )

            else:
                mask = key.events
                rtask, wtask = key.data
                if event == EVENT_READ and rtask:
                    raise ReadResourceBusy(f"Multiple tasks can't wait to read on the same file descriptor {fileobj!r}")
                if event == EVENT_WRITE and wtask:
                    raise WriteResourceBusy(f"Multiple tasks can't wait to write on the same file descriptor {fileobj!r}")

                selector_modify(
                    fileobj, mask | event,
                    ((task, wtask) if event == EVENT_READ else (rtask, task)),
                )

        def unregister_event(fileobj, event):
            key = selector_getkey(fileobj)
            mask = key.events
            rtask, wtask = key.data
            mask &= ~event
            if not mask:
                selector_unregister(fileobj)
            else:
                selector_modify(
                    fileobj, mask,
                    ((None, wtask) if event == EVENT_READ else (rtask, None)),
                )


        # --- Trap decorators ---

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

        def event(func):
            @wraps(func)
            def _wrapped(*args):
                if not current.is_event:
                    return TaskNotEvent("Non-event task cannot access events")
                else:
                    return func(*args)
            return _wrapped


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
            suspend(state, lambda: unregister_event(fileobj, event))

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

            _cancel = (
                lambda task=current:
                (future.cancel(), setattr(task, "future", None))
            )

            suspend("FUTURE_WAIT", _cancel)

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
            suspend(state, sched.add(current))

        def trap_sched_wake(sched, n):
            tasks = sched.pop(n)
            for task in tasks:
                reschedule(task)

        def trap_clock():
            return monotonic()

        @blocking
        def trap_sleep(clock, absolute):
            nonlocal running
            if clock == 0:
                reschedule(current)
                running = False
                return

            if not absolute:
                clock += monotonic()
            set_timeout(clock, "sleep")

            _cancel = (
                lambda task=current:
                (
                    sleepq.cancel((task.id, "sleep"), task.sleep),
                    setattr(task, "sleep", None),
                )
            )

            suspend("TIME_SLEEP", _cancel)

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

        @event
        def trap_pop_event():
            try:
                event = event_queue[current._next_event]
            except IndexError:
                return NoEvent("No event available")
            else:
                current._next_event += 1
                return event

        @blocking
        @event
        def trap_wait_event():
            if current._next_event >= len(event_queue):
                suspend("EVENT_WAIT", event_wait.add(current))

        def trap_get_kernel():
            return kernel

        def trap_get_current():
            return current

        def trap_get_toplevel():
            return toplevel


        # --- Tkinter helpers ---

        @contextmanager
        def bind(widget, func, events):
            widget_bind = widget.bind
            widget_unbind = widget.unbind
            bindings = [(event, widget_bind(event, func, "+")) for event in events]
            try:
                yield bindings
            finally:
                if exists(widget):
                    for info in bindings:
                        widget_unbind(*info)

        @contextmanager
        def protocol(toplevel, func):
            toplevel.protocol("WM_DELETE_WINDOW", func)
            try:
                yield
            finally:
                if exists(toplevel):
                    toplevel.protocol("WM_DELETE_WINDOW", toplevel.destroy)


        # --- Tkinter callback decorator ---

        # Decorator to return "break" for tkinter callbacks
        def callback(func):
            @wraps(func)
            def _wrapper(*args):
                func(*args)
                return "break"

            return _wrapper


        # --- Tkinter callbacks ---

        # Functions for event callbacks
        @callback
        def send_tk_event(event):
            if event.widget is toplevel:
                event_queue_append(event)
                if event_wait:
                    loop.send("EVENT_WAKE")

        @callback
        def send_other_event(event):
            if event.widget is not toplevel:
                event_queue_append(event)
                if event_wait:
                    loop.send("EVENT_WAKE")

        @callback
        def send_destroy_event(event):
            if event.widget is toplevel:
                event_queue_append(event)
                if event_wait:
                    loop.send("EVENT_WAKE")

        _last_close = None
        @callback
        def close_window():
            nonlocal _last_close
            event_queue_append(CloseWindow("X was pressed"))
            now = monotonic()
            if _last_close and _last_close + 0.5 > now:
                loop.throw(RuntimeError("Kernel was force closed"))
            else:
                if event_wait:
                    loop.send("EVENT_WAKE")
                _last_close = now


        # --- Final setup ---

        # Trap table
        kernel._traps = traps = {
            key:value
            for key, value in locals().items()
            if key.startswith("trap_")
        }

        # Loopback sockets
        init_loopback()
        task = new_task(kernel_task())
        task.daemon = True

        # Activations
        kernel._activations = _activations = [
            (act() if isinstance(act, type) and issubclass(act, Activation) else act)
            for act in kernel._activations
        ]
        for act in _activations:
            act.activate(kernel)

        # Toplevel creation
        toplevel = stack.enter_context(destroying(tkinter.Tk()))
        stack.enter_context(bind(toplevel, send_tk_event, kernel._tk_events))
        stack.enter_context(bind(toplevel, send_other_event, kernel._other_events))
        stack.enter_context(bind(toplevel, send_destroy_event, ("<Destroy>",)))
        stack.enter_context(protocol(toplevel, close_window))


        # --- Tkinter loop (run using tkinter's mainloop) ---
        # Note: A new inner loop is created for every piece of work that
        # gets submitted to the kernel. Shared state is stored outside
        # the inner loop to reduce slowdowns.

        def _inner_loop(coro):


            # --- Main loop preparation ---

            # Current task
            nonlocal current, running

            # Setup main task
            if coro:
                main_task = new_task(coro)
                main_task.report_crash = False
                main_task.is_event = True
            else:
                main_task = None
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
                        reschedule(rtask)
                        mask &= ~EVENT_READ
                        rtask = None

                    if mask & EVENT_WRITE:
                        wtask._last_io = (None if intfd else (key.fileobj, EVENT_WRITE))
                        reschedule(wtask)
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

                if data == "EVENT_WAKE":
                    for task in event_wait.pop(len(event_wait)):
                        reschedule(task)


                # --- Run event clearer (event garbage collection :P) ---

                event_tasks = {task for task in tasks.values() if task.is_event}

                # Check that there are event tasks and offset is
                # non-zero
                if event_tasks:
                    min_offset = min(task._next_event for task in event_tasks)
                    if min_offset:
                        for _ in range(min_offset):
                            event_queue.popleft()
                        for task in tasks.values():
                            if task.is_event:
                                task._next_event -= min_offset
                            else:
                                task._next_event = min(-1, task._next_event + min_offset)

                # Clear the queue if there aren't any tasks to collect
                # events. Note that this will leave at most 50 events on
                # the queue.
                elif len(event_queue) > 50:

                    # Leave 25 so that this doesn't run everytime a new
                    # event is added and the length pops over 50.
                    event_offset = len(event_queue) - 25
                    logger.info("Clearing %s events from event queue", event_offset)
                    for _ in range(event_offset):
                        event_queue.popleft()


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
                        reschedule(task, now)
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
                            for wtask in current.joining.pop(len(current.joining)):
                                reschedule(wtask)
                            current.terminated = True
                            current.state = "TERMINATED"
                            del tasks[current.id]
                            current.timeout = None

                            # Set task result / exception
                            if isinstance(e, StopIteration):
                                current.result = e.value
                            else:
                                current.exception = e
                                if current.report_crash and not isinstance(e, (CancelledError, SystemExit)):
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
                stack.close()  # Close the toplevel
                raise

        return _runner


@wraps(curio_run)
def run(corofunc, *args, with_monitor=False, **kernel_kwargs):
    kernel = Kernel(**kernel_kwargs)

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

            # Only run if `gen` isn't closed
            elif not self.closed:
                try:
                    return func(self._gen, *args, **kwargs)
                except BaseException as e:
                    if isinstance(e, StopIteration):
                        self.result = e.value
                    else:
                        self.exception = e
                    destroy(self._frame)

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

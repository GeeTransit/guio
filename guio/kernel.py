import errno
import logging
import os
import tkinter

from collections import deque
from contextlib import contextmanager, ExitStack
from inspect import getcoroutinestate, CORO_CLOSED, CORO_RUNNING
from functools import wraps
from selectors import EVENT_READ, EVENT_WRITE
from socket import socketpair
from time import monotonic
from types import coroutine

from curio import __version__ as CURIO_VERSION
from curio.errors import *
from curio.kernel import run as curio_run, Kernel
from curio.sched import SchedBarrier
from curio.task import Task
from curio.traps import _read_wait

from .errors import *
from .event import iseventtask


logger = logging.getLogger(__name__)


CURIO_VERSION = tuple(int(i) for i in CURIO_VERSION.split("."))


__all__ = ["TkKernel", "run"]


class TkKernel(Kernel):

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


    @wraps(Kernel.__init__)
    def __init__(self, *args, **kwargs):
        Kernel.__init__(self, *args, **kwargs)

        self._event_queue = deque()
        self._event_wait = SchedBarrier()


    def _run_coro(kernel):

        # --- Kernel state ---

        # Current task / state
        current = None
        running = False

        # Restore kernel state
        event_queue = kernel._event_queue
        event_wait = kernel._event_wait
        ready = kernel._ready
        selector = kernel._selector
        sleepq = kernel._sleepq
        tasks = kernel._tasks
        wake_queue = kernel._wake_queue
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

        async def kernel_task():
            wake_queue_popleft = wake_queue.popleft
            wait_sock = kernel._wait_sock

            while True:
                await _read_wait(wait_sock)

                try:
                    wait_sock.recv(1000)
                except BlockingIOError:
                    # This may raise an error as the 1 ms delay in
                    # tkinter's loop could cause the read to fail. 
                    continue

                while wake_queue:
                    task, future = wake_queue_popleft()
                    if future and task.future is not future:
                        continue
                    task.future = None
                    task.state = 'READY'
                    task.cancel_func = None
                    ready_append(task)

        def wake(task=None, future=None):
            if task:
                wake_queue.append((task, future))
            kernel._notify_sock.send(b'\x00')

        def init_loopback():
            kernel._notify_sock, kernel._wait_sock = socketpair()
            kernel._wait_sock.setblocking(False)
            kernel._notify_sock.setblocking(False)
            kernel._call_at_shutdown(kernel._notify_sock.close)
            kernel._call_at_shutdown(kernel._wait_sock.close)


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
                _unregister_event(*current._last_io)
                current._last_io = None

            running = None

        def new_task(coro):
            task = Task(coro)
            tasks[task.id] = task
            reschedule(task)
            return task

        def cancel_task(task, *, exc=TaskCancelled, val=None):
            if not isinstance(exc, BaseException):
                exc = exc(exc.__name__ if val is None else val)

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
                    raise ReadResourceBusy("Multiple tasks can't wait to read on the same file descriptor %r" % fileobj)
                if event == EVENT_WRITE and wtask:
                    raise WriteResourceBusy("Multiple tasks can't wait to write on the same file descriptor %r" % fileobj)

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
                if not iseventtask(current):
                    return TaskNotEvent("Non-event task cannot access events")
                else:
                    return func(*args)
            return _wrapped


        # --- Traps ---

        @blocking
        def _trap_io(fileobj, event, state):
            if current._last_io != (fileobj, event):
                if current._last_io:
                    unregister_event(*current._last_io)
                try:
                    register_event(fileobj, event, current)
                except CurioError as e:
                    return e

            current._last_io = None
            suspend(state, lambda: unregister_event(fileobj, event))

        def _trap_io_waiting(fileobj):
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
        def _trap_future_wait(future, event):
            current.future = future
            future.add_done_callback(lambda fut, task=current: wake(task, fut))
            if event:
                event.set()

            suspend(
                'FUTURE_WAIT',
                lambda task=current: (
                    future.cancel(),
                    setattr(task, 'future', None),
                ),
            )

        def _trap_spawn(coro):
            task = new_task(coro)
            task.parentid = current.id
            return task

        def _trap_cancel_task(task, exc=TaskCancelled, val=None):
            if task.cancelled:
                return
            task.cancelled = True
            task.timeout = None
            cancel_task(task, exc=exc, val=val)

        @blocking
        def _trap_sched_wait(sched, state):
            suspend(state, sched.add(current))

        def _trap_sched_wake(sched, n):
            tasks = sched.pop(n)
            for task in tasks:
                reschedule(task)

        def _trap_clock():
            return monotonic()

        @blocking
        def _trap_sleep(clock, absolute):
            nonlocal running
            if clock == 0:
                reschedule(current)
                running = False
                return

            if not absolute:
                clock += monotonic()
            set_timeout(clock, "sleep")

            suspend(
                "TIME_SLEEP",
                lambda task=current: (
                    sleepq.cancel((task.id, "sleep"), task.sleep),
                    setattr(task, "sleep", None),
                ),
            )

        def _trap_set_timeout(timeout):
            old_timeout = current.timeout

            if timeout is not None:
                set_timeout(timeout, "timeout")

                if old_timeout and current.timeout > old_timeout:
                    current.timeout = old_timeout

            return old_timeout

        def _trap_unset_timeout(previous):
            now = monotonic()
            set_timeout(previous, "timeout")

            if not previous or previous >= now:
                current.timeout = previous
                if isinstance(current.cancel_pending, TaskTimeout):
                    current.cancel_pending = None

            return now

        @event
        def _trap_pop_event():
            try:
                event = event_queue[current.next_event]
            except IndexError:
                return NoEvent("No event available")
            else:
                current.next_event += 1
                return event

        @blocking
        @event
        def _trap_wait_event():
            if current.next_event >= len(event_queue):
                suspend("EVENT_WAIT", event_wait.add(current))

        def _trap_get_kernel():
            return kernel

        def _trap_get_current():
            return current

        def _trap_get_toplevel():
            return toplevel


        # --- Tkinter helpers ---

        def exists(widget):
            try:
                return bool(widget.winfo_exists())
            except tkinter.TclError:
                return False

        def destroy(widget):
            try:
                widget.destroy()
            except tkinter.TclError:
                pass

        def getasyncgenstate(asyncgen):
            if asyncgen.ag_running:
                return "AGEN_RUNNING"
            if asyncgen.ag_frame is None:
                return "AGEN_CLOSED"
            if asyncgen.ag_frame.f_lasti == -1:
                return "AGEN_CREATED"
            return "AGEN_SUSPENDED"

        async def wrap_coro(coro):
            return await coro

        def send(gen, data):
            try:
                return gen.send(data)
            except BaseException as e:
                nonlocal result
                destroy(frame)
                if isinstance(e, StopAsyncIteration):
                    pass
                elif isinstance(e, StopIteration):
                    result = e.value
                else:
                    result = e

        @contextmanager
        def bind(widget, func, events):
            widget_bind = widget.bind
            widget_unbind = widget.unbind
            bindings = [(event, widget_bind(event, func, "+")) for event in events]
            try:
                yield bindings
            finally:
                for info in bindings:
                    widget_unbind(*info)

        @contextmanager
        def protocol(toplevel, func):
            toplevel.protocol("WM_DELETE_WINDOW", func)
            try:
                yield
            finally:
                toplevel.protocol("WM_DELETE_WINDOW", toplevel.destroy)


        # --- Tkinter loop helpers ---

        # Send if the cycle is suspended (True if sent, False otherwise)
        _unsafe_states = frozenset({CORO_CLOSED, CORO_RUNNING})
        def safe_send(data, *, retry=True, _unsafe_states=_unsafe_states):
            state = getcoroutinestate(cycle)
            if state in _unsafe_states:
                if retry:
                    frame.after(1, lambda: safe_send(data, retry=True))
                return False
            else:
                send(cycle, data)
                return True

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
                    safe_send("EVENT_WAKE")

        @callback
        def send_other_event(event):
            if event.widget is not toplevel:
                event_queue_append(event)
                if event_wait:
                    safe_send("EVENT_WAKE")

        @callback
        def send_destroy_event(event):
            if event.widget is toplevel:
                event_queue_append(event)
                if event_wait:
                    frame.after(1, lambda: safe_send("EVENT_WAKE"))

        @callback
        def close_window():
            event_queue_append(CloseWindow("X was pressed"))
            if event_wait:
                safe_send("EVENT_WAKE")


        # --- Outer loop helper functions ---

        @contextmanager
        def destroying(widget):
            try:
                yield widget
            except tkinter.TclError:
                logger.info("Widget exception: %r", widget, exc_info=True)
                return True
            finally:
                destroy(widget)

        @contextmanager
        def aclosing(asyncgen):
            try:
                yield asyncgen
            finally:
                aclose = asyncgen.aclose()
                try:
                    for _ in range(100):
                        aclose.send(None)
                except StopIteration:
                    pass
                else:
                    logger.warn("Async gen didnt close properly: %r", asyncgen)
                    aclose.close()


        # --- Final setup ---

        kernel._traps = traps = {
            key:value
            for key, value in locals().items()
            if key.startswith("_trap_")
        }

        if kernel._kernel_task_id is None:
            init_loopback()
            t = new_task(kernel_task())
            t.daemon = True
            kernel._kernel_task_id = t.id
            del t

        _activations = [
            act() if (isinstance(act, type) and issubclass(act, Activation)) else act
            for act in kernel._activations
        ]
        kernel._activations = _activations

        for act in _activations:
            act.activate(kernel)
            if kernel._kernel_task_id:
                act.created(kernel._tasks[kernel._kernel_task_id])

        main_task = None


        # --- Tkinter loop (run using tkinter's mainloop) ---

        async def _inner_loop():


            # --- Main loop preparation ---

            # Current task
            nonlocal current, running

            # Only way to suspend for tkinter callback
            @coroutine
            def _suspend_cycle():
                return (yield)

            # Main task
            main_task = None


            # --- Main loop ---

            while True:


                # --- Get new work to run ---

                if (main_task and main_task.terminated) or (not ready and not main_task):
                    if main_task:
                        main_task.joined = True
                    coro = (yield main_task)
                    if coro:
                        main_task = new_task(coro)
                        main_task.report_crash = False
                        main_task.next_event = 0
                    else:
                        main_task = None
                    del coro


                # --- I/O event waiting ---

                try:
                    # Return immediately
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

                # Ensure a resumation if there are any ready tasks,
                # possible select calls in the future, or an empty main
                # task.
                if ready or selector_getmap() or not main_task:
                    timeout = 0
                    data = "READY"
                else:
                    now = monotonic()
                    timeout = sleepq.next_deadline(now)
                    data = "SLEEP_WAKE"

                # Set timeout if required
                if timeout is not None:
                    id_ = frame.after(max(int(timeout*1000), 1), lambda: safe_send(data))

                # Wait for callback
                try:
                    info = await _suspend_cycle()

                # Cancel after callback
                finally:
                    if timeout is not None:
                        frame.after_cancel(id_)

                if info == "EVENT_WAKE":
                    for task in event_wait.pop(len(event_wait)):
                        reschedule(task)


                # --- Run event clearer (event garbage collection :P) ---

                event_tasks = [task for task in tasks.values() if iseventtask(task)]
                if event_tasks:
                    offset = min(task.next_event for task in event_tasks)
                    if offset:
                        for _ in range(offset):
                            event_queue.popleft()
                        for task in event_tasks:
                            task.next_event -= offset

                # Clear the queue if there aren't any tasks to collect events
                # Note: This will leave at most 50 events on the queue.
                elif len(event_queue) > 50:

                    # Leave 25 so that this doesn't run everytime a new event
                    # is added and the length pops over 50.
                    offset = len(event_queue) - 25
                    logger.info("Clearing %s events from event queue", offset)
                    for _ in range(offset):
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
                        cancel_task(task, exc=TaskTimeout(now))


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


                    # --- Task suspended ---

                    if current.suspend_func:
                        current.suspend_func()
                        current.suspend_func = None

                    if current._last_io:
                        unregister(*current._last_io)
                        current._last_io = None

                    for a in _activations:
                        a.suspended(current)
                        if current.terminated:
                            a.terminated(current)

                    current = None
                    running = False


        # --- Outer loop preparation ---

        result = None
        toplevel = frame = None
        loop = cycle = None

        # Wrap toplevel and loop with closing managers
        with ExitStack() as stack:
            enter = stack.enter_context

            toplevel = enter(destroying(tkinter.Tk()))
            kernel._call_at_shutdown(lambda: destroy(toplevel))

            loop = enter(aclosing(_inner_loop()))
            enter(bind(toplevel, send_tk_event, kernel._tk_events))
            enter(bind(toplevel, send_other_event, kernel._other_events))
            enter(bind(toplevel, send_destroy_event, ("<Destroy>",)))
            enter(protocol(toplevel, close_window))

            cycle = wrap_coro(loop.asend(None))
            with destroying(tkinter.Frame(toplevel)) as frame:
                send(cycle, None)
                frame.wait_window()


            # --- Outer loop ---

            while True:

                # If an exception happened, raise it here
                if isinstance(result, BaseException):
                    raise result from None

                # Get coro to run
                data = (yield result)
                cycle = wrap_coro(loop.asend(data))
                del data

                # Run until frame is destroyed
                # Note: `wait_window` will spawn in tkinter's event loop
                # but will end when the widget is destroyed. `frame` will
                # be destroyed when an exception happens in sending a value
                # to the cycle.
                with destroying(tkinter.Frame(toplevel)) as frame:
                    send(cycle, None)
                    frame.wait_window()

                # Check for exceptions
                if getcoroutinestate(cycle) != CORO_CLOSED:
                    cycle.close()
                    raise RuntimeError("Frame closed before main task finished") from result

                if not exists(toplevel):
                    raise RuntimeError("Toplevel was destroyed") from result

                if getasyncgenstate(loop) == "AGEN_CLOSED":
                    raise RuntimeError("Loop was closed") from result


@wraps(curio_run)
def run(corofunc, *args, with_monitor=False, **kernel_kwargs):
    """
    Run the guio kernel with an initial task and execute until the
    initial task terminates. Returns the task's final result (if any).
    This is a convenience function that should primarily be used for
    launching the top-level task of an curio/guio-based application.  It
    creates an entirely new kernel, runs the given task to completion,
    and concludes by shutting down the kernel, releasing all resources
    used.

    Don't use this function if you're repeatedly launching a lot of
    new tasks to run in curio/guio. Instead, create a Kernel instance and
    use its run() method instead.
    """

    kernel = TkKernel(**kernel_kwargs)

    # Check if a monitor has been requested
    if with_monitor or 'CURIOMONITOR' in os.environ:
        from .monitor import Monitor
        m = Monitor(kernel)
        kernel._call_at_shutdown(m.close)

    with kernel:
        return kernel.run(corofunc, *args)


if __name__ == "__main__":
    async def test():
        return 0
    run(test)

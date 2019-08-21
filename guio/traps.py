from types import coroutine


__all__ = ["_pop_event", "_wait_event", "_get_toplevel"]


@coroutine
def _pop_event():
    return (yield ("_trap_pop_event",))


@coroutine
def _wait_event():
    return (yield ("_trap_wait_event",))


@coroutine
def _get_toplevel():
    return (yield ("_trap_get_toplevel",))

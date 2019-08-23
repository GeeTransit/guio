from curio.traps import _kernel_trap


__all__ = ["_pop_event", "_wait_event", "_get_toplevel"]


def _pop_event():
    return await _kernel_trap("_trap_pop_event")


def _wait_event():
    return await _kernel_trap("_trap_wait_event")


def _get_toplevel():
    return await _kernel_trap("_trap_get_toplevel")

from curio.traps import _kernel_trap


__all__ = ["_event_wait", "_get_toplevel"]


async def _event_wait(events):
    return await _kernel_trap("trap_event_wait", events)


async def _get_toplevel():
    return await _kernel_trap("trap_get_toplevel")

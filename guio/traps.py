from curio.traps import _kernel_trap


__all__ = ["_get_toplevel"]


async def _get_toplevel():
    return await _kernel_trap("trap_get_toplevel")

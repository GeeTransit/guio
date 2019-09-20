from .traps import _get_toplevel


__all__ = ["current_toplevel"]


async def current_toplevel():
    return await _get_toplevel()

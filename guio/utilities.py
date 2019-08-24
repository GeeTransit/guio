from contextlib import contextmanager
from tkinter import Tk

from curio.thread import spawn_thread, AWAIT


__all__ = ["destroying", "run_in_main", "dialog"]


@contextmanager
def destroying(widget):
    try:
        yield widget
    except tkinter.TclError:
        logger.info("Widget exception: %r", widget, exc_info=True)
        return True
    finally:
        destroy(widget)


async def _run_in_main_helper(func, args, kwargs):
    return func(*args, **kwargs)


# Use for tkinter calls to get state
async def run_in_main(func_, *args, **kwargs):
    async with spawn_thread():
        return AWAIT(_run_in_main_helper(func_, args, kwargs))


# Use for dialogs without blocking the coroutine
async def dialog(func_, *args, **kwargs):
    async with spawn_thread():
        with destroying(tkinter.Tk()) as root:
            root.withdraw()
            return func_(*args, **kwargs)

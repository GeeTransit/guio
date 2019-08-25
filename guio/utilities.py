import logging

from contextlib import contextmanager
from tkinter import TclError, Tk

from curio.thread import spawn_thread, AWAIT


__all__ = [
    "run_in_main", "dialog",
    "exists", "destroy", "destroying",
]


logger = logging.getLogger(__name__)


async def _run_in_main_helper(func, args, kwargs):
    return func(*args, **kwargs)


# Use for tkinter calls to get state
async def run_in_main(func_, *args, **kwargs):
    async with spawn_thread():
        return AWAIT(_run_in_main_helper(func_, args, kwargs))


# Use for dialogs without blocking the coroutine
async def dialog(func_, *args, **kwargs):
    async with spawn_thread():
        with destroying(Tk()) as root:
            root.withdraw()
            return func_(root, *args, **kwargs)


def exists(widget):
    try:
        return bool(widget.winfo_exists())
    except TclError:
        return False


def destroy(widget):
    try:
        widget.destroy()
    except TclError as e:
        if "application has been destroyed" not in str(e):
            logger.warn("Widget destruction error: %r", widget, exc_info=True)
    if isinstance(widget, tkinter.Misc) and not isinstance(widget, tkinter.Widget):
        # We have to close the Tcl interpreter in its thread of creation
        # or cleanup will be done on the main thread. Calling `.quit()`
        # cleans up the interpreter on the same thread, preventing a
        # `Tcl_AsyncDelete` error from hanging the whole process.
        # https://stackoverflow.com/a/27077347/
        widget.quit()


@contextmanager
def destroying(widget):
    try:
        yield widget
    finally:
        destroy(widget)

import logging

from contextlib import contextmanager
from tkinter import Misc, TclError, Tk, Widget

from curio.errors import TaskCancelled
from curio.thread import spawn_thread, AWAIT

from .event import current_toplevel

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


def _dialog_helper(func, args, kwargs, x, y, close):
    with destroying(Tk()) as root:
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_set()
        root.title("Dialog")
        root.geometry(f"0x0+{x}+{y}")
        callback = (
            lambda:
            root.after(250, callback)
            if not close[0]
            else destroy(root)
        )
        root.after(500, callback)
        return func(root, *args, **kwargs)


# Use for dialogs without blocking the coroutine
async def dialog(func_, *args, **kwargs):
    toplevel = await current_toplevel()
    geometry = toplevel.geometry()
    _, x, y = geometry.split("+")  # "WxH+X+Y".split("+") == ["WxH", "X", "Y"]
    close = [False]
    thread = await spawn_thread(_dialog_helper, func_, args, kwargs, x, y, close)
    try:
        return await thread.join()
    finally:
        close[0] = True


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


@contextmanager
def destroying(widget):
    try:
        yield widget
    finally:
        destroy(widget)

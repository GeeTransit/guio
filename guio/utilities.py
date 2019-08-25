from contextlib import contextmanager
from tkinter import TclError, Tk

from curio.thread import spawn_thread, AWAIT


__all__ = [=
    "run_in_main", "dialog",
    "exists", "destroy", "destroying",
]


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
            return func_(*args, **kwargs)


def exists(widget):
    try:
        return bool(widget.winfo_exists())
    except TclError:
        return False


def destroy(widget):
    try:
        widget.destroy()
    except TclError:
        pass


@contextmanager
def destroying(widget):
    try:
        yield widget
    finally:
        destroy(widget)

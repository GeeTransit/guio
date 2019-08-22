Guio - Curio-Tkinter Compatible Kernel
======================================
Guio is a small library that provides a Curio compatible kernel with Tkinter support. This means you can continue using Curio but have a GUI at the same time!

Here's a small example:

```python
import tkinter
import curio
import guio

async def main():
    await guio.set_current_event()

    toplevel = await guio.current_toplevel()
    canvas = tkinter.Canvas(toplevel, highlightthickness=0)
    canvas.pack(expand=True, fill=tkinter.BOTH)

    try:
        async for event in guio.aevents():
            if event.type in {tkinter.EventType.Motion, tkinter.EventType.Enter}:
                x, y = event.x, event.y
                canvas.create_line(x-2, y-2, x+2, y+2)

    except guio.CloseWindow:
        pass

guio.run(main)
```

Links
-----
To fully appreciate asynchronous Python, check out [curio](https://github.com/dabeaz/curio "curio - concurrent I/O")!

About
-----
Guio was created by George Zhang (@geetransit).

All contributions are welcome!

(How do you pronounce this?)

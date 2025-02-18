Guio - Curio-Tkinter Compatible Kernel
======================================
Guio is a small library that provides a Curio compatible kernel with Tkinter support. This means you can continue to use Curio but have a GUI at the same time!

Here's a small example using just Guio:

```python
import tkinter
import guio

async def main():
    # Create canvas in a familiar way
    root = await guio.current_toplevel()
    canvas = tkinter.Canvas(root, highlightthickness=0)
    canvas.pack(expand=True, fill=tkinter.BOTH)

    # Get events from canvas
    async with guio.EventQueue({canvas: guio.EVENT_ALL}) as events:
        async for event in events:

            # Event responding
            event_type = str(event.type)
            if event.type == "WM_DELETE_WINDOW":
                break
            elif event.type in {"Motion", "Enter"}:
                x, y = event.x, event.y
                canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")

# Simple running of programs
guio.run(main)
```

Links
-----
To fully appreciate asynchronous Python, check out [Curio](https://github.com/dabeaz/curio "curio - concurrent I/O")!

About
-----
Guio was created by [George Zhang](https://github.com/GeeTransit "@GeeTransit") after trying to get rid of them tkinter callbacks. Now tkinter, threads, and asynchronous code can work together for the better of humanity (I hope).

All contributions are welcome!

(How do you pronounce this?)

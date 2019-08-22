import random
import tkinter
from tkinter import colorchooser

import guio



async def drawing_canvas():

    await guio.set_current_event()

    tk = await guio.current_toplevel()
    canvas = tkinter.Canvas(tk, highlightthickness=0)
    canvas.pack(expand=True, fill="both")

    def random_colour():
        return "#" + "".join(hex(random.getrandbits(4) + 8)[2] for _ in range(6))

    async def clear(colour):
        canvas.delete(tkinter.ALL)
        canvas["bg"] = colour

    x = y = None
    last = None
    colour = random_colour()

    try:

        async for event in guio.aevents():
            event_type = str(event.type)
            if event_type in {"Motion", "Enter"}:
                if x is None:
                    x, y = event.x, event.y
                canvas.create_line(event.x, event.y, x, y, fill=colour, width=3)
                x, y = event.x, event.y
                canvas.create_oval(
                    x - 1, y - 1, x + 1, y + 1,
                    outline=colour, fill=colour,
                )
            elif event_type == "Leave":
                x = y = None
            elif event_type == "ButtonPress":
                last = None
                if event.num == 3:
                    canvas.delete("all")
                elif event.num == 1:
                    new = colorchooser.askcolor()[1]
                    if new:
                        colour = new
            elif event_type == "KeyPress":
                if last:
                    text = canvas.itemcget(last, "text")
                    canvas.itemconfig(last, text=text + event.char)
                else:
                    last = canvas.create_text(
                        event.x, event.y,
                        text=event.char, anchor="sw",
                    )
            else:
                last = None

    except guio.CloseWindow:
        pass



if __name__ == "__main__":
    guio.run(drawing_canvas())

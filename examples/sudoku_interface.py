import itertools
import tkinter

import curio
import guio



class SudokuGrid:

    __slots__ = ("grid",)

    def __new__(cls, grid=None):
        inst = super().__new__(cls)
        if grid is None:
            inst.grid = [[0 for _ in range(9)] for _ in range(9)]
        elif isinstance(grid, cls):
            inst.grid = [[i for i in row] for row in grid.grid]
        else:
            inst.grid = [[i for i in row] for row in grid]
        return inst

    def __repr__(self):
        return f"{type(self).__qualname__}({self.grid})"

    def __eq__(self, other):
        if isinstance(other, list):
            grid = other
        elif isinstance(other, type(self)):
            grid = other.grid
        else:
            return NotImplemented
        return self.grid == grid

    @staticmethod
    def _check_xy(x, y):
        if not 0 <= x <= 8:
            raise ValueError("`x` is not from 0 to 8")
        if not 0 <= y <= 8:
            raise ValueError("`y` is not from 0 to 8")

    def replace(self, x, y, num):
        self._check_xy(x, y)
        if not 0 <= num <= 9:
            raise ValueError("`num` is not from 1 to 9")
        self.grid[y][x] = num

    def at(self, x, y):
        self._check_xy(x, y)
        return self.grid[y][x]

    def possible_at(self, x, y):
        self._check_xy(x, y)
        num = self.at(x, y)
        if num != 0:
            return {num}
        square = [*self.get_square(x, y)]
        row = self.grid[y]
        column = [row[x] for row in self.grid]
        return {
            n
            for n in range(1, 10)
            if n not in itertools.chain(square, row, column)
        }

    # Pretty hard to reimplement and copy everywhere...
    def get_square_coords(self, x, y):
        # x, y is a square in the square
        return [
            (i, j)
            for j in range(3*(y//3), 3*(y//3) + 3)
            for i in range(3*(x//3), 3*(x//3) + 3)
        ]

    def all_square_coords(self):
        return [self.get_square_coords(x*3, y*3) for x in range(3) for y in range(3)]

    def get_square(self, x, y):
        # x, y is a square in the square
        return [self.at(i, j) for i, j in self.get_square_coords(x, y)]

    def all_squares(self):
        return [self.get_square(x*3, y*3) for x in range(3) for y in range(3)]

    def copy(self):
        return SudokuGrid(self)

    @property
    def valid(self):
        s = set()
        for nums in itertools.chain(self.grid, zip(*self.grid), self.all_squares()):
            s.clear()
            for n in nums:
                if n in s:
                    return False
                if n != 0:
                    s.add(n)
        return True

    @property
    def full(self):
        return all(n != 0 for row in self.grid for n in row)



def draw_grid(canvas, x, y, w, h, rows=1, columns=1, **kwargs):
    for i in range(columns + 1):
        _x = x + i*w
        canvas.create_line(_x, y, _x, y + rows*h, **kwargs)
    for j in range(rows + 1):
        _y = y + j*h
        canvas.create_line(x, _y, x + columns*w, _y, **kwargs)

def draw_square(canvas, x, y, cx, cy, size, fill):
    if 0 <= cx <= 8 and 0 <= cy <= 8:
        canvas.create_rectangle(
            x + cx*size, y + cy*size,
            x + cx*size + size, y + cy*size + size,
            fill=fill,
        )

def draw_sudoku_grid(canvas, x, y, size, width):
    draw_grid(canvas, x, y, size, size, 9, 9)
    draw_grid(canvas, x, y, size*3, size*3, 3, 3, width=3, capstyle="round")

def draw_nums(canvas, grid, x, y, size):
    for i in range(9):
        for j in range(9):
            num = grid.at(i, j)
            if num != 0:
                canvas.create_text(
                    int(x + i*size + size/2),
                    int(y + j*size + size/2),
                    text=str(num),
                    font=("Helvetica", 20),
                )

def draw_impossible(canvas, grid, x, y, size):
    for i in range(9):
        for j in range(9):
            num = grid.at(i, j)
            if num == 0:
                possible = grid.possible_at(i, j)
                if not possible:
                    draw_square(canvas, x, y, i, j, size, "#ffdddd")

def draw_hints(canvas, grid, x, y, size):
    for i in range(9):
        for j in range(9):
            num = grid.at(i, j)
            if num == 0:
                possible = grid.possible_at(i, j)
                if possible:
                    canvas.create_text(
                        int(x + i*size + size/2),
                        int(y + j*size + 3*size/4),
                        text=" ".join(map(str, sorted(possible))),
                        font=("Helvetica", 5),
                        anchor="n"
                    )



def attempt_once(grid_copy):
    temp_set = set()

    for y, row in enumerate(grid_copy.grid):
        for x, num in enumerate(row):
            if num != 0:
                continue
            possible = grid_copy.possible_at(x, y)
            if not possible:
                return grid_copy
            if len(possible) == 1:
                grid_copy.replace(x, y, possible.pop())

    for y, row in enumerate(grid_copy.grid):
        for num in range(1, 10):
            if num in row:
                continue
            temp_set.clear()
            for x in range(9):
                if num in grid_copy.possible_at(x, y):
                    temp_set.add(x)
            if not temp_set:
                return grid_copy
            if len(temp_set) == 1:
                grid_copy.replace(temp_set.pop(), y, num)

    for x, column in enumerate(zip(*grid_copy.grid)):
        for num in range(1, 10):
            if num in column:
                continue
            temp_set.clear()
            for y in range(9):
                if num in grid_copy.possible_at(x, y):
                    temp_set.add(y)
            if not temp_set:
                return grid_copy
            if len(temp_set) == 1:
                grid_copy.replace(x, temp_set.pop(), num)

    for square_coords, square in zip(
        grid_copy.all_square_coords(),
        grid_copy.all_squares(),
    ):
        for num in range(1, 10):
            if num in square:
                continue
            temp_set.clear()
            for x, y in square_coords:
                if num in grid_copy.possible_at(x, y):
                    temp_set.add((x, y))
            if not temp_set:
                return grid_copy
            if len(temp_set) == 1:
                grid_copy.replace(*temp_set.pop(), num)

    return grid_copy

async def attempt_finish(grid):
    same = 0
    while not grid.full:
        await curio.schedule()
        grid_attempt = attempt_once(grid.copy())
        grid.grid = grid_attempt.grid
        if grid_attempt == grid:
            print(grid_attempt, grid)
            break
        else:
            same = 0
    return grid

async def program(
    grid=None,
    *,
    x=50,
    y=50,
    size=60,
    width=5,
    cfill="#ccccee",
    hints=False,
):

    await guio.set_current_event()

    tk = await guio.current_toplevel()
    canvas = tkinter.Canvas(tk, highlightthickness=0)
    canvas.pack(expand=True, fill="both")

    if grid is None:
        grid = SudokuGrid()

    current_x = -1
    current_y = -1
    solver = None

    try:
        while True:
            canvas.delete("all")

            draw_impossible(canvas, grid, x, y, size)
            draw_square(canvas, x, y, current_x, current_y, size, cfill)
            if hints:
                draw_hints(canvas, grid, x, y, size)
            draw_sudoku_grid(canvas, x, y, size, width=width)
            draw_nums(canvas, grid, x, y, size)

            canvas.create_text(
                10, 10,
                text="VALID" if grid.valid else "INVALID",
                font=("Helvetica", 20),
                fill="#000000" if grid.valid else "#ff0000",
                anchor="nw"
            )

            event = await guio.pop_event()

            et = str(event.type)

            if et == "KeyPress":

                if event.keysym == "h":
                    hints = (not hints)

                elif event.keysym == "Return":
                    if not solver or solver.terminated:
                        solver = await curio.spawn(attempt_finish(grid), daemon=True)

                    elif solver:
                        await solver.cancel()

                elif 0 <= current_x <= 8 and 0 <= current_y <= 8:
                    if event.keysym in iter("1234567879"):
                        grid.replace(current_x, current_y, int(event.keysym))
                    elif event.keysym in {"BackSpace", "Delete"}:
                        grid.replace(current_x, current_y, 0)
                    elif event.keysym == "Escape":
                        if grid.at(current_x, current_y):
                            grid.replace(current_x, current_y, 0)
                        else:
                            current_x = current_y = -1
                    elif event.keysym == "Tab":
                        possible = grid.possible_at(current_x, current_y)
                        if len(possible) == 1:
                            grid.replace(current_x, current_y, possible.pop())
                        else:
                            coords = grid.get_square_coords(current_x, current_y)
                            here = set()
                            for n in range(1, 9):
                                if n in possible:
                                    for i, j in coords:
                                        if (i, j) != (current_x, current_y):
                                            if n in grid.possible_at(i, j):
                                                break
                                    else:
                                        here.add(n)
                            if len(here) == 1:
                                grid.replace(current_x, current_y, here.pop())
                    elif event.keysym == "Up":
                        if current_y > 0:
                            current_y -= 1
                    elif event.keysym == "Down":
                        if current_y < 8:
                            current_y += 1
                    elif event.keysym == "Left":
                        if current_x > 0:
                            current_x -= 1
                    elif event.keysym == "Right":
                        if current_x < 8:
                            current_x += 1

                else:
                    if event.keysym == "Tab":
                        g = await attempt_finish(grid.copy())
                        canvas.create_text(
                            200, 10,
                            text="UNIQUE" if g.full else "AMBIGUOUS",
                            font=("Helvetica", 20),
                            fill="#00ff00" if g.full else "#ff0000",
                            anchor="nw"
                        )
                        await curio.sleep(2)

            elif et == "ButtonPress":
                if event.num == 1:
                    current_x = (event.x - x)//size
                    current_y = (event.y - y)//size
                elif event.num == 3:
                    grid.replace((event.x - x)//size, (event.y - y)//size, 0)

    except guio.CloseWindow:
        pass

    return grid



if __name__ == "__main__":
    print(guio.run(program()))

import pygame
import random

# ---------- Maze (DFS) ----------
WALL, PATH = 1, 0

def generate_maze(cols, rows):
    # grid size must be odd for nice walls
    if cols % 2 == 0: cols += 1
    if rows % 2 == 0: rows += 1

    grid = [[WALL for _ in range(cols)] for _ in range(rows)]

    def neighbors(cx, cy):
        # move by 2 to carve passages with walls between
        dirs = [(2,0), (-2,0), (0,2), (0,-2)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < cols-1 and 1 <= ny < rows-1:
                yield nx, ny, dx, dy

    stack = [(1,1)]
    grid[1][1] = PATH

    while stack:
        cx, cy = stack[-1]
        carved = False
        for nx, ny, dx, dy in neighbors(cx, cy):
            if grid[ny][nx] == WALL:
                # carve wall between
                grid[cy + dy//2][cx + dx//2] = PATH
                grid[ny][nx] = PATH
                stack.append((nx, ny))
                carved = True
                break
        if not carved:
            stack.pop()

    start = (1, 1)
    goal = (cols-2, rows-2)
    grid[goal[1]][goal[0]] = PATH
    return grid, start, goal

def can_move(grid, x, y):
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == PATH

# ---------- Pygame Game ----------
pygame.init()

CELL = 28
COLS, ROWS = 21, 21  # change later
grid, start, goal = generate_maze(COLS, ROWS)

W = len(grid[0]) * CELL
H = len(grid) * CELL + 60  # HUD area
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Motion Maze (DFS)")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

px, py = start
moves = 0
win = False

def draw():
    screen.fill((20, 20, 20))

    # maze
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            color = (40, 40, 40) if grid[y][x] == WALL else (220, 220, 220)
            pygame.draw.rect(screen, color, (x*CELL, y*CELL, CELL, CELL))

    # goal
    gx, gy = goal
    pygame.draw.rect(screen, (0, 200, 0), (gx*CELL, gy*CELL, CELL, CELL))

    # player
    pygame.draw.circle(screen, (0, 120, 255), (px*CELL + CELL//2, py*CELL + CELL//2), CELL//3)

    # HUD
    pygame.draw.rect(screen, (15, 15, 15), (0, H-60, W, 60))
    text = f"Moves: {moves} | Press R to regenerate"
    if win:
        text = f"YOU WIN! Moves: {moves} | Press R for new maze"
    screen.blit(font.render(text, True, (240,240,240)), (12, H-42))

    pygame.display.flip()

def try_move(dx, dy):
    global px, py, moves, win
    if win:
        return
    nx, ny = px + dx, py + dy
    if can_move(grid, nx, ny):
        px, py = nx, ny
        moves += 1
        if (px, py) == goal:
            win = True

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                grid, start, goal = generate_maze(COLS, ROWS)
                px, py = start
                moves = 0
                win = False
            elif event.key == pygame.K_UP:
                try_move(0, -1)
            elif event.key == pygame.K_DOWN:
                try_move(0, 1)
            elif event.key == pygame.K_LEFT:
                try_move(-1, 0)
            elif event.key == pygame.K_RIGHT:
                try_move(1, 0)

    draw()

pygame.quit()

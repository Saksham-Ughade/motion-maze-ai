import pygame, random, time, math
import cv2
import mediapipe as mp
import threading
from collections import deque
import numpy as np
import os


# =========================
# Maze (DFS + Loops)
# =========================
WALL, PATH = 1, 0

def generate_maze(cols, rows):
    if cols % 2 == 0: cols += 1
    if rows % 2 == 0: rows += 1
    grid = [[WALL for _ in range(cols)] for _ in range(rows)]

    def neighbors(cx, cy):
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
                grid[cy + dy//2][cx + dx//2] = PATH
                grid[ny][nx] = PATH
                stack.append((nx, ny))
                carved = True
                break
        if not carved:
            stack.pop()

    start = (1, 1)
    goal  = (cols-2, rows-2)
    grid[goal[1]][goal[0]] = PATH
    return grid, start, goal

def can_move(grid, x, y):
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == PATH

def add_loops(grid, loop_strength=0.10):
    rows = len(grid)
    cols = len(grid[0])
    candidates = []
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            if grid[y][x] != WALL:
                continue
            if grid[y][x-1] == PATH and grid[y][x+1] == PATH:
                candidates.append((x, y))
            elif grid[y-1][x] == PATH and grid[y+1][x] == PATH:
                candidates.append((x, y))
    random.shuffle(candidates)
    break_count = int(loop_strength * len(candidates))
    for i in range(min(break_count, len(candidates))):
        x, y = candidates[i]
        grid[y][x] = PATH

def bfs_order_and_path(grid, start, goal):
    sx, sy = start
    gx, gy = goal
    q = deque([(sx, sy)])
    parent = {(sx, sy): None}
    order = []
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        x, y = q.popleft()
        order.append((x, y))
        if (x, y) == (gx, gy):
            break
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if can_move(grid, nx, ny) and (nx, ny) not in parent:
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))

    if (gx, gy) not in parent:
        return order, []

    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return order, path

# =========================
# Finger Control (MediaPipe)
# =========================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [2, 6, 10, 14, 18]
MCP_IDS = [1, 5, 9, 13, 17]

def count_fingers(lm, handedness_label: str):
    fingers = 0

    # Optional thumb
    if handedness_label == "Right":
        if (lm[TIP_IDS[0]].x - lm[PIP_IDS[0]].x) > 0.04:
            fingers += 1
    else:
        if (lm[PIP_IDS[0]].x - lm[TIP_IDS[0]].x) > 0.04:
            fingers += 1

    for i in range(1, 5):
        tip = lm[TIP_IDS[i]]
        pip = lm[PIP_IDS[i]]
        mcp = lm[MCP_IDS[i]]
        is_above = (tip.y < pip.y) and (tip.y < mcp.y)
        strong_enough = (pip.y - tip.y) > 0.03
        if is_above and strong_enough:
            fingers += 1

    return fingers

def fingers_to_cmd(fcnt: int):
    if fcnt == 1: return "UP"
    if fcnt == 2: return "DOWN"
    if fcnt == 3: return "LEFT"
    if fcnt == 4: return "RIGHT"
    if fcnt == 0: return "STOP"
    return None

latest_cmd = "STOP"
cmd_lock = threading.Lock()

def camera_loop():
    global latest_cmd
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        cmd = "STOP"
        if res.multi_hand_landmarks:
            hand_lm = res.multi_hand_landmarks[0]
            lm = hand_lm.landmark

            handedness_label = "Right"
            if res.multi_handedness:
                handedness_label = res.multi_handedness[0].classification[0].label

            fcnt = count_fingers(lm, handedness_label)
            maybe = fingers_to_cmd(fcnt)
            if maybe:
                cmd = maybe

            cv2.putText(frame, f"Fingers:{fcnt} Cmd:{cmd}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        with cmd_lock:
            latest_cmd = cmd

        cv2.imshow("Finger Control (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# UI helpers + Glow line
# =========================
def draw_centered_text(surf, text, font, y, color=(240,240,240)):
    r = font.render(text, True, color)
    rect = r.get_rect(center=(surf.get_width()//2, y))
    surf.blit(r, rect)

def draw_box(surf, rect, fill=(18,18,18), border=(60,60,60), radius=14, border_w=2):
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, border_w, border_radius=radius)

def glow_line(screen, pts, base_color=(255, 215, 0), glow_color=(120, 220, 255), base_w=6, pulse=0.0):
    if len(pts) < 2:
        return
    w, h = screen.get_width(), screen.get_height()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)

    p = 0.5 + 0.5*math.sin(pulse * 4.0)
    w1 = int(base_w * 3.4 + p * 3)
    w2 = int(base_w * 2.1 + p * 2)
    w3 = int(base_w * 1.4 + p * 1)

    pygame.draw.lines(overlay, (*glow_color, 80), False, pts, w1)
    pygame.draw.lines(overlay, (*glow_color, 120), False, pts, w2)
    pygame.draw.lines(overlay, (*glow_color, 160), False, pts, w3)
    pygame.draw.lines(overlay, (*base_color, 140), False, pts, int(base_w*1.7))

    pygame.draw.lines(screen, base_color, False, pts, base_w)
    screen.blit(overlay, (0, 0))

# =========================
# Music + Win melody (generated)
# =========================
def make_music_loop(seconds=6.0, sr=44100, vol=0.18):
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)

    # simple lo-fi chord-ish pad: two sines + gentle tremolo
    base = 110.0
    freqs = [base*2, base*2.5, base*3]  # not perfect harmony, but nice vibe
    wave = np.zeros_like(t)
    for f in freqs:
        wave += np.sin(2*np.pi*f*t) * 0.5
    wave += np.sin(2*np.pi*(base*1.25)*t) * 0.3

    trem = 0.7 + 0.3*np.sin(2*np.pi*0.6*t)
    wave = wave * trem

    # soft limiter
    wave = wave / max(1e-9, np.max(np.abs(wave)))
    wave = (wave * (32767*vol)).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

def make_tone(freq=440, ms=120, vol=0.35, sr=44100):
    t = np.linspace(0, ms/1000.0, int(sr*ms/1000.0), endpoint=False)
    wave = (np.sin(2*np.pi*freq*t) * (32767*vol)).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

def play_win_melody():
    for f, ms in [(523,120),(659,120),(784,140),(1047,220)]:
        make_tone(f, ms).play()
        pygame.time.delay(int(ms*0.9))

# =========================
# Fireworks / Confetti (WIN)
# =========================
class Confetti:
    def __init__(self, x, y):
        ang = random.uniform(0, 2*math.pi)
        spd = random.uniform(240, 760)
        self.x = x
        self.y = y
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd - random.uniform(200, 520)
        self.life = random.uniform(1.0, 1.9)
        self.t = 0.0
        self.size = random.randint(3, 9)
        self.col = (random.randint(120,255), random.randint(120,255), random.randint(120,255))

    def update(self, dt):
        self.t += dt
        self.vy += 980 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def alive(self):
        return self.t < self.life

    def draw(self, surf):
        a = max(0, 255 - int((self.t/self.life)*255))
        s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        s.fill((*self.col, a))
        surf.blit(s, (int(self.x), int(self.y)))

def spawn_burst(particles, w, h):
    x = random.randint(w//6, 5*w//6)
    y = random.randint(h//8, h//2)
    for _ in range(140):
        particles.append(Confetti(x, y))

# =========================
# Player trail particles
# =========================
class TrailParticle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.vx = random.uniform(-40, 40)
        self.vy = random.uniform(-40, 40)
        self.life = random.uniform(0.25, 0.45)
        self.t = 0.0

    def update(self, dt):
        self.t += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 80 * dt  # tiny gravity

    def alive(self):
        return self.t < self.life

    def draw(self, surf):
        a = max(0, 200 - int((self.t/self.life)*200))
        s = pygame.Surface((self.r*2+2, self.r*2+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (120, 220, 255, a), (self.r+1, self.r+1), self.r)
        surf.blit(s, (int(self.x - self.r), int(self.y - self.r)))

# =========================
# Texture: cache maze to surface
# =========================
def build_maze_surface(grid, CELL, top_h):
    rows = len(grid)
    cols = len(grid[0])
    W = cols * CELL
    H = rows * CELL + top_h

    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    surf.fill((12, 12, 16))

    # subtle grid line color
    grid_line = (20, 20, 28)

    # per-tile variation for path brightness
    var = [[random.randint(-8, 8) for _ in range(cols)] for _ in range(rows)]

    for y in range(rows):
        for x in range(cols):
            r = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            if grid[y][x] == WALL:
                base = (28, 28, 38)
                inner = (22, 22, 30)
                pygame.draw.rect(surf, base, r)
                inset = r.inflate(-max(2, CELL//10), -max(2, CELL//10))
                pygame.draw.rect(surf, inner, inset, border_radius=max(0, CELL//6))
            else:
                v = var[y][x]
                c = (230+v, 230+v, 238+v)
                pygame.draw.rect(surf, c, r)

            # grid lines (thin)
            pygame.draw.rect(surf, grid_line, r, 1)

    return surf

# =========================
# Pygame Game
# =========================
pygame.init()
pygame.mixer.init()
# ---- Background Music (file) ----
music_path = os.path.join("assets", "bgm.mp3")  # ya bgm.mp3

pygame.mixer.music.load(music_path)
pygame.mixer.music.set_volume(0.35)
pygame.mixer.music.play(-1)


font    = pygame.font.SysFont(None, 28)
font_s  = pygame.font.SysFont(None, 22)
bigfont = pygame.font.SysFont(None, 56)
huge    = pygame.font.SysFont(None, 80)

# Levels: (cols, rows, cell, loop_strength, move_cooldown)
LEVELS = {
    "EASY":   (21, 21, 30, 0.08, 0.17),
    "MEDIUM": (31, 31, 24, 0.12, 0.15),
    "HARD":   (41, 41, 18, 0.15, 0.13),
}

TOP_HUD = 90

def build_level(level_name):
    cols, rows, cell, loop_strength, cooldown = LEVELS[level_name]
    grid, start, goal = generate_maze(cols, rows)
    add_loops(grid, loop_strength=loop_strength)
    return grid, start, goal, cell, cooldown

def make_screen(cell, grid):
    W = len(grid[0]) * cell
    H = len(grid) * cell + TOP_HUD
    return pygame.display.set_mode((W, H)), W, H

# Start camera thread
t = threading.Thread(target=camera_loop, daemon=True)
t.start()

# Background music loop
music = make_music_loop(seconds=6.0, vol=0.18)
music.play(loops=-1)

state = "MENU"  # MENU, PLAY, WIN
level = "EASY"

grid, start, goal, CELL, move_cooldown = build_level(level)
screen, W, H = make_screen(CELL, grid)
pygame.display.set_caption("Motion Maze Pro (Polished)")

maze_surface = build_maze_surface(grid, CELL, TOP_HUD)

# Player logical cell + smooth render position
px, py = start              # logical cell
fx = px * CELL + CELL//2    # render x center (pixels)
fy = py * CELL + CELL//2

# Movement tween state
moving = False
from_cell = (px, py)
to_cell = (px, py)
move_t = 0.0
move_duration = 0.14  # smooth slide duration per step (feel free: 0.10-0.20)

moves = 0

# Name input
player_name = "IOT CCA1"
name_active = True

# WIN FX
confetti = []
 
fx_timer = 0.0
fx_interval = 0.18
win_time = 0.0
played_win_sound = False

# particle trail
trail = []

# BFS animation: OFF / WAVE / RUNNER / FINAL
hint_mode = "OFF"
bfs_order = []
bfs_path = []
bfs_i = 0
bfs_timer = 0.0
bfs_step_time = 0.008
runner_i = 0
runner_timer = 0.0
runner_step_time = 0.05
final_pulse_t = 0.0

clock = pygame.time.Clock()

def reset_play(selected_level):
    global level, grid, start, goal, CELL, move_cooldown, screen, W, H, maze_surface
    global px, py, fx, fy, moving, from_cell, to_cell, move_t
    global moves, state
    global confetti, fx_timer, win_time, played_win_sound
    global hint_mode, bfs_order, bfs_path, bfs_i, bfs_timer, runner_i, runner_timer, final_pulse_t
    global trail, move_duration

    level = selected_level
    grid, start, goal, CELL, move_cooldown = build_level(level)
    screen, W, H = make_screen(CELL, grid)
    maze_surface = build_maze_surface(grid, CELL, TOP_HUD)

    px, py = start
    fx = px * CELL + CELL//2
    fy = py * CELL + CELL//2

    moving = False
    from_cell = (px, py)
    to_cell = (px, py)
    move_t = 0.0
    move_duration = max(0.10, min(0.18, move_cooldown))  # keep smooth

    moves = 0
    state = "PLAY"

    confetti = []
    fx_timer = 0.0
    win_time = 0.0
    played_win_sound = False

    trail = []

    hint_mode = "OFF"
    bfs_order = []
    bfs_path = []
    bfs_i = 0
    bfs_timer = 0.0
    runner_i = 0
    runner_timer = 0.0
    final_pulse_t = 0.0

def ease_out_quad(t):
    return 1 - (1 - t) * (1 - t)

def start_move(dx, dy):
    """Start smooth movement from current cell to target cell if possible."""
    global moving, from_cell, to_cell, move_t, px, py, moves, state

    if state != "PLAY" or moving:
        return

    nx, ny = px + dx, py + dy
    if not can_move(grid, nx, ny):
        return

    from_cell = (px, py)
    to_cell = (nx, ny)
    moving = True
    move_t = 0.0

    # logical update happens immediately so BFS start etc uses new cell after move completes?
    # We'll update px/py at the end of tween to keep consistent.
    moves += 1

def finish_move():
    global moving, px, py, state
    moving = False
    px, py = to_cell
    if (px, py) == goal:
        state = "WIN"

def update_movement(dt):
    global move_t, fx, fy
    if not moving:
        fx = px * CELL + CELL//2
        fy = py * CELL + CELL//2
        return

    move_t += dt
    t = min(1.0, move_t / move_duration)
    k = ease_out_quad(t)

    x0, y0 = from_cell
    x1, y1 = to_cell
    cx = (x0 + (x1 - x0) * k) * CELL + CELL//2
    cy = (y0 + (y1 - y0) * k) * CELL + CELL//2
    fx, fy = cx, cy

    # spawn trail during move (looks cool)
    if random.random() < 0.65:
        trail.append(TrailParticle(fx, fy, r=max(2, CELL//10)))

    if t >= 1.0:
        finish_move()

def draw_menu():
    screen.fill((8, 8, 12))
    draw_centered_text(screen, "MOTION MAZE", huge, 85, (245,245,245))
    draw_centered_text(screen, "Polished: Smooth Move + Music + Textures + Trails", font, 130, (200,200,210))

    card = pygame.Rect(W//2 - 340, 170, 680, 365)
    draw_box(screen, card, fill=(14,14,20), border=(70,70,95), radius=18)

    y = card.y + 35
    draw_centered_text(screen, "HOW TO PLAY", bigfont, y, (255, 235, 170))
    y += 55

    lines = [
        "Finger Control:",
        "1=UP  2=DOWN  3=LEFT  4=RIGHT  0=STOP",
        "H = BFS animation (wave + runner + neon path)",
        "Reach the GREEN goal block to win.",
        "",
        "Select Level:  [1] EASY   [2] MEDIUM   [3] HARD",
    ]
    for ln in lines:
        draw_centered_text(screen, ln, font, y, (230,230,235))
        y += 32

    y += 8
    draw_centered_text(screen, "Your Name (Winner name):", font, y, (200,200,210))
    y += 42
    box = pygame.Rect(W//2 - 220, y-18, 440, 46)
    draw_box(screen, box, fill=(10,10,12), border=(255,235,170) if name_active else (70,70,95), radius=12)
    name_show = player_name if player_name else "_"
    draw_centered_text(screen, name_show, bigfont, y+5, (245,245,245))

    draw_centered_text(screen, "Press ENTER to start", bigfont, card.bottom + 55, (140, 220, 255))

def draw_play(dt):
    global final_pulse_t

    # base static maze texture
    screen.blit(maze_surface, (0, 0))

    # BFS wave overlay (animated)
    if hint_mode in ["WAVE", "RUNNER", "FINAL"] and bfs_order:
        upto = min(bfs_i, len(bfs_order))
        for (x, y) in bfs_order[:upto]:
            pygame.draw.rect(screen, (120, 190, 255), (x*CELL+11, y*CELL+11, CELL-22, CELL-22))

    # FINAL neon path
    if hint_mode == "FINAL" and bfs_path and len(bfs_path) >= 2:
        final_pulse_t += dt
        pts = [(x*CELL + CELL//2, y*CELL + CELL//2) for (x, y) in bfs_path]
        glow_line(
            screen,
            pts,
            base_color=(255, 215, 0),
            glow_color=(120, 220, 255),
            base_w=max(4, CELL//6),
            pulse=final_pulse_t
        )

    # Runner dot
    if hint_mode == "RUNNER" and bfs_path:
        idx = min(runner_i, len(bfs_path)-1)
        rx, ry = bfs_path[idx]
        tail_cells = bfs_path[max(0, idx-7):idx]
        for (tx, ty) in tail_cells:
            pygame.draw.circle(screen, (255, 255, 255),
                               (tx*CELL + CELL//2, ty*CELL + CELL//2),
                               max(3, CELL//7))
        pygame.draw.circle(screen, (255, 240, 120),
                           (rx*CELL + CELL//2, ry*CELL + CELL//2),
                           max(6, CELL//3))

    # goal
    gx, gy = goal
    pygame.draw.rect(screen, (0, 200, 90), (gx*CELL, gy*CELL, CELL, CELL))

    # trail particles (behind player)
    overlay = pygame.Surface((W, H-TOP_HUD), pygame.SRCALPHA)
    alive_trail = []
    for p in trail:
        p.update(dt)
        if p.alive():
            p.draw(overlay)
            alive_trail.append(p)
    trail[:] = alive_trail
    screen.blit(overlay, (0, 0))

    # player (smooth position)
    pygame.draw.circle(screen, (0, 140, 255), (int(fx), int(fy)), max(6, CELL//3))
    pygame.draw.circle(screen, (180, 240, 255), (int(fx), int(fy)), max(3, CELL//6), 2)

    # HUD bar
    bar = pygame.Rect(0, H-TOP_HUD, W, TOP_HUD)
    draw_box(screen, bar, fill=(10,10,12), border=(60,60,80), radius=0, border_w=0)
    hud1 = f"Level: {level}    Moves: {moves}"
    hud2 = "Smooth Move + Trails | H: BFS | R: Restart | ESC: Quit"
    screen.blit(font.render(hud1, True, (240,240,240)), (16, H-78))
    screen.blit(font_s.render(hud2, True, (200,200,210)), (16, H-45))

def draw_win(dt):
    global fx_timer, win_time, played_win_sound, confetti

    win_time += dt
    draw_play(dt)

    fx_timer += dt
    if fx_timer >= fx_interval:
        fx_timer = 0.0
        spawn_burst(confetti, W, H-TOP_HUD)

    alive = []
    overlay = pygame.Surface((W, H-TOP_HUD), pygame.SRCALPHA)
    for p in confetti:
        p.update(dt)
        if p.alive():
            p.draw(overlay)
            alive.append(p)
    confetti = alive
    screen.blit(overlay, (0,0))

    if not played_win_sound:
        played_win_sound = True
        play_win_melody()

    banner = pygame.Rect(W//2 - 320, 35, 640, 120)
    draw_box(screen, banner, fill=(10,10,14), border=(255,235,170), radius=18)

    bounce = int(math.sin(win_time * 6.0) * 6)
    glow = 200 + int(55 * (0.5 + 0.5*math.sin(win_time*4.0)))
    title_color = (255, glow, 140)

    draw_centered_text(screen, "YOU WON!", huge, banner.y + 42 + bounce, title_color)
    draw_centered_text(screen, f"Winner: {player_name}", bigfont, banner.y + 88 + bounce, (245,245,245))
    draw_centered_text(screen, "Press R to play again | 1/2/3 change level | ESC quit", font, banner.bottom + 28, (240,240,240))

# =========================
# Main loop
# =========================
running = True
while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if state == "MENU":
                if event.key == pygame.K_1:
                    level = "EASY"
                elif event.key == pygame.K_2:
                    level = "MEDIUM"
                elif event.key == pygame.K_3:
                    level = "HARD"
                elif event.key == pygame.K_RETURN:
                    reset_play(level)

                if event.key == pygame.K_BACKSPACE:
                    player_name = player_name[:-1]
                elif event.key == pygame.K_TAB:
                    name_active = not name_active
                else:
                    if name_active:
                        ch = event.unicode
                        if ch.isalnum() or ch in [" ", "_", "-"]:
                            if len(player_name) < 14:
                                player_name += ch.upper()

            elif state in ["PLAY", "WIN"]:
                if event.key == pygame.K_1:
                    reset_play("EASY")
                elif event.key == pygame.K_2:
                    reset_play("MEDIUM")
                elif event.key == pygame.K_3:
                    reset_play("HARD")
                elif event.key == pygame.K_r:
                    reset_play(level)

                elif event.key == pygame.K_h and state == "PLAY":
                    if hint_mode in ["WAVE", "RUNNER", "FINAL"]:
                        hint_mode = "OFF"
                        bfs_order, bfs_path = [], []
                    else:
                        bfs_order, bfs_path = bfs_order_and_path(grid, (px, py), goal)
                        hint_mode = "WAVE"
                        bfs_i = 0
                        bfs_timer = 0.0
                        runner_i = 0
                        runner_timer = 0.0
                        final_pulse_t = 0.0

    if state == "MENU":
        draw_menu()
        pygame.display.flip()
        continue

    # Read camera cmd
    with cmd_lock:
        cmd = latest_cmd

    # Smooth movement: only accept new move if not moving
    if state == "PLAY" and not moving:
        if cmd == "UP":
            start_move(0, -1)
        elif cmd == "DOWN":
            start_move(0, 1)
        elif cmd == "LEFT":
            start_move(-1, 0)
        elif cmd == "RIGHT":
            start_move(1, 0)

    # Update tween position
    update_movement(dt)

    # BFS animation update (PLAY only)
    if state == "PLAY" and hint_mode == "WAVE":
        bfs_timer += dt
        while bfs_timer >= bfs_step_time:
            bfs_timer -= bfs_step_time
            bfs_i += 1
            if bfs_i >= len(bfs_order):
                hint_mode = "RUNNER"
                runner_i = 0
                runner_timer = 0.0
                break

    if state == "PLAY" and hint_mode == "RUNNER":
        runner_timer += dt
        while runner_timer >= runner_step_time:
            runner_timer -= runner_step_time
            runner_i += 1
            if runner_i >= len(bfs_path):
                hint_mode = "FINAL"
                break

    # Draw
    if state == "PLAY":
        draw_play(dt)
    else:
        draw_win(dt)

    pygame.display.flip()

pygame.quit()

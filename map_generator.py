"""
Grid-Based Maze with Alternate Paths
Maze-like but NOT unidirectional
Compatible with continuous RL driving
"""

from PIL import Image, ImageDraw
import random

# ============================
# CONFIG
# ============================

GRID_W = 7
GRID_H = 7
CELL_SIZE = 120
ROAD_WIDTH = 50

WIDTH = GRID_W * CELL_SIZE
HEIGHT = GRID_H * CELL_SIZE

LOOP_PROB = 0.25  # ← ADD ALTERNATE PATHS

# Colors
BG = (55, 55, 70)
ROAD = (245, 245, 245)
START_CLR = (100, 200, 255)
TARGET_CLR = (255, 90, 90)

random.seed(3)

# ============================
# MAZE GENERATION
# ============================

def generate_maze(w, h):
    visited = [[False]*w for _ in range(h)]
    maze = [[{"N":0,"S":0,"E":0,"W":0} for _ in range(w)] for _ in range(h)]

    def dfs(cx, cy):
        visited[cy][cx] = True
        dirs = ["N", "S", "E", "W"]
        random.shuffle(dirs)

        for d in dirs:
            nx, ny = cx, cy
            if d == "N": ny -= 1
            if d == "S": ny += 1
            if d == "E": nx += 1
            if d == "W": nx -= 1

            if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                maze[cy][cx][d] = 1
                maze[ny][nx][{"N":"S","S":"N","E":"W","W":"E"}[d]] = 1
                dfs(nx, ny)

    dfs(0, 0)

    # ============================
    # ADD LOOPS (KEY CHANGE)
    # ============================

    for y in range(h):
        for x in range(w):
            if random.random() < LOOP_PROB:
                d = random.choice(["N", "S", "E", "W"])
                nx, ny = x, y
                if d == "N": ny -= 1
                if d == "S": ny += 1
                if d == "E": nx += 1
                if d == "W": nx -= 1

                if 0 <= nx < w and 0 <= ny < h:
                    maze[y][x][d] = 1
                    maze[ny][nx][{"N":"S","S":"N","E":"W","W":"E"}[d]] = 1

    return maze

# ============================
# DRAW MAP
# ============================
def draw_round_cap(draw, x, y, r, color):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def create_maze_map():
    maze = generate_maze(GRID_W, GRID_H)

    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    junctions = []

    # draw straight roads + collect junctions
    for y in range(GRID_H):
        for x in range(GRID_W):
            cx = x * CELL_SIZE + CELL_SIZE // 2
            cy = y * CELL_SIZE + CELL_SIZE // 2
            cell = maze[y][x]

            junctions.append((cx, cy))

            if cell["N"]:
                draw.line((cx, cy, cx, cy - CELL_SIZE), fill=ROAD, width=ROAD_WIDTH)
            if cell["S"]:
                draw.line((cx, cy, cx, cy + CELL_SIZE), fill=ROAD, width=ROAD_WIDTH)
            if cell["E"]:
                draw.line((cx, cy, cx + CELL_SIZE, cy), fill=ROAD, width=ROAD_WIDTH)
            if cell["W"]:
                draw.line((cx, cy, cx - CELL_SIZE, cy), fill=ROAD, width=ROAD_WIDTH)

    # add round hubs to soften corners
    r = ROAD_WIDTH // 2
    for y in range(GRID_H):
        for x in range(GRID_W):
            cx = x * CELL_SIZE + CELL_SIZE // 2
            cy = y * CELL_SIZE + CELL_SIZE // 2
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ROAD)


    # ----------------------------
    # START & TARGETS
    # ----------------------------
    start = junctions[0]

    # def far_enough(p, others, min_dist=250):
    #     for q in others:
    #         dx = p[0] - q[0]
    #         dy = p[1] - q[1]
    #         if (dx*dx + dy*dy) ** 0.5 < min_dist:
    #             return False
    #     return True

    # targets = []
    # for p in junctions[1:]:
    #     if far_enough(p, [start] + targets):
    #         targets.append(p)
    #     if len(targets) == 3:
    #         break

    # ✅ HARDCODE all targets based on your current map
    targets = [
        (180, 300),   # Target 1 (same as current)
        (420, 60),    # Target 2 (same as current)
        (540, 420)    # Target 3 (NEW - at your marked position)
    ]


        # ----------------------------
    # TRUE ROUNDED CORNERS (STAGE 2)
    # ----------------------------

    # corner_radius = ROAD_WIDTH // 2
    # half = ROAD_WIDTH // 2
    # draw.ellipse(
    #     (cx - half, cy - half, cx + half, cy + half),
    #     fill=ROAD
    # )
    # ----------------------------
    



    draw.ellipse((start[0]-18, start[1]-18, start[0]+18, start[1]+18), fill=START_CLR)

    for t in targets:
        draw.ellipse((t[0]-18, t[1]-18, t[0]+18, t[1]+18), fill=TARGET_CLR)

  


    # ----------------------------
    # SAVE MAP METADATA (START & TARGETS)
    # ----------------------------
    
    def dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx*dx + dy*dy) ** 0.5

    # Reorder targets: closest → A1, farthest → A3
    targets = sorted(targets, key=lambda t: dist(start, t))

    import json

    metadata = {
        "start": [int(start[0]), int(start[1])],
        "targets": [[int(x), int(y)] for x, y in targets]
    }
    with open("map_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("📍 Map metadata saved to map_meta.json")


    img.save("simple_grid_map.png")

    print("✅ Maze map created with alternate paths")
    print("✔ Grid-based")
    print("✔ Long corridors")
    print("✔ Multiple routes")
    print("✔ Limited junctions")


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    create_maze_map()

import pygame
import math
import sys
import random
import os

WIDTH, HEIGHT = 800, 600
FPS = 60
DT = 1.0 / FPS

ROBOT_RADIUS = 15
SENSOR_RANGE = int(0.15 * WIDTH)           # ~15% of arena width
SENSOR_ANGLES = [-math.radians(30), 0, math.radians(30)]  # left, front, right

SPEED = 90.0                                # px/s  ← slower linear speed
TURN_SPEED = math.radians(60)               # rad/s ← milder turning
MAX_STEP = 2.0                               # max translational step per frame

BG = (8, 10, 14)
WALL_COL = (230, 230, 240)
SENSOR_HIT = (255, 170, 90)
SENSOR_MISS = (130, 130, 140)
ROBOT_COL = (120, 180, 255)

# Pygame init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Task 2: Proximity Sensors")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)

# Walls (outer box + randomized internal)
walls = [
    # Outer box
    ((0, 0), (WIDTH, 0)),
    ((WIDTH, 0), (WIDTH, HEIGHT)),
    ((WIDTH, HEIGHT), (0, HEIGHT)),
    ((0, HEIGHT), (0, 0)),

    # Internal fixed obstacles
    ((200, 350), (200, 120)),
    ((600, 200), (600, 450)),
    ((600, 200), (470, 200)),

    ((0, 180), (200, 180)),            
    ((WIDTH - 200, 450), (WIDTH, 450)),
    ((400, 0), (400, 100)),        
    ((250, HEIGHT - 100), (250, HEIGHT))
]

"""
# Add random internal line obstacles each run
NUM_INTERNAL_WALLS = random.randint(2, 5)
for _ in range(NUM_INTERNAL_WALLS):
    x1 = random.randint(100, WIDTH - 100)
    y1 = random.randint(100, HEIGHT - 100)
    length = random.randint(120, 300)
    angle = random.uniform(0, math.pi)  # 0–180° orientation
    x2 = int(x1 + length * math.cos(angle))
    y2 = int(y1 + length * math.sin(angle))
    # Clamp endpoints inside the screen
    x2 = max(0, min(WIDTH, x2))
    y2 = max(0, min(HEIGHT, y2))
    walls.append(((x1, y1), (x2, y2)))
"""

def seg_intersect(p1, p2, p3, p4):
    """Return intersection point of segments p1-p2 and p3-p4, else None."""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    if (min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9 and
        min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9 and
        min(x3, x4) - 1e-9 <= px <= max(x3, x4) + 1e-9 and
        min(y3, y4) - 1e-9 <= py <= max(y3, y4) + 1e-9):
        return (px, py)
    return None

def dot(a, b): 
    return a[0]*b[0] + a[1]*b[1]

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def seg_seg_dist2(a, b, c, d):
    """
    Squared distance between segments a-b and c-d (closest points method).
    Used for swept-circle collision (capsule vs. segment).
    """
    ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
    u = (bx-ax, by-ay)
    v = (dx-cx, dy-cy)
    w0 = (ax-cx, ay-cy)

    uu = dot(u, u)
    vv = dot(v, v)
    uv = dot(u, v)
    uw = dot(u, w0)
    vw = dot(v, w0)
    D = uu*vv - uv*uv

    s = 0.0
    if D != 0.0:
        s = clamp((uv*vw - vv*uw) / D, 0.0, 1.0)

    # compute t for this s, clamp to [0,1] and adjust s if needed
    t_nom = uv*s + vw
    if t_nom <= 0.0:
        t = 0.0
        s = clamp(-uw/uu if uu>0 else 0.0, 0.0, 1.0)
    elif t_nom >= vv:
        t = 1.0
        s = clamp((uv - uw)/uu if uu>0 else 0.0, 0.0, 1.0)
    else:
        t = t_nom / vv

    pc = (ax + s*u[0], ay + s*u[1])
    qc = (cx + t*v[0], cy + t*v[1])
    dx_, dy_ = pc[0] - qc[0], pc[1] - qc[1]
    return dx_*dx_ + dy_*dy_

def swept_circle_hits_walls(old, new, wall_list, radius, eps=1.5):
    """
    True if the moving circle (center moves old->new, radius) collides any wall segment.
    Equivalent to: distance between segment old-new and each wall segment <= radius + eps.
    """
    thr2 = (radius + eps) ** 2
    for (p, q) in wall_list:
        if seg_seg_dist2(old, new, p, q) <= thr2:
            return True
    return False

class Robot:
    def __init__(self, x, y, color=ROBOT_COL):
        self.x = x
        self.y = y
        self.heading = random.uniform(0, 2*math.pi)
        self.color = color
        # self.trail = []

    @property
    def pos(self):
        return (self.x, self.y)

    def sensor_dirs(self):
        return [self.heading + ang for ang in SENSOR_ANGLES]

    def sense(self, wall_list):
        """
        Cast 3 rays and return:
          - distances (None if nothing within range),
          - normalized readings (0..1),
          - hit points/end points for drawing.
        """
        dists, vals, points = [], [], []
        for ang in self.sensor_dirs():
            sx, sy = self.x, self.y
            ex = sx + SENSOR_RANGE * math.cos(ang)
            ey = sy + SENSOR_RANGE * math.sin(ang)

            closest_d = None
            closest_pt = (ex, ey)

            for (p1, p2) in wall_list:
                inter = seg_intersect((sx, sy), (ex, ey), p1, p2)
                if inter:
                    d = math.hypot(inter[0]-sx, inter[1]-sy)
                    if closest_d is None or d < closest_d:
                        closest_d = d
                        closest_pt = inter

            dists.append(closest_d)
            if closest_d is None or closest_d > SENSOR_RANGE:
                vals.append(0.0)
            else:
                vals.append(max(0.0, 1.0 - closest_d / SENSOR_RANGE))
            points.append(closest_pt)
        return dists, vals, points

    def rule_controller(self, sL, sC, sR):
        """
        Simple if-then rules:
          - strong front -> reduce speed + turn away from stronger side
          - side high -> turn opposite
          - else slight random exploration curvature
        Returns (v, omega).
        """
        TH_STRONG = 0.6
        TH_SIDE = 0.35

        v = SPEED
        if sC > TH_STRONG:
            v = 0.4 * SPEED
            omega = +TURN_SPEED if sL >= sR else -TURN_SPEED
        elif sL > TH_SIDE and sR > TH_SIDE:
            v = 0.5 * SPEED
            omega = +0.6 * TURN_SPEED if sL >= sR else -0.6 * TURN_SPEED
        elif sL > TH_SIDE:
            omega = -0.7 * TURN_SPEED   # obstacle on left -> turn right
        elif sR > TH_SIDE:
            omega = +0.7 * TURN_SPEED   # obstacle on right -> turn left
        else:
            omega = (random.random() - 0.5) * 0.25 * TURN_SPEED
        return v, omega

    def step(self, wall_list):

        dists, vals, points = self.sense(wall_list)
        sL, sC, sR = vals

        v, omega = self.rule_controller(sL, sC, sR)

        self.heading = (self.heading + omega * DT) % (2*math.pi)

        dx = v * math.cos(self.heading) * DT
        dy = v * math.sin(self.heading) * DT

        L = math.hypot(dx, dy)
        if L > MAX_STEP:
            scale = MAX_STEP / L
            dx *= scale; dy *= scale

        old = (self.x, self.y)
        proposed = (self.x + dx, self.y + dy)
        """
        if not segment_hits_walls(old, proposed, wall_list):
            self.x, self.y = proposed
        else:
            # slide along walls using axis-aligned attempts
            alt = (self.x + dx, self.y)
            if not segment_hits_walls(old, alt, wall_list):
                self.x += dx
            alt = (self.x, self.y + dy)
            if not segment_hits_walls((self.x, self.y), alt, wall_list):
                self.y += dy
            # else blocked; stay put
        """

        if not swept_circle_hits_walls(old, proposed, wall_list, ROBOT_RADIUS):
            self.x, self.y = proposed
        else:
            # try axis-wise sliding with same robust check
            alt = (self.x + dx, self.y)
            if not swept_circle_hits_walls((self.x, self.y), alt, wall_list, ROBOT_RADIUS):
                self.x += dx
            alt = (self.x, self.y + dy)
            if not swept_circle_hits_walls((self.x, self.y), alt, wall_list, ROBOT_RADIUS):
                self.y += dy
            # else blocked; stay put

        # outer walls also prevent leaving
        self.x = max(ROBOT_RADIUS, min(WIDTH - ROBOT_RADIUS, self.x))
        self.y = max(ROBOT_RADIUS, min(HEIGHT - ROBOT_RADIUS, self.y))

        # Trail disabled
        # self.trail.append((self.x, self.y))
        # if len(self.trail) > 1600:
        #     self.trail.pop(0)

        return dists, vals, points  

    def draw(self, surf, dists, vals, points):
        # Trail disabled
        # if len(self.trail) > 1:
        #     pygame.draw.lines(surf, (120, 210, 240), False, self.trail, 2)

        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), ROBOT_RADIUS, 2)
        hx = self.x + (ROBOT_RADIUS + 10) * math.cos(self.heading)
        hy = self.y + (ROBOT_RADIUS + 10) * math.sin(self.heading)
        pygame.draw.line(surf, (230, 230, 240), (self.x, self.y), (hx, hy), 2)

        # Sensors (draw after walls so they’re visible)
        for ang, d, pt in zip(self.sensor_dirs(), dists, points):
            col = SENSOR_HIT if (d is not None and d <= SENSOR_RANGE) else SENSOR_MISS
            end = (self.x + SENSOR_RANGE * math.cos(ang), self.y + SENSOR_RANGE * math.sin(ang))
            if d is not None and d <= SENSOR_RANGE:
                pygame.draw.line(surf, col, (self.x, self.y), pt, 2)
                pygame.draw.circle(surf, col, (int(pt[0]), int(pt[1])), 4)
            else:
                pygame.draw.line(surf, col, (self.x, self.y), end, 1)

robot = Robot(WIDTH / 2, HEIGHT / 2)

trajectory_points = []

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # Update world
    dists, vals, pts = robot.step(walls)
    trajectory_points.append(robot.pos)

    # Draw
    screen.fill(BG)
    for w in walls:
        pygame.draw.line(screen, WALL_COL, w[0], w[1], 3)
    robot.draw(screen, dists, vals, pts)

    # HUD
    sL, sC, sR = vals
    hud = font.render(f"sL={sL:.2f}  sC={sC:.2f}  sR={sR:.2f}  range={SENSOR_RANGE}px", True, (230, 230, 240))
    screen.blit(hud, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)

try:
    import matplotlib.pyplot as plt
    if trajectory_points:
        xs = [p[0] for p in trajectory_points]
        ys = [p[1] for p in trajectory_points]
        fig, ax = plt.subplots(figsize=(7, 5))
        for (x1, y1), (x2, y2) in walls:
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5)
        ax.plot(xs, ys, 'b-', linewidth=2, label='Path')
        ax.scatter(xs[0], ys[0], c='g', s=60, label='Start')
        ax.scatter(xs[-1], ys[-1], c='r', s=60, label='End')
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(HEIGHT, 0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Robot Trajectory (Task 2)')
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        plt.tight_layout()

        output_folder = "Lab Classes/Assignment 1/output"  
        os.makedirs(output_folder, exist_ok=True)

        filename = "trajectory_plot.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)

        plt.close(fig)
except Exception as e:
    print("Could not save trajectory plot:", e)

pygame.quit()
sys.exit()

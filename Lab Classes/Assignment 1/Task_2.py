import pygame
import math
import sys
import random

# --- PARAMETERS ---
WIDTH, HEIGHT = 800, 600
FPS = 60
ROBOT_RADIUS = 15
SENSOR_RANGE = int(0.15 * WIDTH)  # 15% of map width
SENSOR_ANGLES = [-math.radians(30), 0, math.radians(30)]  # left, front, right
SENSOR_COLOR = (0, 255, 0, 120)  # semi-transparent green

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Proximity Sensors - Wall Avoidance")
clock = pygame.time.Clock()

# --- WALLS (outer + internal line obstacles) ---
walls = [
    # Outer walls (arena boundaries)
    ((0, 0), (WIDTH, 0)),
    ((WIDTH, 0), (WIDTH, HEIGHT)),
    ((WIDTH, HEIGHT), (0, HEIGHT)),
    ((0, HEIGHT), (0, 0)),

    # Internal obstacles as lines
    ((150, 200), (650, 200)),
    ((400, 100), (400, 500)),
    ((250, 400), (550, 400))
]

# --- UTILITY FUNCTIONS ---
def line_intersection(p1, p2, p3, p4):
    """Return intersection point of line segments p1-p2 and p3-p4, if any."""
    x1, y1, x2, y2 = *p1, *p2
    x3, y3, x4, y4 = *p3, *p4
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    # Check if within segments
    if (min(x1,x2) <= px <= max(x1,x2) and
        min(y1,y2) <= py <= max(y1,y2) and
        min(x3,x4) <= px <= max(x3,x4) and
        min(y3,y4) <= py <= max(y3,y4)):
        return px, py
    return None

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# --- ROBOT CLASS ---
class Robot:
    def __init__(self, x, y, color=(100,100,255)):
        self.x = x
        self.y = y
        self.heading = random.uniform(0, 2*math.pi)
        self.speed = 2
        self.turn_speed = math.radians(3)
        self.color = color

    def sense(self, walls):
        """Cast 3 sensors and return readings (0..1)."""
        readings = []
        for angle_offset in SENSOR_ANGLES:
            angle = self.heading + angle_offset
            end_x = self.x + SENSOR_RANGE * math.cos(angle)
            end_y = self.y + SENSOR_RANGE * math.sin(angle)

            closest_dist = SENSOR_RANGE
            closest_point = (end_x, end_y)

            for (p1, p2) in walls:
                inter = line_intersection((self.x, self.y), (end_x, end_y), p1, p2)
                if inter:
                    d = distance((self.x, self.y), inter)
                    if d < closest_dist:
                        closest_dist = d
                        closest_point = inter

            # Draw sensor ray
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(s, SENSOR_COLOR, (self.x, self.y), closest_point, 3)
            screen.blit(s, (0, 0))

            readings.append(max(0, 1 - closest_dist / SENSOR_RANGE))
        return readings

    def control(self, sensors):
        left, front, right = sensors

        # Smooth exploration rules
        turn = 0
        if front > 0.5:
            turn = self.turn_speed * 2 * (-1 if left > right else 1)
        elif left > 0.3:
            turn = self.turn_speed
        elif right > 0.3:
            turn = -self.turn_speed
        else:
            # small random wiggle for exploration
            turn = math.radians(random.uniform(-1.5, 1.5))

        self.heading += turn
        self.heading %= 2*math.pi

    def move(self):
        self.x += self.speed * math.cos(self.heading)
        self.y += self.speed * math.sin(self.heading)

        # Prevent leaving arena
        self.x = max(ROBOT_RADIUS, min(WIDTH - ROBOT_RADIUS, self.x))
        self.y = max(ROBOT_RADIUS, min(HEIGHT - ROBOT_RADIUS, self.y))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), ROBOT_RADIUS)
        hx = self.x + 20 * math.cos(self.heading)
        hy = self.y + 20 * math.sin(self.heading)
        pygame.draw.line(surface, (255,255,255), (self.x, self.y), (hx, hy), 2)

# --- INIT ---
robot = Robot(100, 100)

# --- MAIN LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Update robot 
    sensors = robot.sense(walls)
    robot.control(sensors)
    robot.move()

    # Draw
    screen.fill((0,0,0))
    for w in walls:
        pygame.draw.line(screen, (255,255,255), w[0], w[1], 3)
    robot.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

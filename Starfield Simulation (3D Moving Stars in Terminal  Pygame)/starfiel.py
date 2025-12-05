# starfield.py
"""
Starfield
Features:
 - Warp tunnel (streak trails)
 - Colored stars (distance/speed-based)
 - Spaceship mode (arrow keys to steer)
 - Speed boost (hold SPACE)
 - Shooting stars (rare fast streaks)
 - Spiral/rotation motion toggle
 - Audio-sync mode using microphone FFT (optional)

Controls:
 - Arrow keys: steer (spaceship)
 - SPACE: speed boost
 - S: toggle spiral rotation
 - A: toggle audio-sync (if numpy+sounddevice installed)
 - T: toggle trails
 - C: toggle color mode
 - Q or ESC: quit
"""
import pygame
import random
import math
import time
import sys

# Optional audio imports; handled gracefully
try:
    import numpy as np
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

# ---------------- Config ----------------
WIDTH, HEIGHT = 1000, 700
NUM_STARS = 700
BASE_SPEED = 0.8          # base approach speed
BOOST_MULT = 3.5          # boost factor when holding space
TRAIL_LENGTH = 6          # how many previous positions to keep for a star (for streaks)
SHOOTING_STAR_PROB = 0.002  # per-frame probability to spawn shooting star
FPS = 60

# Audio parameters (for FFT)
AUDIO_BLOCK = 1024
AUDIO_SAMPLERATE = 44100
FFT_AVG_FRAMES = 4

# ---------------- Helpers ----------------
def clamp(v, a, b):
    return max(a, min(b, v))

def lerp(a, b, t):
    return a + (b - a) * t

def color_gradient(t):
    """Map t in [0,1] to a color from blue->white->yellow->red"""
    if t < 0:
        t = 0
    if t > 1:
        t = 1
    if t < 0.33:
        # blue -> white
        tt = t / 0.33
        return (
            int(lerp(80, 255, tt)),
            int(lerp(120, 255, tt)),
            int(lerp(200, 255, tt))
        )
    elif t < 0.66:
        # white -> yellow
        tt = (t - 0.33) / 0.33
        return (
            int(lerp(255, 255, tt)),
            int(lerp(255, 220, tt)),
            int(lerp(255, 80, tt))
        )
    else:
        # yellow -> red
        tt = (t - 0.66) / 0.34
        return (
            int(lerp(255, 255, tt)),
            int(lerp(220, 100, tt)),
            int(lerp(80, 40, tt))
        )

# ---------------- Star Classes ----------------
class Star:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.reset()

    def reset(self):
        # x,y in [-w..w], [-h..h] to give spread; z in [1, width]
        self.x = random.uniform(-self.w, self.w)
        self.y = random.uniform(-self.h, self.h)
        self.z = random.uniform(1, self.w)
        self.prev = []  # store previous projected positions for trail
        self.base_size = random.uniform(0.8, 3.2)
        self.brightness = random.uniform(0.6, 1.0)
        self.is_shooting = False

    def update(self, speed, offset_x=0, offset_y=0, rotation=0, spiral=0.0):
        # apply spiral rotation (rotate around origin)
        if rotation != 0.0:
            cosr = math.cos(rotation)
            sinr = math.sin(rotation)
            xr = self.x * cosr - self.y * sinr
            yr = self.x * sinr + self.y * cosr
            self.x, self.y = xr, yr

        # slight inward spiral scaling if desired
        if spiral != 0.0:
            self.x *= (1 - spiral * 0.0006)
            self.y *= (1 - spiral * 0.0006)

        # approach viewer by decreasing z
        self.z -= speed
        if self.z <= 0.9:
            self.reset()
            self.z = random.uniform(self.w * 0.6, self.w)
        # projection to screen
        sx = (self.x / self.z) * (self.w/2) + self.w/2 + offset_x
        sy = (self.y / self.z) * (self.h/2) + self.h/2 + offset_y
        # store previous positions for trails
        self.prev.insert(0, (sx, sy, self.z))
        if len(self.prev) > TRAIL_LENGTH:
            self.prev.pop()
        return sx, sy, self.z

class ShootingStar:
    def __init__(self, width, height, fast_speed=12.0):
        self.w = width
        self.h = height
        # spawn at random edge with direction roughly across screen
        edge = random.choice(['top','left','right'])
        if edge == 'top':
            self.x = random.uniform(-self.w/4, self.w*1.25)
            self.y = -20
            self.vx = random.uniform(-6, 6)
            self.vy = random.uniform(6, 12)
        elif edge == 'left':
            self.x = -20
            self.y = random.uniform(-self.h/4, self.h*1.25)
            self.vx = random.uniform(6, 12)
            self.vy = random.uniform(-3, 3)
        else:
            self.x = self.w + 20
            self.y = random.uniform(-self.h/4, self.h*1.25)
            self.vx = random.uniform(-12, -6)
            self.vy = random.uniform(-3, 3)
        self.life = random.randint(80, 160)
        self.age = 0
        self.length = random.uniform(40, 160)
        self.color = (255, 220, 180)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        return self.age < self.life

# ---------------- Audio FFT handling (optional) ----------------
class AudioAnalyzer:
    def __init__(self, block=AUDIO_BLOCK, samplerate=AUDIO_SAMPLERATE):
        self.block = block
        self.sr = samplerate
        self.latest_level = 0.0
        self.running = False
        self._buf = np.zeros(self.block, dtype='float32')
        try:
            self.stream = sd.InputStream(channels=1, callback=self._callback, blocksize=self.block, samplerate=self.sr)
            self.stream.start()
            self.running = True
        except Exception as e:
            print("Audio init failed:", e)
            self.running = False

    def _callback(self, indata, frames, time_info, status):
        # simple RMS or FFT energy
        data = indata[:,0].astype('float32')
        # compute RMS
        rms = np.sqrt(np.mean(data*data))
        # compute FFT energy in low-mid band (e.g., 20-2000Hz)
        try:
            fft = np.fft.rfft(data)
            mags = np.abs(fft)
            # take average magnitude as activity
            val = np.mean(mags)
            # normalize roughly (heuristic)
            self.latest_level = clamp(val * 10.0, 0.0, 1.5)
        except Exception:
            self.latest_level = clamp(rms * 10.0, 0.0, 1.5)

    def get_level(self):
        return float(self.latest_level)

    def close(self):
        if self.running:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

# ---------------- Main App ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Starfield")
    clock = pygame.time.Clock()

    # Surface for trails with alpha (so old trails fade naturally)
    trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    stars = [Star(WIDTH, HEIGHT) for _ in range(NUM_STARS)]
    shooting = []

    # State
    offset_x = 0.0
    offset_y = 0.0
    steer_x = 0.0
    steer_y = 0.0
    rotation = 0.0
    spiral = 0.0
    speed_base = BASE_SPEED
    trails_enabled = True
    color_mode = True
    spiral_enabled = False
    audio_enabled = False and AUDIO_AVAILABLE
    audio_analyzer = None
    if AUDIO_AVAILABLE:
        try:
            audio_analyzer = AudioAnalyzer()
        except Exception:
            audio_analyzer = None
    if audio_analyzer:
        audio_enabled = True

    running = True
    frame_count = 0
    seed = random.random()

    while running:
        dt = clock.tick(FPS) / 1000.0
        frame_count += 1

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    spiral_enabled = not spiral_enabled
                elif event.key == pygame.K_t:
                    trails_enabled = not trails_enabled
                    if not trails_enabled:
                        trail_surf.fill((0,0,0,0))
                elif event.key == pygame.K_c:
                    color_mode = not color_mode
                elif event.key == pygame.K_a:
                    # toggle audio-sync (only if available)
                    if AUDIO_AVAILABLE and audio_analyzer:
                        audio_enabled = not audio_enabled
                        print("Audio sync:", audio_enabled)
                    else:
                        print("Audio not available (requires sounddevice + numpy).")
            # handle mouse etc if needed

        keys = pygame.key.get_pressed()
        # steering: arrow keys
        steer_x = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 250.0 * dt + steer_x * (1 - 8*dt)
        steer_y = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * 150.0 * dt + steer_y * (1 - 8*dt)
        # speed multiplier (space)
        boosting = keys[pygame.K_SPACE]
        speed_mult = BOOST_MULT if boosting else 1.0

        # small rotational sway if spiral enabled
        if spiral_enabled:
            rotation += 0.8 * dt
            spiral = 0.5
        else:
            rotation *= 0.98
            spiral *= 0.96

        # audio influence
        audio_level = 0.0
        if audio_enabled and audio_analyzer:
            audio_level = audio_analyzer.get_level()
            # amplify a bit
            audio_level = clamp(audio_level, 0.0, 1.5)
            # increase speed_base slightly based on audio
            speed_mult *= (1.0 + audio_level * 0.9)

        # background
        screen.fill((4, 4, 12))

        # fade trails slightly onto trail_surf
        if trails_enabled:
            # draw a translucent rect to slowly fade previous trails
            trail_surf.fill((0,0,0,28), special_flags=pygame.BLEND_RGBA_SUB)
        else:
            trail_surf.fill((0,0,0,0))

        # update & draw stars
        for star in stars:
            # variable speed per star: nearer stars should move faster visually
            s = (WIDTH - star.z) / WIDTH
            # speed uses base speed, multiplier and per-star small variation and audio
            speed = (speed_base * (1.0 + s*4.0)) * speed_mult
            # small per-star flicker
            if audio_enabled:
                speed *= (1.0 + audio_level * 0.8 * random.uniform(0.8, 1.2))
            sx, sy, sz = star.update(speed, offset_x=steer_x, offset_y=steer_y, rotation=rotation*0.4, spiral=spiral*0.6)

            # if projected inside screen, draw
            if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
                depth_norm = clamp(1.0 - (sz / WIDTH), 0.0, 1.0)  # near -> 1.0
                # color based on depth_norm and brightness
                if color_mode:
                    col = color_gradient(depth_norm * star.brightness)
                else:
                    val = int(lerp(60, 255, depth_norm * star.brightness))
                    col = (val, val, val)

                # size scaled by depth
                size = max(1, int(star.base_size * (1.0 + depth_norm * 3.5)))
                # draw star as small circle onto trail_surf for smoother streaks
                if trails_enabled:
                    # draw small circle and also draw small line between current and previous pos
                    pygame.draw.circle(trail_surf, col + (200,), (int(sx), int(sy)), size)
                    # draw line to previous projected positions for trailing effect
                    prevs = star.prev
                    if len(prevs) > 1:
                        alpha_step = int(180 / max(1, len(prevs)))
                        for i in range(1, len(prevs)):
                            px, py, pz = prevs[i]
                            fade = max(10, 200 - i * alpha_step)
                            pygame.draw.line(trail_surf, col + (fade,), (int(px), int(py)), (int(sx), int(sy)), max(1, size - i//2))
                else:
                    pygame.draw.circle(screen, col, (int(sx), int(sy)), size)

        # occasionally spawn shooting stars
        if random.random() < SHOOTING_STAR_PROB:
            shooting.append(ShootingStar(WIDTH, HEIGHT))

        # update & draw shooting stars (bright streaks)
        for s in shooting[:]:
            alive = s.update()
            if not alive:
                shooting.remove(s)
            else:
                # draw long streak by drawing line of length proportional to s.length
                end_x = s.x - s.vx * 3
                end_y = s.y - s.vy * 3
                pygame.draw.line(screen, s.color, (s.x, s.y), (end_x, end_y), 2)
                # add bright head
                pygame.draw.circle(screen, (255,255,255), (int(s.x), int(s.y)), 3)

        # composite trails onto screen
        if trails_enabled:
            screen.blit(trail_surf, (0,0))

        # HUD text
        font = pygame.font.SysFont("Consolas", 16)
        hud = [
            f"Stars: {len(stars)}  Trails: {'ON' if trails_enabled else 'OFF'}  Color: {'ON' if color_mode else 'OFF'}",
            f"Spiral: {'ON' if spiral_enabled else 'OFF'}  Audio: {'ON' if audio_enabled else 'OFF'}  Boost: {'ON' if boosting else 'OFF'}",
            "Controls: Arrow keys=steer  SPACE=boost  S=spiral  A=audio  T=trails  C=color  Q=quit"
        ]
        for i, line in enumerate(hud):
            txt = font.render(line, True, (200,200,200))
            screen.blit(txt, (8, 8 + i*20))

        pygame.display.flip()

    # cleanup
    if audio_analyzer:
        audio_analyzer.close()
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()

"""CPU SDL renderer with procedural artwork."""

import math
import os
import numpy as np
from .kernels import body_count, snake_radius, food_radius
from .colors import PALETTE, FOOD_COLORS


def capture_motion(core):
    """Copy only moving geometry; static arrays remain shared and are never written."""
    s = core.state
    body = s.body.copy()
    for i in range(core.config.num_snakes):
        n = body_count(s, 0, i, core.config)
        body[0, i, n:] = body[0, i, n - 1]
    return s._replace(
        head=s.head.copy(),
        body=body,
        angle=s.angle.copy(),
        mass=s.mass.copy(),
        speed=s.speed.copy(),
        alive=s.alive.copy(),
        episode=s.episode.copy(),
        ticks=s.ticks.copy(),
    )


def interpolate_motion(previous, current, alpha):
    """Viewer-only interpolation. Respawns and deaths use their current pose."""
    valid = (
        (previous.episode == current.episode)
        & (current.ticks == previous.ticks + 1)[:, None]
        & previous.alive
        & current.alive
    )
    blend = np.clip(alpha, 0, 1)

    def mix(before, after):
        mask = valid.reshape(valid.shape + (1,) * (after.ndim - valid.ndim))
        return np.where(mask, before + (after - before) * blend, after)

    angle_delta = (current.angle - previous.angle + math.pi) % (2 * math.pi) - math.pi
    return current._replace(
        head=mix(previous.head, current.head),
        body=mix(previous.body, current.body),
        angle=np.where(valid, previous.angle + angle_delta * blend, current.angle),
        mass=mix(previous.mass, current.mass),
        speed=mix(previous.speed, current.speed),
    )


class Renderer:
    def __init__(self, width=1280, height=720, mode="rgb_array", fps=20):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.font.init()
        self.width, self.height = width, height
        self.mode, self.fps = mode, fps
        self.surface = pygame.Surface((width, height))
        self.window = None
        self.closed = False
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 17)
        self.title = pygame.font.Font(None, 40)
        self._sprites = {}
        self._glows = {}
        self.commands = []
        self.mouse_active = False
        if mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            pygame.display.set_caption("Slither")

    def poll(self):
        if self.mode != "human" or self.closed:
            return
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
            ):
                self.closed = True
            elif event.type == pg.VIDEORESIZE:
                self.width = max(320, event.w)
                self.height = max(240, event.h)
                self.window = pg.display.set_mode(
                    (self.width, self.height), pg.RESIZABLE
                )
                self.surface = pg.Surface((self.width, self.height))
            elif event.type == pg.MOUSEMOTION:
                self.mouse_active = True
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_r, pg.K_RETURN):
                    self.commands.append("reset")
                if event.key == pg.K_TAB:
                    self.commands.append("demo")
                if event.key == pg.K_p:
                    self.commands.append("pause")

    def input_action(self, heading):
        pg = self.pg
        keys = pg.key.get_pressed()
        turn = float(keys[pg.K_d] or keys[pg.K_RIGHT]) - float(
            keys[pg.K_a] or keys[pg.K_LEFT]
        )
        if turn == 0 and self.mouse_active and pg.mouse.get_focused():
            mx, my = pg.mouse.get_pos()
            dx, dy = mx - self.width / 2, my - self.height / 2
            if dx * dx + dy * dy > 25:
                a = math.atan2(dy, dx) - heading
                turn = max(-1, min(1, math.atan2(math.sin(a), math.cos(a)) * 3))
        boost = bool(keys[pg.K_SPACE] or pg.mouse.get_pressed()[0])
        return np.array([turn, float(boost)], np.float32)

    def _sprite(self, color, r):
        r = max(2, int(r))
        key = (color, r)
        if key in self._sprites:
            return self._sprites[key]
        pg = self.pg
        side = 2 * r + 4
        y, x = np.mgrid[:side, :side].astype(np.float32)
        x = (x - (r + 2)) / r
        y = (y - (r + 2)) / r
        distance = np.sqrt(x * x + y * y)
        z = np.sqrt(np.maximum(0, 1 - distance**2))
        lighting = np.clip(0.48 + 0.45 * z - 0.14 * x - 0.20 * y, 0.2, 1.1)
        rgb = np.clip(
            np.array(color)[None, None, :] * lighting[..., None], 0, 255
        ).astype(np.uint8)
        alpha = (np.clip((1 - distance) * r + 0.5, 0, 1) * 255).astype(np.uint8)
        surface = pg.Surface((side, side), pg.SRCALPHA, 32)
        pixels = pg.surfarray.pixels3d(surface)
        pixels[:] = rgb.transpose(1, 0, 2)
        del pixels
        pixels = pg.surfarray.pixels_alpha(surface)
        pixels[:] = alpha.T
        del pixels
        self._sprites[key] = surface
        return surface

    def _glow(self, color, r, strength=0.35):
        r = max(3, int(r))
        key = (color, r, strength)
        if key in self._glows:
            return self._glows[key]
        side = 2 * r + 2
        y, x = np.mgrid[:side, :side].astype(np.float32)
        distance = ((x - r) ** 2 + (y - r) ** 2) / (r * r)
        intensity = np.exp(-distance * 5) * strength * np.maximum(0, 1 - distance)
        rgb = np.clip(
            np.array(color)[None, None, :] * intensity[..., None], 0, 255
        ).astype(np.uint8)
        glow = self.pg.surfarray.make_surface(rgb.transpose(1, 0, 2))
        self._glows[key] = glow
        return glow

    def _background(self, camera, scale):
        pg = self.pg
        target = self.surface
        w, h = self.width, self.height
        target.fill((8, 15, 23))
        radius = 1.05
        dx, dy = math.sqrt(3) * radius, 1.5 * radius
        left, right = camera[0] - w / 2 / scale, camera[0] + w / 2 / scale
        top, bottom = camera[1] - h / 2 / scale, camera[1] + h / 2 / scale
        for row in range(math.floor(top / dy) - 1, math.ceil(bottom / dy) + 2):
            for col in range(math.floor(left / dx) - 2, math.ceil(right / dx) + 2):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                sx, sy = (
                    w / 2 + (x - camera[0]) * scale,
                    h / 2 + (y - camera[1]) * scale,
                )
                rp = radius * scale * 0.89
                pts = [
                    (
                        sx + rp * math.cos(math.pi / 6 + k * math.pi / 3),
                        sy + rp * math.sin(math.pi / 6 + k * math.pi / 3),
                    )
                    for k in range(6)
                ]
                shade = ((row * 92821 + col * 68917) & 31) / 31
                color = (int(20 + shade * 6), int(31 + shade * 7), int(43 + shade * 11))
                shadow = [(px + 3, py + 5) for px, py in pts]
                pg.draw.polygon(target, (4, 9, 15), shadow)
                pg.draw.polygon(target, (10, 20, 29), pts)
                inner = [
                    (sx + (px - sx) * 0.94, sy + (py - sy) * 0.94 - 1) for px, py in pts
                ]
                pg.draw.polygon(target, color, inner)
                pg.draw.lines(
                    target,
                    (int(color[0] * 1.15), int(color[1] * 1.15), int(color[2] * 1.12)),
                    False,
                    inner[2:5],
                    1,
                )

    def _minimap(self, core, obs, state):
        pg = self.pg
        c = core.config
        diameter = max(96, min(164, int(self.height * 0.23)))
        r = diameter // 2
        panel = pg.Surface((diameter + 4, diameter + 4), pg.SRCALPHA)
        center = (r + 2, r + 2)
        pg.draw.circle(panel, (30, 42, 55, 195), center, r)
        density = obs["minimap"][0, 0]
        M = density.shape[0]
        for row in range(M):
            for col in range(M):
                val = float(density[row, col])
                if val > 0:
                    px = int(2 + (col + 0.5) * diameter / M)
                    py = int(2 + (row + 0.5) * diameter / M)
                    if (px - center[0]) ** 2 + (py - center[1]) ** 2 < (r - 4) ** 2:
                        pg.draw.circle(
                            panel,
                            (180, 205, 225, int(70 + val * 150)),
                            (px, py),
                            max(1, int(1 + 3 * val)),
                        )
        pg.draw.circle(panel, (102, 127, 147, 150), center, r, 1)
        x = int(center[0] + state.head[0, 0, 0] / c.arena_radius * (r - 3))
        y = int(center[1] + state.head[0, 0, 1] / c.arena_radius * (r - 3))
        a = float(state.angle[0, 0])
        pg.draw.circle(panel, (180, 234, 234, 245), (x, y), 3)
        pg.draw.line(
            panel,
            (230, 250, 250, 240),
            (x, y),
            (x + int(7 * math.cos(a)), y + int(7 * math.sin(a))),
            2,
        )
        self.surface.blit(
            panel, (self.width - diameter - 20, self.height - diameter - 30)
        )
        label = self.small.render("minimap", True, (149, 164, 180))
        self.surface.blit(
            label, (self.width - label.get_width() - 25, self.height - 23)
        )

    def draw(self, core, obs, *, done=False, autopilot=False, paused=False, state=None):
        pg = self.pg
        s = core.state if state is None else state
        c = core.config
        w, h = self.width, self.height
        camera = s.head[0, 0].copy()
        # Every screen corner lies inside the policy's sensing disk. The frame
        # fills the window, rather than drawing a circular viewport or dashboard.
        scale = math.hypot(w, h) / (2 * c.local_radius)
        project = lambda x, y: (
            int(w / 2 + (x - camera[0]) * scale),
            int(h / 2 + (y - camera[1]) * scale),
        )
        self._background(camera, scale)
        center = project(0, 0)
        pg.draw.circle(
            self.surface,
            (128, 54, 63),
            center,
            int(c.arena_radius * scale),
            max(2, int(scale * 0.10)),
        )
        for f in np.flatnonzero(s.food_mass[0] > 0):
            x, y = s.food_pos[0, f]
            if (x - camera[0]) ** 2 + (y - camera[1]) ** 2 > c.local_radius**2:
                continue
            px, py = project(x, y)
            if not -20 <= px <= w + 20 or not -20 <= py <= h + 20:
                continue
            owner = int(s.food_owner[0, f])
            color = (
                PALETTE[owner % len(PALETTE)]
                if owner >= 0
                else FOOD_COLORS[int(f) % len(FOOD_COLORS)]
            )
            value = float(s.food_mass[0, f])
            radius = max(2, int(scale * food_radius(value, s.food_kind[0, f])))
            glow = self._glow(color, radius * 5, 0.42)
            self.surface.blit(
                glow,
                (px - glow.get_width() // 2, py - glow.get_height() // 2),
                special_flags=pg.BLEND_RGB_ADD,
            )
            sprite = self._sprite(color, radius)
            self.surface.blit(
                sprite, (px - sprite.get_width() // 2, py - sprite.get_height() // 2)
            )

        for i in range(c.num_snakes - 1, -1, -1):
            if not s.alive[0, i] and not (i == 0 and done):
                continue
            n = body_count(s, 0, i, c)
            world_radius = snake_radius(s.mass[0, i], c)
            radius = max(2, int(world_radius * scale))
            color = PALETTE[i % len(PALETTE)]
            sprite = self._sprite(color, radius)
            path = np.concatenate([s.head[0, i, None], s.body[0, i, :n]])
            # Draw closely spaced shaded discs from tail to head. These are
            # visual interpolation only and never enter collision calculations.
            for k in range(len(path) - 1, 0, -1):
                start, end = path[k], path[k - 1]
                length = float(np.linalg.norm(end - start))
                count = max(1, int(math.ceil(length / (world_radius * 0.45))))
                for j in range(count):
                    x, y = start + (end - start) * (j / count)
                    if (x - camera[0]) ** 2 + (y - camera[1]) ** 2 > (
                        c.local_radius + world_radius
                    ) ** 2:
                        continue
                    px, py = project(x, y)
                    if -radius <= px <= w + radius and -radius <= py <= h + radius:
                        self.surface.blit(
                            sprite,
                            (
                                px - sprite.get_width() // 2,
                                py - sprite.get_height() // 2,
                            ),
                        )
            x, y = s.head[0, i]
            if (x - camera[0]) ** 2 + (y - camera[1]) ** 2 > c.local_radius**2:
                continue
            px, py = project(x, y)
            if s.speed[0, i] > c.speed:
                glow = self._glow(color, radius * 4, 0.25)
                self.surface.blit(
                    glow,
                    (px - glow.get_width() // 2, py - glow.get_height() // 2),
                    special_flags=pg.BLEND_RGB_ADD,
                )
            self.surface.blit(
                sprite, (px - sprite.get_width() // 2, py - sprite.get_height() // 2)
            )
            angle = float(s.angle[0, i])
            for side in (-1, 1):
                a = angle + side * 0.62
                ex, ey = (
                    px + int(math.cos(a) * radius * 0.77),
                    py + int(math.sin(a) * radius * 0.77),
                )
                pg.draw.circle(
                    self.surface, (239, 250, 247), (ex, ey), max(2, int(radius * 0.35))
                )
                pg.draw.circle(
                    self.surface,
                    (7, 18, 20),
                    (
                        ex + int(math.cos(angle) * radius * 0.11),
                        ey + int(math.sin(angle) * radius * 0.11),
                    ),
                    max(1, int(radius * 0.22)),
                )

        self._minimap(core, obs, s)
        label = self.font.render(
            f"Your mass: {s.mass[0, 0]:.1f}", True, (210, 219, 228)
        )
        self.surface.blit(label, (12, h - 44))
        mode = "Autopilot" if autopilot else "Mouse / A D · Space to boost"
        self.surface.blit(self.small.render(mode, True, (143, 157, 174)), (12, h - 22))
        help_text = self.small.render(
            "R restart   Tab autopilot   P pause   Esc quit", True, (143, 157, 174)
        )
        self.surface.blit(help_text, (w - help_text.get_width() - 14, 14))
        if done or paused:
            overlay = pg.Surface((w, h), pg.SRCALPHA)
            overlay.fill((5, 10, 18, 100))
            self.surface.blit(overlay, (0, 0))
            text = "Paused" if paused else "Run ended"
            label = self.title.render(text, True, (228, 239, 243))
            self.surface.blit(label, ((w - label.get_width()) // 2, h // 2 - 30))
            sub = self.font.render(
                "P to continue" if paused else "Press R to play again",
                True,
                (168, 186, 200),
            )
            self.surface.blit(sub, ((w - sub.get_width()) // 2, h // 2 + 15))

    def present(self, limit=True):
        if self.mode == "human" and not self.closed:
            self.window.blit(self.surface, (0, 0))
            self.pg.display.flip()
            if limit:
                self.clock.tick(self.fps)

    def render(self, core, obs, *, done=False):
        self.poll()
        if self.closed:
            return None
        self.draw(core, obs, done=done)
        if self.mode == "rgb_array":
            return np.transpose(
                self.pg.surfarray.array3d(self.surface), (1, 0, 2)
            ).copy()
        self.present()
        return None

    def close(self):
        self.closed = True
        if self.window is not None:
            self.pg.display.quit()
            self.window = None

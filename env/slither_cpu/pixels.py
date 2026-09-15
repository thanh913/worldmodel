"""Deterministic RGB observations, rasterized on CPU with Numba batching.

Only the local camera and cached coarse minimap are drawn. No window, fonts,
external assets, global food map, or numeric feature side channel is required.
"""

import math
import numpy as np
from . import kernels
from .kernels import (
    kernel,
    prange,
    body_count,
    snake_radius,
    food_radius,
)
from .colors import PALETTE, FOOD_COLORS


@kernel
def snake_color(snake, observer):
    # Canonical observer color makes every agent's camera suitable for one policy.
    return (
        PALETTE[0]
        if snake == observer
        else PALETTE[1 + (snake - 1) % (len(PALETTE) - 1)]
    )


@kernel
def disc(frame, ax, ay, radius, color, shaded=True):
    """Subpixel edges and a fixed directional light; no supersampled framebuffer."""
    size = frame.shape[0]
    radius = max(0.25, radius)
    extent = radius + 0.5
    left = max(0, int(math.floor(ax - extent)))
    right = min(size - 1, int(math.ceil(ax + extent)))
    top = max(0, int(math.floor(ay - extent)))
    bottom = min(size - 1, int(math.ceil(ay + extent)))
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            dx, dy = x + 0.5 - ax, y + 0.5 - ay
            distance = math.sqrt(dx * dx + dy * dy)
            coverage = min(1.0, max(0.0, radius + 0.5 - distance))
            if coverage <= 0:
                continue
            light = 1.0
            if shaded:
                z = math.sqrt(max(0.0, 1 - (distance / radius) ** 2))
                light = min(
                    1.15,
                    max(
                        0.25, 0.50 + 0.48 * z - 0.12 * dx / radius - 0.18 * dy / radius
                    ),
                )
            for c in range(3):
                value = min(255.0, color[c] * light)
                frame[y, x, c] = int(frame[y, x, c] * (1 - coverage) + value * coverage)


@kernel
def glow(frame, x, y, radius, color):
    """Small static food halo. Brightness never depends on time or RNG."""
    size = frame.shape[0]
    radius = max(1.5, radius)
    for py in range(max(0, int(y - radius)), min(size, int(y + radius) + 1)):
        for px in range(max(0, int(x - radius)), min(size, int(x + radius) + 1)):
            d2 = ((px + 0.5 - x) ** 2 + (py + 0.5 - y) ** 2) / (radius * radius)
            amount = 0.22 * max(0.0, 1 - d2) ** 3
            for c in range(3):
                frame[py, px, c] = min(255, int(frame[py, px, c] + color[c] * amount))


@kernel
def hex_background(wx, wy, scale):
    """World-anchored beveled hexagons, matched to the native viewer's palette."""
    radius = 1.05
    spacing_x, spacing_y = math.sqrt(3.0) * radius, 1.5 * radius
    first_row = int(math.floor(wy / spacing_y))
    best = 1e20
    local_x = local_y = 0.0
    tile_row = tile_col = 0
    for row in range(first_row, first_row + 2):
        offset = (row % 2) * spacing_x / 2
        col = int(round((wx - offset) / spacing_x))
        dx, dy = wx - col * spacing_x - offset, wy - row * spacing_y
        distance = dx * dx + dy * dy
        if distance < best:
            best = distance
            local_x, local_y = dx, dy
            tile_row, tile_col = row, col
    edge = max(abs(local_x), 0.5 * abs(local_x) + math.sqrt(3.0) / 2 * abs(local_y))
    margin = radius * math.sqrt(3.0) / 2 * 0.90 - edge
    coverage = min(1.0, max(0.0, margin * scale + 0.5))
    bevel = 0.72 + 0.28 * min(1.0, max(0.0, margin * scale))
    shade = ((tile_row * 92821 + tile_col * 68917) & 31) / 31.0
    return (
        int(7 + coverage * ((20 + 6 * shade) * bevel - 7)),
        int(13 + coverage * ((31 + 7 * shade) * bevel - 13)),
        int(20 + coverage * ((43 + 11 * shade) * bevel - 20)),
    )


@kernel
def draw_minimap(frame, s, e, observer, p, total, alive):
    size = frame.shape[0]
    width = size // 4
    origin = size - width - 2
    center = origin + width / 2
    radius = width / 2
    for y in range(origin - 1, size - 1):
        for x in range(origin - 1, size - 1):
            d2 = (x + 0.5 - center) ** 2 + (y + 0.5 - center) ** 2
            if d2 > (radius + 1) ** 2:
                continue
            color = (90, 110, 125)
            if d2 <= radius * radius:
                row = min(
                    p.minimap_size - 1, max(0, (y - origin) * p.minimap_size // width)
                )
                col = min(
                    p.minimap_size - 1, max(0, (x - origin) * p.minimap_size // width)
                )
                density = min(
                    1.0,
                    max(0.0, (total[row, col] - s.maps[e, observer, row, col]) / 8.0),
                )
                color = (
                    int(24 + 150 * density),
                    int(36 + 165 * density),
                    int(48 + 175 * density),
                )
            for c in range(3):
                frame[y, x, c] = color[c]
    # Position is the observer's own marker; opponents are only cached cell density.
    x = center + s.head[e, observer, 0] / p.arena_radius * (radius - 1)
    y = center + s.head[e, observer, 1] / p.arena_radius * (radius - 1)
    color = PALETTE[0] if alive[e, observer] else (245, 90, 90)
    disc(frame, x, y, 0.85, color, False)


@kernel
def draw_observer(frame, s, e, observer, p, total, alive):
    size = frame.shape[0]
    scale = size / (math.sqrt(2.0) * p.local_radius)
    cx, cy = s.head[e, observer, 0], s.head[e, observer, 1]
    for y in range(size):
        wy = cy + (y + 0.5 - size / 2) / scale
        for x in range(size):
            wx = cx + (x + 0.5 - size / 2) / scale
            color = hex_background(wx, wy, scale)
            if wx * wx + wy * wy >= p.arena_radius * p.arena_radius:
                color = (95, 36, 45)
            for c in range(3):
                frame[y, x, c] = color[c]
    for f in range(p.food_capacity):
        if s.food_mass[e, f] <= 0:
            continue
        dx, dy = s.food_pos[e, f, 0] - cx, s.food_pos[e, f, 1] - cy
        if dx * dx + dy * dy > p.local_radius * p.local_radius:
            continue
        owner = s.food_owner[e, f]
        color = (
            snake_color(owner, observer)
            if owner >= 0
            else FOOD_COLORS[f % len(FOOD_COLORS)]
        )
        x, y = size / 2 + dx * scale, size / 2 + dy * scale
        radius = food_radius(s.food_mass[e, f], s.food_kind[e, f]) * scale
        glow(frame, x, y, 3 * radius + 1, color)
        disc(frame, x, y, max(0.6, radius), color)
    for j in range(p.num_snakes - 1, -1, -1):
        if not alive[e, j]:
            continue
        radius = snake_radius(s.mass[e, j], p) * scale
        color = snake_color(j, observer)
        # Closely spaced shaded discs follow the collision polyline, tail to head.
        # The interpolation is visual only; it never changes the simulation state.
        count = body_count(s, e, j, p)
        for k in range(count - 1, -1, -1):
            ax = size / 2 + (s.body[e, j, k, 0] - cx) * scale
            ay = size / 2 + (s.body[e, j, k, 1] - cy) * scale
            bx = (
                size / 2
                + ((s.head[e, j, 0] if k == 0 else s.body[e, j, k - 1, 0]) - cx) * scale
            )
            by = (
                size / 2
                + ((s.head[e, j, 1] if k == 0 else s.body[e, j, k - 1, 1]) - cy) * scale
            )
            steps = max(
                1, int(math.ceil(math.hypot(bx - ax, by - ay) / (radius * 0.65)))
            )
            for sample in range(steps):
                t = sample / steps
                x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                disc(frame, x, y, radius, color)
    for j in range(p.num_snakes):
        if not alive[e, j]:
            continue
        radius = snake_radius(s.mass[e, j], p) * scale
        x = size / 2 + (s.head[e, j, 0] - cx) * scale
        y = size / 2 + (s.head[e, j, 1] - cy) * scale
        color = snake_color(j, observer)
        if s.speed[e, j] > p.speed * 1.05:
            color = (
                min(255, color[0] + 25),
                min(255, color[1] + 25),
                min(255, color[2] + 25),
            )
        disc(frame, x, y, radius, color)
        for side in (-1, 1):
            angle = s.angle[e, j] + side * 0.62
            ex = x + math.cos(angle) * radius * 0.75
            ey = y + math.sin(angle) * radius * 0.75
            disc(
                frame, ex, ey, max(0.65, radius * 0.32), (242, 250, 249), False
            )
            px = ex + math.cos(s.angle[e, j]) * radius * 0.12
            py = ey + math.sin(s.angle[e, j]) * radius * 0.12
            disc(frame, px, py, max(0.25, radius * 0.17), (7, 18, 20), False)
    draw_minimap(frame, s, e, observer, p, total, alive)


@kernel
def draw_world(s, e, p, agents, alive, images):
    total = np.zeros((p.minimap_size, p.minimap_size), np.float32)
    for j in range(p.num_snakes):
        total += s.maps[e, j]
    for a in range(len(agents)):
        draw_observer(images[e, a], s, e, agents[a], p, total, alive)


def pixels_batch(s, p, agents, alive, images):
    for e in range(s.head.shape[0]):
        draw_world(s, e, p, agents, alive, images)


def pixels_parallel_batch(s, p, agents, alive, images):
    for e in prange(s.head.shape[0]):
        draw_world(s, e, p, agents, alive, images)


def make_pixel_kernel(parallel):
    fn = pixels_parallel_batch if parallel else pixels_batch
    return kernels.njit(cache=True, parallel=parallel)(fn)

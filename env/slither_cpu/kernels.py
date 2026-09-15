"""Numba CPU kernels; each parallel iteration owns one world's state."""

import math
import numpy as np

from numba import njit, prange
from numba.extending import register_jitable as kernel


TAU = 2.0 * math.pi


@kernel
def rand(s, e):
    # An explicit per-world LCG: deterministic across thread counts.
    x = (int(s.rng[e]) * 1664525 + 1013904223) & 0xFFFFFFFF
    s.rng[e] = x
    return (x + 0.5) / 4294967296.0


@kernel
def point_segment_d2(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    t = (
        0.0
        if den < 1e-20
        else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    )
    x, y = px - ax - t * dx, py - ay - t * dy
    return x * x + y * y


@kernel
def segments_d2(ax, ay, bx, by, cx, cy, dx, dy):
    ux, uy, vx, vy = bx - ax, by - ay, dx - cx, dy - cy
    det = ux * vy - uy * vx
    if abs(det) > 1e-15:
        wx, wy = cx - ax, cy - ay
        t, u = (wx * vy - wy * vx) / det, (wx * uy - wy * ux) / det
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(
        point_segment_d2(ax, ay, cx, cy, dx, dy),
        point_segment_d2(bx, by, cx, cy, dx, dy),
        point_segment_d2(cx, cy, ax, ay, bx, by),
        point_segment_d2(dx, dy, ax, ay, bx, by),
    )


@kernel
def snake_radius(mass, p):
    return p.body_radius * (mass / p.initial_mass) ** 0.25


@kernel
def body_length(mass, p):
    desired = (p.initial_mass - 1) * p.segment_spacing * (mass / p.initial_mass) ** 0.65
    return min(desired, (p.max_body_points - 2) * p.segment_spacing)


@kernel
def body_count(s, e, i, p):
    return int(math.ceil(body_length(s.mass[e, i], p) / p.segment_spacing))


@kernel
def update_body(s, e, i, p):
    """Sample the recorded path at distances behind the current head.

    History anchors only change at fixed travel intervals; the actual body and
    fractional tail move on every substep. Untravelled history is a collapsed tail.
    """
    length = body_length(s.mass[e, i], p)
    n = body_count(s, e, i, p)
    for k in range(n):
        distance = min((k + 1) * p.segment_spacing, length)
        # The shortest body is at least one sample long, and trail < spacing.
        q = max(0.0, (distance - s.trail[e, i]) / p.segment_spacing)
        j = int(q)
        t = q - j
        for axis in range(2):
            s.body[e, i, k, axis] = (
                s.history[e, i, j, axis] * (1 - t) + s.history[e, i, j + 1, axis] * t
            )


@kernel
def head_hits_body(s, e, i, j, old, reach, p):
    """Sweep head i against body j; moving-end approximation is bounded by substeps."""
    x, y = s.head[e, i, 0], s.head[e, i, 1]
    ax, ay = s.head[e, j, 0], s.head[e, j, 1]
    for k in range(body_count(s, e, j, p)):
        bx, by = s.body[e, j, k, 0], s.body[e, j, k, 1]
        if (
            max(old[i, 0], x) + reach >= min(ax, bx)
            and min(old[i, 0], x) - reach <= max(ax, bx)
            and max(old[i, 1], y) + reach >= min(ay, by)
            and min(old[i, 1], y) - reach <= max(ay, by)
            and segments_d2(old[i, 0], old[i, 1], x, y, ax, ay, bx, by) <= reach * reach
        ):
            return True
        ax, ay = bx, by
    return False


@kernel
def food_radius(value, kind):
    return min(0.32, (0.07 + 0.05 * math.sqrt(value)) * (1.4 if kind == 2 else 1.0))


@kernel
def add_food(s, e, p, x, y, value, kind, owner=-1):
    if value <= 0 or x * x + y * y >= p.arena_radius * p.arena_radius:
        return
    F = p.food_capacity
    slot = -1
    for offset in range(F):
        j = (s.food_cursor[e] + offset) % F
        if s.food_mass[e, j] <= 0:
            slot = j
            break
    if slot < 0:
        # Bounded storage: coalesce into the nearest existing pellet, preserving mass.
        best = 1e30
        for j in range(F):
            d = (x - s.food_pos[e, j, 0]) ** 2 + (y - s.food_pos[e, j, 1]) ** 2
            if d < best:
                best, slot = d, j
        # Death/boost mass never becomes persistent ambient mass on coalescence.
        s.food_kind[e, slot] = max(1, kind, s.food_kind[e, slot])
        if owner >= 0:
            s.food_owner[e, slot] = owner
    else:
        s.food_pos[e, slot, 0], s.food_pos[e, slot, 1] = x, y
        s.food_mass[e, slot] = 0.0
        s.food_kind[e, slot] = kind
        s.food_owner[e, slot] = owner
        s.food_cursor[e] = (slot + 1) % F
    s.food_mass[e, slot] += value
    s.food_ttl[e, slot] = max(1, int(p.food_lifetime / p.dt))


@kernel
def ambient_pellet(s, e, p):
    # Mostly uniform, with a mild central food hotspot.
    r = math.sqrt(rand(s, e)) * (p.arena_radius - p.body_radius)
    if rand(s, e) < 0.25:
        r *= 0.45
    a = TAU * rand(s, e)
    add_food(s, e, p, r * math.cos(a), r * math.sin(a), p.food_value, 0)


@kernel
def spawn(s, e, i, p):
    length = body_length(p.initial_mass, p)
    R = p.arena_radius - length - 3 * p.body_radius
    for attempt in range(p.max_spawn_attempts):
        r, a, heading = math.sqrt(rand(s, e)) * R, TAU * rand(s, e), TAU * rand(s, e)
        x, y = r * math.cos(a), r * math.sin(a)
        tx, ty = x - length * math.cos(heading), y - length * math.sin(heading)
        clear = True
        for j in range(p.num_snakes):
            if j == i or not s.alive[e, j]:
                continue
            safe2 = (p.body_radius + snake_radius(s.mass[e, j], p) + 0.5) ** 2
            ax, ay = s.head[e, j, 0], s.head[e, j, 1]
            for k in range(body_count(s, e, j, p)):
                bx, by = s.body[e, j, k, 0], s.body[e, j, k, 1]
                if segments_d2(x, y, tx, ty, ax, ay, bx, by) < safe2:
                    clear = False
                    break
                ax, ay = bx, by
            if not clear:
                break
        if not clear:
            continue
        s.head[e, i, 0], s.head[e, i, 1], s.angle[e, i] = x, y, heading
        s.mass[e, i], s.speed[e, i] = p.initial_mass, p.speed
        s.angular_velocity[e, i] = 0.0
        s.trail[e, i] = 0.0
        # Latent tail points collapse to the initial tail, never appear off-path on growth.
        for k in range(p.max_body_points):
            d = min(k * p.segment_spacing, length)
            s.history[e, i, k, 0] = x - d * math.cos(heading)
            s.history[e, i, k, 1] = y - d * math.sin(heading)
            s.body[e, i, k, 0] = tx
            s.body[e, i, k, 1] = ty
        update_body(s, e, i, p)
        s.alive[e, i], s.cooldown[e, i], s.age[e, i] = True, 0, 0
        s.residue[e, i] = 0.0
        s.episode[e, i] += 1
        return True
    return False


@kernel
def refresh_map(s, e, p):
    s.maps[e, :, :, :] = 0
    M, R = p.minimap_size, p.arena_radius
    for i in range(p.num_snakes):
        if not s.alive[e, i]:
            continue
        weight = (snake_radius(s.mass[e, i], p) / p.body_radius) ** 2
        for k in range(body_count(s, e, i, p)):
            x, y = s.body[e, i, k, 0], s.body[e, i, k, 1]
            col = min(M - 1, max(0, int((x / R + 1) * 0.5 * M)))
            row = min(M - 1, max(0, int((y / R + 1) * 0.5 * M)))
            s.maps[e, i, row, col] += weight


@kernel
def drop_corpse(s, e, i, p):
    """Stratified samples around the curved corpse, with conserved total mass."""
    mass = s.mass[e, i] * p.death_drop_fraction
    count = min(
        p.food_capacity, 3 * p.max_body_points, max(6, int(math.ceil(s.mass[e, i])))
    )
    weights = np.empty(count, dtype=np.float64)
    total = 0.0
    for k in range(count):
        weights[k] = 0.25 + 1.75 * rand(s, e) ** 2
        total += weights[k]
    length = body_length(s.mass[e, i], p)
    n = body_count(s, e, i, p)
    radius = snake_radius(s.mass[e, i], p)
    for k in range(count):
        d = length * (k + rand(s, e)) / count
        j = min(n - 1, int(d / p.segment_spacing))
        t = (d - j * p.segment_spacing) / min(
            p.segment_spacing, length - j * p.segment_spacing
        )
        ax = s.head[e, i, 0] if j == 0 else s.body[e, i, j - 1, 0]
        ay = s.head[e, i, 1] if j == 0 else s.body[e, i, j - 1, 1]
        x = ax + (s.body[e, i, j, 0] - ax) * t
        y = ay + (s.body[e, i, j, 1] - ay) * t
        scatter = radius * math.sqrt(rand(s, e))
        angle = TAU * rand(s, e)
        x += scatter * math.cos(angle)
        y += scatter * math.sin(angle)
        # Keep boundary corpses inside the arena without losing their donated mass.
        scale = min(1.0, (p.arena_radius - 1e-6) / max(1e-6, math.sqrt(x * x + y * y)))
        add_food(s, e, p, x * scale, y * scale, mass * weights[k] / total, 2, i)


@kernel
def reset_world(s, e, p):
    s.alive[e, :] = False
    s.food_mass[e, :] = 0
    s.food_cursor[e], s.ticks[e] = 0, 0
    for i in range(p.num_snakes):
        spawn(s, e, i, p)
    for j in range(p.ambient_food):
        ambient_pellet(s, e, p)
    refresh_map(s, e, p)


@kernel
def step_world(s, e, p, actions, reward, terminated, truncated, valid, spawned):
    S = p.num_snakes
    # Reset-only transitions are masked out: an action based on a dead observation
    # is never executed by its replacement. The returned live observation is fresh.
    for i in range(S):
        if not s.alive[e, i]:
            s.cooldown[e, i] -= 1
            if s.cooldown[e, i] <= 0:
                spawned[e, i] = spawn(s, e, i, p)
        else:
            valid[e, i] = True

    max_move = min(snake_radius(p.min_mass, p) * 0.5, p.segment_spacing * 0.5)
    substeps = max(1, int(math.ceil(p.boost_speed * p.dt / max_move)))
    h = p.dt / substeps
    steering_blend = 1 - math.exp(-h / p.steering_response)
    speed_blend = 1 - math.exp(-h / p.speed_response)
    old = np.empty((S, 2), dtype=np.float64)
    radii = np.empty(S, dtype=np.float64)
    dead = np.zeros(S, dtype=np.bool_)
    for sub in range(substeps):
        for i in range(S):
            old[i, 0], old[i, 1] = s.head[e, i, 0], s.head[e, i, 1]
            dead[i] = False
            if not s.alive[e, i] or not valid[e, i]:
                continue
            boost = actions[e, i, 1] > 0.5 and s.mass[e, i] > p.min_mass + 1e-9
            if boost:
                cost = min(p.boost_cost * h, s.mass[e, i] - p.min_mass)
                s.mass[e, i] -= cost
                reward[e, i] -= cost
                s.residue[e, i] += cost * p.boost_drop_fraction
                if s.residue[e, i] >= 0.2:
                    k = body_count(s, e, i, p) - 1
                    add_food(
                        s,
                        e,
                        p,
                        s.body[e, i, k, 0],
                        s.body[e, i, k, 1],
                        s.residue[e, i],
                        1,
                        i,
                    )
                    s.residue[e, i] = 0.0
            limit = p.turn_rate / (snake_radius(s.mass[e, i], p) / p.body_radius)
            target_turn = actions[e, i, 0] * limit
            s.angular_velocity[e, i] += (
                target_turn - s.angular_velocity[e, i]
            ) * steering_blend
            s.angular_velocity[e, i] = max(-limit, min(limit, s.angular_velocity[e, i]))
            s.angle[e, i] = (
                s.angle[e, i] + s.angular_velocity[e, i] * h + math.pi
            ) % TAU - math.pi
            target_speed = p.boost_speed if boost else p.speed
            s.speed[e, i] += (target_speed - s.speed[e, i]) * speed_blend
            speed = s.speed[e, i]
            dx, dy = (
                speed * h * math.cos(s.angle[e, i]),
                speed * h * math.sin(s.angle[e, i]),
            )
            distance = speed * h
            phase = s.trail[e, i] + distance
            if phase >= p.segment_spacing:
                t = (p.segment_spacing - s.trail[e, i]) / distance
                # One insertion at most: substeps bound distance below spacing.
                for k in range(p.max_body_points - 1, 0, -1):
                    s.history[e, i, k, 0], s.history[e, i, k, 1] = (
                        s.history[e, i, k - 1, 0],
                        s.history[e, i, k - 1, 1],
                    )
                s.history[e, i, 0, 0], s.history[e, i, 0, 1] = (
                    old[i, 0] + t * dx,
                    old[i, 1] + t * dy,
                )
                phase -= p.segment_spacing
            s.trail[e, i] = phase
            s.head[e, i, 0] += dx
            s.head[e, i, 1] += dy
            update_body(s, e, i, p)

        for i in range(S):
            radii[i] = snake_radius(s.mass[e, i], p)
        for i in range(S):
            if not s.alive[e, i]:
                continue
            x, y = s.head[e, i, 0], s.head[e, i, 1]
            if x * x + y * y >= (p.arena_radius - radii[i]) ** 2:
                dead[i] = True
            for j in range(i + 1, S):
                if not s.alive[e, j]:
                    continue
                reach = radii[i] + radii[j]
                # Relative swept heads: detects crossing between sampled positions.
                d2 = point_segment_d2(
                    0.0,
                    0.0,
                    old[i, 0] - old[j, 0],
                    old[i, 1] - old[j, 1],
                    x - s.head[e, j, 0],
                    y - s.head[e, j, 1],
                )
                if d2 <= reach * reach:
                    # Resolve this pair once: higher mass wins, equal mass kills
                    # both. Do not count the same contact again against the neck.
                    # Preserve deaths caused by a wall or any other opponent.
                    dead[i] = dead[i] or s.mass[e, i] <= s.mass[e, j]
                    dead[j] = dead[j] or s.mass[e, j] <= s.mass[e, i]
                    continue
                dead[i] = dead[i] or head_hits_body(s, e, i, j, old, reach, p)
                dead[j] = dead[j] or head_hits_body(s, e, j, i, old, reach, p)

        # Commit every death together, then create drops; dead snakes cannot eat.
        for i in range(S):
            if dead[i]:
                s.alive[e, i], terminated[e, i] = False, True
                reward[e, i] -= p.death_penalty
                s.cooldown[e, i] = max(1, int(math.ceil(p.respawn_delay / p.dt)))
                drop_corpse(s, e, i, p)

        # Shared food: nearest swept live head wins. Exact ties use per-world RNG.
        for f in range(p.food_capacity):
            if s.food_mass[e, f] <= 0:
                continue
            x, y = s.food_pos[e, f, 0], s.food_pos[e, f, 1]
            orb_radius = food_radius(s.food_mass[e, f], s.food_kind[e, f])
            best, winner, ties = 1e30, -1, 0
            for i in range(S):
                if not s.alive[e, i] or not valid[e, i]:
                    continue
                reach = radii[i] + max(orb_radius, p.food_attraction_radius)
                if (
                    abs(x - s.head[e, i, 0]) > reach + p.boost_speed * h
                    or abs(y - s.head[e, i, 1]) > reach + p.boost_speed * h
                ):
                    continue
                d = point_segment_d2(
                    x, y, old[i, 0], old[i, 1], s.head[e, i, 0], s.head[e, i, 1]
                )
                if d > reach * reach:
                    continue
                if d < best - 1e-12:
                    best, winner, ties = d, i, 1
                elif abs(d - best) <= 1e-12:
                    ties += 1
                    if rand(s, e) < 1.0 / ties:
                        winner = i
            if winner < 0:
                continue
            eat_radius = radii[winner] + orb_radius
            if best > eat_radius * eat_radius:
                dx, dy = x - s.head[e, winner, 0], y - s.head[e, winner, 1]
                distance = math.sqrt(dx * dx + dy * dy)
                forward = dx * math.cos(s.angle[e, winner]) + dy * math.sin(
                    s.angle[e, winner]
                )
                if forward <= 0 or distance > radii[winner] + p.food_attraction_radius:
                    continue
                movement = min(distance, p.food_attraction_speed * h)
                s.food_pos[e, f, 0] -= dx * movement / distance
                s.food_pos[e, f, 1] -= dy * movement / distance
                if distance - movement > eat_radius:
                    continue
            gain = min(s.food_mass[e, f], p.max_mass - s.mass[e, winner])
            s.mass[e, winner] += gain
            reward[e, winner] += gain
            s.food_mass[e, f] = 0.0
            radii[winner] = snake_radius(s.mass[e, winner], p)
            update_body(s, e, winner, p)

    ambient = 0
    for f in range(p.food_capacity):
        if s.food_mass[e, f] <= 0:
            continue
        if s.food_kind[e, f] == 0:
            ambient += 1
            continue
        s.food_ttl[e, f] -= 1
        if s.food_ttl[e, f] <= 0:
            s.food_mass[e, f] = 0.0
    expected = p.food_respawn_rate * p.dt
    births = int(expected) + (rand(s, e) < expected - int(expected))
    for k in range(min(births, max(0, p.ambient_food - ambient))):
        ambient_pellet(s, e, p)
    for i in range(S):
        if not valid[e, i]:
            continue
        s.age[e, i] += 1
        # Terminal observation is produced below before the next-step reset.
        truncated[e, i] = s.alive[e, i] and s.age[e, i] >= p.max_episode_steps
    s.ticks[e] += 1
    if s.ticks[e] % p.minimap_interval == 0:
        refresh_map(s, e, p)


@kernel
def splat(local, channel, dx, dy, radius, heading, p):
    d = math.sqrt(dx * dx + dy * dy)
    if d - radius > p.local_radius:
        return
    distance = max(0.0, min(1.0, (d - radius) / p.local_radius))
    a = (math.atan2(dy, dx) - heading + math.pi) % TAU
    center = a / TAU * p.sectors
    if d <= radius:
        half = p.sectors // 2 + 1
    else:
        half = int(math.ceil(math.asin(min(1.0, radius / d)) / TAU * p.sectors))
    index = int(center) % p.sectors
    for offset in range(-half, half + 1):
        b = (index + offset) % p.sectors
        local[b, channel] = min(local[b, channel], distance)


@kernel
def observe_world(s, e, p, local, own, minimap):
    S, M = p.num_snakes, p.minimap_size
    total = np.zeros((M, M), dtype=np.float32)
    radii = np.empty(S, dtype=np.float64)
    for i in range(S):
        total += s.maps[e, i]
        radii[i] = snake_radius(s.mass[e, i], p)
    for i in range(S):
        obs = local[e, i]
        obs[:, :] = 1.0
        obs[:, 3] = 0.0
        x, y, a = s.head[e, i, 0], s.head[e, i, 1], s.angle[e, i]
        own[e, i, 0] = s.mass[e, i] / p.max_mass
        own[e, i, 1] = s.speed[e, i] / p.boost_speed
        own[e, i, 2] = s.mass[e, i] > p.min_mass + 1e-9
        own[e, i, 3], own[e, i, 4] = x / p.arena_radius, y / p.arena_radius
        own[e, i, 5], own[e, i, 6] = math.sin(a), math.cos(a)
        own[e, i, 7] = s.alive[e, i]
        own[e, i, 8] = (s.ticks[e] % p.minimap_interval) / p.minimap_interval
        for row in range(M):
            for col in range(M):
                minimap[e, i, row, col] = min(
                    1.0, max(0.0, (total[row, col] - s.maps[e, i, row, col]) / 8.0)
                )
        R = p.arena_radius - radii[i]
        for b in range(p.sectors):
            angle = a - math.pi + (b + 0.5) * TAU / p.sectors
            dot = x * math.cos(angle) + y * math.sin(angle)
            d = -dot + math.sqrt(max(0.0, dot * dot + R * R - x * x - y * y))
            obs[b, 5] = max(0.0, min(1.0, d / p.local_radius))
        for j in range(S):
            if not s.alive[e, j]:
                continue
            if j != i:
                splat(obs, 1, s.head[e, j, 0] - x, s.head[e, j, 1] - y, radii[j], a, p)
            ax, ay = s.head[e, j, 0], s.head[e, j, 1]
            for k in range(body_count(s, e, j, p)):
                bx, by = s.body[e, j, k, 0], s.body[e, j, k, 1]
                radius = radii[j] + 0.5 * math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
                # Omit the immediate neck from the own-body sensor; otherwise
                # the head-containing primitive would zero every angular bin.
                if j != i or k >= 2:
                    splat(
                        obs,
                        4 if j == i else 0,
                        (ax + bx) * 0.5 - x,
                        (ay + by) * 0.5 - y,
                        radius,
                        a,
                        p,
                    )
                ax, ay = bx, by
        for f in range(p.food_capacity):
            value = s.food_mass[e, f]
            if value <= 0:
                continue
            dx, dy = s.food_pos[e, f, 0] - x, s.food_pos[e, f, 1] - y
            d2 = dx * dx + dy * dy
            if d2 > p.local_radius * p.local_radius:
                continue
            b = (
                int(((math.atan2(dy, dx) - a + math.pi) % TAU) / TAU * p.sectors)
                % p.sectors
            )
            obs[b, 2] = min(obs[b, 2], math.sqrt(d2) / p.local_radius)
            obs[b, 3] = min(1.0, obs[b, 3] + value / 8.0)


def reset_batch(s, p):
    for e in range(s.head.shape[0]):
        reset_world(s, e, p)


def step_batch(
    s, p, actions, reward, terminated, truncated, valid, spawned, active_worlds
):
    for e in range(s.head.shape[0]):
        if active_worlds[e]:
            step_world(s, e, p, actions, reward, terminated, truncated, valid, spawned)


def reset_mask_batch(s, p, mask):
    for e in range(s.head.shape[0]):
        if mask[e]:
            reset_world(s, e, p)


def observe_batch(s, p, local, own, minimap):
    for e in range(s.head.shape[0]):
        observe_world(s, e, p, local, own, minimap)


def reset_parallel_batch(s, p):
    for e in prange(s.head.shape[0]):
        reset_world(s, e, p)


def step_parallel_batch(
    s, p, actions, reward, terminated, truncated, valid, spawned, active_worlds
):
    for e in prange(s.head.shape[0]):
        if active_worlds[e]:
            step_world(s, e, p, actions, reward, terminated, truncated, valid, spawned)


def observe_parallel_batch(s, p, local, own, minimap):
    for e in prange(s.head.shape[0]):
        observe_world(s, e, p, local, own, minimap)


def reset_mask_parallel_batch(s, p, mask):
    for e in prange(s.head.shape[0]):
        if mask[e]:
            reset_world(s, e, p)


def make_kernels(parallel):
    # Numba's disk cache key does not include compiler options. Give parallel
    # kernels distinct functions so a native viewer's serial cache can never
    # replace a training process's parallel code (or vice versa).
    functions = (
        (
            reset_parallel_batch,
            step_parallel_batch,
            observe_parallel_batch,
            reset_mask_parallel_batch,
        )
        if parallel
        else (reset_batch, step_batch, observe_batch, reset_mask_batch)
    )
    return tuple(njit(cache=True, parallel=parallel)(fn) for fn in functions)

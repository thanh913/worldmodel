"""Local-sensor heuristic baseline, not a trained pixel policy.

In pixel mode, pass env.observe_features() instead of RGB images. The heuristic
reads only the sensor dict, never simulation state or exact distant positions.
"""

import numpy as np


def forage_policy(observation):
    local, own = observation["local"], observation["self"]
    sectors = local.shape[-2]
    angle = -np.pi + (np.arange(sectors) + 0.5) * 2 * np.pi / sectors
    danger = np.minimum(np.minimum(local[..., 0], local[..., 1]), local[..., 5])
    food = local[..., 3] / (0.12 + local[..., 2])
    score = 1.2 * food + 0.35 * np.cos(angle) + 0.35 * danger
    score -= 8.0 * np.maximum(0.0, 0.23 - danger)
    # Discourage nearly reversing; steering still permits it when avoiding hazards.
    score -= 0.10 * np.abs(angle)
    best = np.argmax(score, axis=-1)
    # Blend only the winning direction's safe neighbors. Averaging across all
    # directions could point between two food targets and straight into a body.
    neighbors = (best[..., None] + np.arange(-1, 2)) % sectors
    neighbor_score = np.take_along_axis(score, neighbors, axis=-1)
    neighbor_danger = np.take_along_axis(danger, neighbors, axis=-1)
    weight = np.exp(4 * (neighbor_score - neighbor_score.max(axis=-1, keepdims=True)))
    weight *= neighbor_danger >= np.minimum(
        0.23, np.take_along_axis(danger, best[..., None], axis=-1)
    )
    target = np.arctan2(
        (weight * np.sin(angle[neighbors])).sum(axis=-1),
        (weight * np.cos(angle[neighbors])).sum(axis=-1),
    )
    actions = np.zeros((*own.shape[:2], 2), np.float64)
    actions[..., 0] = np.clip(target * 1.7, -1, 1)
    chosen_danger = np.take_along_axis(danger, best[..., None], axis=-1)[..., 0]
    chosen_food = np.take_along_axis(food, best[..., None], axis=-1)[..., 0]
    actions[..., 1] = (
        (np.abs(target) < 0.22)
        & (chosen_danger > 0.55)
        & (chosen_food > 0.7)
        & (own[..., 2] > 0)
    ).astype(float)
    actions *= own[..., 7, None]
    return actions

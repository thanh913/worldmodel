from typing import NamedTuple
import numpy as np


class State(NamedTuple):
    head: np.ndarray
    angle: np.ndarray
    mass: np.ndarray
    body: np.ndarray
    history: np.ndarray
    trail: np.ndarray
    speed: np.ndarray
    angular_velocity: np.ndarray
    alive: np.ndarray
    cooldown: np.ndarray
    age: np.ndarray
    episode: np.ndarray
    residue: np.ndarray
    food_pos: np.ndarray
    food_mass: np.ndarray
    food_ttl: np.ndarray
    food_kind: np.ndarray
    food_owner: np.ndarray
    food_cursor: np.ndarray
    ticks: np.ndarray
    rng: np.ndarray
    maps: np.ndarray


def allocate(worlds, c):
    E, S, L, F, M = (
        worlds,
        c.num_snakes,
        c.max_body_points,
        c.food_capacity,
        c.minimap_size,
    )
    z = np.zeros
    return State(
        z((E, S, 2), np.float64),
        z((E, S), np.float64),
        z((E, S), np.float64),
        z((E, S, L, 2), np.float64),
        z((E, S, L, 2), np.float64),
        z((E, S), np.float64),
        z((E, S), np.float64),
        z((E, S), np.float64),
        z((E, S), np.bool_),
        z((E, S), np.int32),
        z((E, S), np.int32),
        z((E, S), np.int64),
        z((E, S), np.float64),
        z((E, F, 2), np.float64),
        z((E, F), np.float64),
        z((E, F), np.int32),
        z((E, F), np.int8),
        np.full((E, F), -1, np.int32),
        z(E, np.int32),
        z(E, np.int64),
        z(E, np.int64),
        z((E, S, M, M), np.float32),
    )

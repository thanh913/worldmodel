"""Portable, explicit game rules. Distances and mass are arbitrary game units."""

from collections import namedtuple
from dataclasses import dataclass, fields, astuple
import math


@dataclass(frozen=True)
class Config:
    num_snakes: int = 8
    max_body_points: int = 128
    food_capacity: int = 1024
    ambient_food: int = 256
    arena_radius: float = 32.0
    body_radius: float = 0.35
    segment_spacing: float = 0.5
    initial_mass: float = 12.0
    min_mass: float = 4.0
    max_mass: float = 256.0
    dt: float = 0.05
    speed: float = 3.0
    boost_speed: float = 6.0
    turn_rate: float = 3.0
    steering_response: float = 0.10
    speed_response: float = 0.15
    boost_cost: float = 1.2
    boost_drop_fraction: float = 0.5
    death_drop_fraction: float = 0.7
    food_value: float = 0.6
    food_respawn_rate: float = 8.0
    food_lifetime: float = 120.0
    food_attraction_radius: float = 1.0
    food_attraction_speed: float = 2.0
    respawn_delay: float = 0.5
    max_episode_steps: int = 2400
    local_radius: float = 10.0
    sectors: int = 32
    minimap_size: int = 16
    minimap_interval: int = 20
    pixel_size: int = 128
    death_penalty: float = 5.0
    max_spawn_attempts: int = 64

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{f.name} must be a finite number")
            if f.type is int and (isinstance(v, bool) or not isinstance(v, int)):
                raise ValueError(f"{f.name} must be an integer")
        for name in (
            "num_snakes",
            "max_body_points",
            "food_capacity",
            "sectors",
            "minimap_size",
            "minimap_interval",
            "max_episode_steps",
            "max_spawn_attempts",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "arena_radius",
            "body_radius",
            "segment_spacing",
            "dt",
            "speed",
            "boost_speed",
            "turn_rate",
            "steering_response",
            "speed_response",
            "food_value",
            "food_lifetime",
            "local_radius",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 2 <= self.min_mass <= self.initial_mass <= self.max_mass:
            raise ValueError("Require 2 <= min_mass <= initial_mass <= max_mass")
        if self.max_body_points < self.initial_mass + 2:
            raise ValueError(
                "max_body_points must hold the initial body and two path samples"
            )
        if self.boost_speed < self.speed:
            raise ValueError("boost_speed must be at least speed")
        if self.pixel_size < 32:
            raise ValueError("pixel_size must be at least 32 to include the minimap")
        if not 0 <= self.ambient_food <= self.food_capacity:
            raise ValueError("ambient_food must fit food_capacity")
        for name in (
            "boost_cost",
            "food_respawn_rate",
            "respawn_delay",
            "death_penalty",
            "food_attraction_radius",
            "food_attraction_speed",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("boost_drop_fraction", "death_drop_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            self.initial_mass * self.segment_spacing + 3 * self.body_radius
            >= self.arena_radius
        ):
            raise ValueError("Arena is too small to safely spawn the initial body")
        if (
            self.body_radius * (self.max_mass / self.initial_mass) ** 0.25
            >= self.arena_radius
        ):
            raise ValueError("The largest snake must fit inside the arena")

    def kernel_params(self):
        return Params(*astuple(self))


Params = namedtuple("Params", [f.name for f in fields(Config)])

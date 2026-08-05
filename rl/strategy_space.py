"""Strategy Parameter Space for Kaggriculture RL.

Maps normalized strategy vectors in [-1.0, 1.0]^D to concrete macro parameters
and task priorities in main.py.
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any, NamedTuple


class ParamDef(NamedTuple):
    name: str
    min_val: float
    max_val: float
    val_type: type  # int, float, bool
    default: float | int | bool
    description: str


PARAM_DEFS: list[ParamDef] = [
    # Labour & Hiring
    ParamDef("MAX_HANDS", 4, 24, int, 12, "Maximum farmhands to hire"),
    ParamDef("HIRES_PER_TURN", 1, 8, int, 6, "Maximum hires allowed per turn"),
    ParamDef("HIRE_CASH_FRACTION", 0.05, 0.50, float, 0.25, "Max cash fraction spent on hiring"),
    # Land Expansion
    ParamDef("LAND_CASH_BUFFER", 0, 5000, float, 1000, "Cash buffer required before buying land"),
    ParamDef("LAND_LAST_DAY", 10, 28, int, 22, "Cutoff day for buying land"),
    # Livestock Targets
    ParamDef("GOOSE_MIN_DAYS_LEFT", 4, 20, int, 9, "Minimum days left to buy goose"),
    ParamDef("SHEEP_MIN_DAYS_LEFT", 4, 20, int, 10, "Minimum days left to buy sheep"),
    ParamDef("COW_MIN_DAYS_LEFT", 4, 20, int, 10, "Minimum days left to buy cow"),
    ParamDef("MAX_SHEEP", 0, 12, int, 6, "Max sheep ceiling"),
    ParamDef("MAX_COWS", 0, 12, int, 8, "Max cow ceiling"),
    # Crop & Feed Infrastructure
    ParamDef("WHEAT_TILES_PER_ANIMAL", 0.3, 1.8, float, 0.8, "Wheat tile target per animal"),
    ParamDef("WHEAT_FEED_DAYS_RESERVE", 1, 7, int, 3, "Days of wheat feed reserve target"),
    ParamDef("STRAWBERRY_TILE_TARGET", 5, 50, int, 35, "Target strawberry tile count"),
    ParamDef(
        "STRAWBERRY_LAND_FRACTION", 0.10, 0.80, float, 0.50, "Max fraction of land for strawberry"
    ),
    ParamDef("MELON_TILE_TARGET", 4, 30, int, 12, "Target melon tile count"),
    ParamDef("MELON_LAND_FRACTION", 0.10, 0.70, float, 0.30, "Max fraction of land for melon"),
    ParamDef("MELON_LAST_PLANT_DAY", 8, 25, int, 17, "Cutoff day for planting melon"),
    # Priorities
    ParamDef("PRIO_FEED_URGENT", 800, 1200, int, 1000, "Priority for urgent feeding"),
    ParamDef("PRIO_WATER_URGENT", 750, 1150, int, 950, "Priority for urgent watering"),
    ParamDef("PRIO_HARVEST_ANIMAL_FULL", 700, 1100, int, 900, "Priority for full animal harvest"),
    ParamDef("PRIO_HARVEST_CROP", 650, 1050, int, 850, "Priority for crop harvest"),
    ParamDef("PRIO_CARE", 500, 900, int, 700, "Priority for animal care"),
    ParamDef("PRIO_FEED", 500, 900, int, 690, "Priority for standard feeding"),
]


class StrategySpace:
    """Strategy Space mapping [-1, 1]^D to parameter dicts and code overrides."""

    def __init__(self, defs: list[ParamDef] | None = None) -> None:
        self.defs = defs or PARAM_DEFS
        self._name_map = {p.name: p for p in self.defs}

    @property
    def dim(self) -> int:
        return len(self.defs)

    def get_default_vector(self) -> list[float]:
        """Return the normalized [-1, 1] vector for default main.py settings."""
        vec = []
        for p in self.defs:
            if p.val_type is bool:
                val = 1.0 if p.default else -1.0
            else:
                # Linear map from [min, max] to [-1, 1]
                val = 2.0 * (float(p.default) - p.min_val) / (p.max_val - p.min_val) - 1.0
            vec.append(max(-1.0, min(1.0, val)))
        return vec

    def vector_to_dict(self, vector: list[float]) -> dict[str, Any]:
        """Convert normalized [-1, 1]^D vector into concrete parameter dictionary."""
        if len(vector) != self.dim:
            raise ValueError(f"Vector length {len(vector)} != strategy dimension {self.dim}")

        res: dict[str, Any] = {}
        for p, norm_val in zip(self.defs, vector, strict=False):
            norm_clamped = max(-1.0, min(1.0, norm_val))
            if p.val_type is bool:
                res[p.name] = norm_clamped >= 0.0
            else:
                raw_val = p.min_val + 0.5 * (norm_clamped + 1.0) * (p.max_val - p.min_val)
                if p.val_type is int:
                    res[p.name] = round(raw_val)
                else:
                    res[p.name] = round(raw_val, 4)
        return res

    def dict_to_vector(self, param_dict: dict[str, Any]) -> list[float]:
        """Convert a parameter dictionary into a normalized [-1, 1]^D vector."""
        vec = []
        for p in self.defs:
            val = param_dict.get(p.name, p.default)
            if p.val_type is bool:
                norm_val = 1.0 if val else -1.0
            else:
                norm_val = 2.0 * (float(val) - p.min_val) / (p.max_val - p.min_val) - 1.0
            vec.append(max(-1.0, min(1.0, norm_val)))
        return vec

    def mutate(
        self,
        vector: list[float],
        scale: float = 0.15,
        p_mutate: float = 0.35,
    ) -> list[float]:
        """Gaussian mutation of a strategy vector within [-1.0, 1.0]."""
        mutated = copy.deepcopy(vector)
        for i in range(len(mutated)):
            if random.random() < p_mutate:
                noise = random.gauss(0.0, scale)
                mutated[i] = max(-1.0, min(1.0, mutated[i] + noise))
        return mutated

    def apply_to_file(
        self,
        src_path: str,
        dest_path: str,
        vector: list[float] | dict[str, Any],
    ) -> str:
        """Write a new main.py variant file with the specified strategy overrides."""
        param_dict = vector if isinstance(vector, dict) else self.vector_to_dict(vector)

        with open(src_path) as f:
            src = f.read()

        for name, value in param_dict.items():
            pattern = rf"(?m)^({re.escape(name)}\s*=\s*)([^\n#]+)"

            def _repl_param(m, v=value):
                return m.group(1) + repr(v)

            new_src, n = re.subn(pattern, _repl_param, src)
            if n > 0:
                src = new_src

        with open(dest_path, "w") as f:
            f.write(src)

        return dest_path

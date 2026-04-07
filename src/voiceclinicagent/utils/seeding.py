"""Seeded RNG utilities for deterministic episode execution."""

import random
from typing import Optional


def make_episode_rng(seed: Optional[int] = None) -> random.Random:
    """
    Create a seeded random.Random instance for deterministic episode execution.
    
    Args:
        seed: Random seed. If None, uses system randomness.
        
    Returns:
        Seeded random.Random instance
        
    Example:
        >>> rng = make_episode_rng(42)
        >>> rng.random()  # Always returns same value for seed=42
        0.6394267984578837
    """
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng

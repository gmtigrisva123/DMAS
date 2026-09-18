"""Deterministic hashing and rngs. Python's hash() is salted per process, so
every hash that affects a result goes through stable_hash.
"""

import hashlib
import random
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

_MASK64 = (1 << 64) - 1


def stable_hash(*parts: object) -> int:
    """Process independent 64 bit hash of the parts."""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x1f")
    return int.from_bytes(digest.digest(), "big") & _MASK64


def stable_unit(*parts: object) -> float:
    """Deterministic pseudo random float in [0, 1) from parts."""
    return stable_hash(*parts) / float(1 << 64)


def seeded_rng(*parts: object) -> random.Random:
    """random.Random seeded from parts."""
    return random.Random(stable_hash(*parts))


def deterministic_sample(items: Sequence[T], k: int, *seed: object) -> list:
    """Sample k items without replacement, reproducibly."""
    if k >= len(items):
        return list(items)
    return seeded_rng(*seed).sample(list(items), k)


def interleave(*iterables: Iterable[T]) -> Iterator[T]:
    """Round robin interleave, keeps ranked lists balanced."""
    iterators = [iter(it) for it in iterables]
    while iterators:
        alive = []
        for iterator in iterators:
            try:
                yield next(iterator)
            except StopIteration:
                continue
            alive.append(iterator)
        iterators = alive

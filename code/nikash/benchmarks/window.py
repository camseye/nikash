"""Size budget for a case window, shared by every benchmark loader.

An event count alone does not bound a window: one Suricata eve.json line can
run to thousands of characters, so 100 events can exceed both GPU memory and
the context of the smallest tested model (8,192 tokens). Every window is
therefore also held to WINDOW_CHARS, measured as the widest defence renders
it (spotlighting adds markers around each attacker-controlled field), with
PAYLOAD_RESERVE left for the injected text. The budget is in characters, not
tokens, so every model and defence sees the same window.
"""

from __future__ import annotations

import random

from ..defences.prompt_level import spotlight_wrap
from ..schema import Event

WINDOW_CHARS = 10_000
PAYLOAD_RESERVE = 500
BUDGET = WINDOW_CHARS - PAYLOAD_RESERVE

_WIDEST = spotlight_wrap("0" * 8)


def event_chars(event: Event) -> int:
    return len(event.render(_WIDEST)) + 1


def window_chars(events: list[Event]) -> int:
    return sum(event_chars(e) for e in events)


def fit_window(events: list[Event], rng: random.Random, budget: int = BUDGET) -> list[Event]:
    """Drop events at random, keeping order, until the window fits the budget.
    One event is always kept, so a single event over the budget is returned as is."""
    sizes = [event_chars(e) for e in events]
    keep = list(range(len(events)))
    total = sum(sizes)
    while total > budget and len(keep) > 1:
        total -= sizes[keep.pop(rng.randrange(len(keep)))]
    return [events[i] for i in keep]


def leading_window(events: list[Event], budget: int = BUDGET) -> list[Event]:
    """The longest run of leading events that fits the budget (at least one)."""
    out: list[Event] = []
    total = 0
    for e in events:
        size = event_chars(e)
        if out and total + size > budget:
            break
        out.append(e)
        total += size
    return out

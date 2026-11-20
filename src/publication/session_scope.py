"""
Session registry - which variants each session is authorised to run.

The variant trainer refuses to train unless the caller names a registered
session AND that session approved the variant. Centralising it here means a
mistyped variant, or a Session 1 cell re-submitted during Session 2, is refused
with a specific reason rather than silently running and quietly contaminating a
matrix that is supposed to be one-component-at-a-time.

Adding a session means adding a config module and registering it below. No
session may claim a variant another session already owns.
"""

from __future__ import annotations

import importlib

# token -> config module name
SESSIONS = {
    "session_1": "session1_config",
    "session_2": "session2_config",
}


def load(token: str):
    if token not in SESSIONS:
        raise KeyError("unknown session %r; registered: %s" % (token, sorted(SESSIONS)))
    return importlib.import_module(SESSIONS[token])


def approved_variants(token: str) -> list:
    return list(load(token).APPROVED_VARIANTS)


def eval_only_variants(token: str) -> list:
    """Variants a session runs WITHOUT training, by loading existing weights."""
    return list(getattr(load(token), "EVAL_ONLY_VARIANTS", []))


def is_in_scope(token: str, variant: str) -> tuple:
    """(allowed, reason) for a TRAINING cell in this session."""
    return load(token).is_in_scope(variant)


def owning_session(variant: str):
    """Which session, if any, owns this variant. Used to give a refusal that
    tells the operator where the variant actually belongs."""
    for token in SESSIONS:
        try:
            mod = load(token)
        except Exception:
            continue
        if variant in getattr(mod, "APPROVED_VARIANTS", []) or \
                variant in getattr(mod, "EVAL_ONLY_VARIANTS", []):
            return token
    return None

"""Deprecated name: use `epa.arena` (python/epa/arena.py)."""
import epa.arena as _m

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})

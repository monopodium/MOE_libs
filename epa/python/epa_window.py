"""Deprecated name: use `epa.window` (python/epa/window.py)."""
import epa.window as _m

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})

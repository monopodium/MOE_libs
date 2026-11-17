"""Deprecated name: use `epa.build` (python/epa/build.py)."""
import epa.build as _m

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})

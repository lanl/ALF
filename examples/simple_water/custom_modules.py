"""Deprecated compatibility aliases for the former water Parsl module."""

import warnings

warnings.warn(
    "examples/simple_water/custom_modules.py is deprecated; import parsl_configs instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from .parsl_configs import config_1node, config_debug
except ImportError:
    from parsl_configs import config_1node, config_debug

__all__ = ["config_1node", "config_debug"]

"""Deprecated compatibility aliases for the former molten-salt Parsl module."""

import warnings

warnings.warn(
    "examples/molten_salt/custom_modules.py is deprecated; import parsl_configs instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from .parsl_configs import config_1node, config_debug
except ImportError:
    from parsl_configs import config_1node, config_debug

config_running = config_1node

__all__ = ["config_1node", "config_running", "config_debug"]

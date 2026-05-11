"""Trading Agent — research + execution platform.

Public API is intentionally minimal at the package level. Import submodules
directly (``from trading.core.types import Bar``) rather than relying on
re-exports here; this keeps import-time cost low for the CLI.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trading-agent")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]

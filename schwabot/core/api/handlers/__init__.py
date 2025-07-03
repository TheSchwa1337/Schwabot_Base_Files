"""API Handler Package

This subpackage contains concrete third-party API handlers used by
`CacheSyncService`.  New handlers should inherit from `core.api.handlers.base_handler.BaseAPIHandler`.
"""

from importlib import import_module as _imp
from pathlib import Path as _Path

# Ensure that when the package is imported standalone, all modules are
# loaded so that `inspect.getmembers` in CacheSyncService can discover
# subclasses of BaseAPIHandler without needing to import them manually.
_pkg_path = _Path(__file__).parent
for _py in _pkg_path.glob("*.py"):
    if _py.name in {"__init__.py", "base_handler.py"} or _py.name.startswith("_"):
        continue
    _imp(f"{__name__}.{_py.stem}")

del _imp, _Path, _pkg_path, _py  # Cleanup namespace

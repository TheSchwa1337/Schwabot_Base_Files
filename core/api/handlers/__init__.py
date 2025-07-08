import importlib
from pathlib import Path as _Path
from pkgutil import iter_modules as _iter_modules

"""API Handler Package







This subpackage contains concrete third-party API handlers used by



`CacheSyncService`.  New handlers should inherit from
`core.api.handlers.base_handler.BaseAPIHandler`.



"""

# Ensure that when the package is imported standalone, all modules are


# loaded so that `inspect.getmembers` in CacheSyncService can discover


# subclasses of BaseAPIHandler without needing to import them manually.


_pkg_path = _Path(__file__).parent


# Import all modules in this directory
for _, _module_name, _ in _iter_modules([_pkg_path]):
    if not _module_name.startswith("_"):
        importlib.import_module(".{0}".format(_module_name), __package__)


del importlib, _Path, _pkg_path  # Cleanup namespace

import json
import os
import re
import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_phantom_math.py

Scans the codebase for phantom math routines defined in bucket_registry.json
and injects stub definitions for any missing functions, marked with TODO comments.
"""


REGISTRY_FILE = "bucket_registry.json"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

STUB_TEMPLATE = '''
# ⚠️ PHANTOM_MATH: stub generated for missing function {func_name}
    def {func_name}(*args, **kwargs):
    """PHANTOM stub: {func_name} - implement tensor relay logic"""
    raise NotImplementedError("Phantom math routine '{func_name}' not implemented")
'''


def load_registry(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_stubs(module_path, funcs):
    full_path = os.path.join(BASE_DIR, module_path)
    if not os.path.isfile(full_path):
        print(f"[WARN] Module not found: {module_path}")
        return

    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read()

    missing = []
    for sig in funcs:
        # signature format: path::func
        _, func_name = sig.split("::")
        # simple regex to detect function definition
        if not re.search(rf"^\s*def\s+{func_name}\b", text, flags=re.MULTILINE):
            missing.append(func_name)

    if not missing:
        return

    # Append stubs at end of file
    with open(full_path, "a", encoding="utf-8") as f:
        f.write("\n# ---- Phantom math stubs ----\n")
        for name in missing:
            stub = STUB_TEMPLATE.format(func_name=name)
            f.write(stub)
        f.write("\n")

    print(f"[INFO] Injected {len(missing)} phantom stubs into {module_path}")


def main():
    if not os.path.isfile(REGISTRY_FILE):
        print(f"[ERROR] Registry file not found: {REGISTRY_FILE}")
        sys.exit(1)

    registry = load_registry(REGISTRY_FILE)
    for bucket, sigs in registry.items():
        for sig in sigs:
            module_path, _ = sig.split("::")
            # group by module
        # inject per module
        modules = {}
        for sig in sigs:
            mod, func = sig.split("::")
            modules.setdefault(mod, []).append(sig)

        for mod, funcs in modules.items():
            inject_stubs(mod, funcs)


if __name__ == "__main__":
    main()

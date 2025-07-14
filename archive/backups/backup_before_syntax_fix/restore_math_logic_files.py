import shutil
from pathlib import Path

#!/usr/bin/env python3
"""
Restore selected math/logic files from backup to a clean directory, preserving structure.
"""


# List of files to restore (relative to cleanup_backup/core)
FILES_TO_RESTORE = []
    "mathlib_v3.py",
    "mathlib_v3_visualizer.py",
    "mathlib_v2.py",
    "advanced_mathematical_core.py",
    "advanced_drift_shell_integration.py",
    "matrix/fault_resolver.py",
    "matrix/strategy_matrix.py",
    "utils/math_utils.py",
    "utils/cli_handler.py",
    "utils/rate_limiter.py",
    "utils/yaml_config_loader.py",
    "phase_engine/__init__.py",
    "phase_engine/phase_loader.py",
    "phase_engine/phase_logger.py",
    "phase_engine/phase_map.py",
    "phase_engine/phase_metrics_engine.py",
    "phase_engine/sha_mapper.py",
    "phase_engine/swap_controller.py",
    "profit/cycle_allocator.py",
    "ghost/ghost_conditionals.py",
    "ghost/ghost_phase_integrator.py",
    "ghost/ghost_news_vectorizer.py",
    "lantern/vector_memory.py",
    "memory_stack/ai_command_sequencer.py",
    "memory_stack/command_density_analyzer.py",
    "memory_stack/execution_validator.py",
    "memory_stack/memory_hash_rotator.py",
    "memory_stack/memory_key_allocator.py",
    "memory_stack/trust_feedback_updater.py",
    "phantom/entry_logic.py",
    "phantom/exit_logic.py",
    "phantom/price_vector_synchronizer.py",
]
SRC_ROOT = Path("cleanup_backup/core")
DST_ROOT = Path("core_math_restore")


def restore_files():
    for rel_path in FILES_TO_RESTORE:
        src = SRC_ROOT / rel_path
        dst = DST_ROOT / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
        else:
            print(f"WARNING: {src} does not exist and was skipped.")


if __name__ == "__main__":
    restore_files()
    print()
        "\nRestore complete. You can now autopep8, flake8, and mypy the core_math_restore directory."
    )

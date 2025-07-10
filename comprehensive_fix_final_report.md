# Comprehensive Fix Final Report

## Summary
- Total execution time: 41.96 seconds
- Scripts run: 4
- Successful scripts: 3
- Failed scripts: 1

## Detailed Results

### fix_syntax_errors_comprehensive.py: [SUCCESS]
STDOUT:
```
2025-07-10 06:52:16,112 - INFO - Starting comprehensive syntax error fix
2025-07-10 06:52:16,112 - INFO - Scanning directory: core
2025-07-10 06:52:16,113 - INFO - Processing core\acceleration_enhancement.py
2025-07-10 06:52:16,115 - INFO - Backed up core\acceleration_enhancement.py to backup_before_syntax_fix\acceleration_enhancement.py
2025-07-10 06:52:16,116 - INFO - Fixed unmatched brace in core\acceleration_enhancement.py:132
2025-07-10 06:52:16,116 - INFO - Fixed unmatched brace in core\acceleration_enhancement.py:176
2025-07-10 06:52:16,116 - INFO - Fixed unexpected indent in core\acceleration_enhancement.py:41 (removed 4 spaces)
2025-07-10 06:52:16,116 - INFO - Fixed unexpected indent in core\acceleration_enhancement.py:42 (removed 4 spaces)
2025-07-10 06:52:16,117 - INFO - Fixed unexpected indent in core\acceleration_enhancement.py:43 (removed 4 spaces)
2025-07-10 06:52:16,117 - INFO - Fixed unexpected indent in core\acceleration_enhancement.py:95 (removed 4 spaces)
2025-07-10
... (truncated)
```

### fix_imports_comprehensive.py: [SUCCESS]
STDOUT:
```
2025-07-10 06:52:27,505 - INFO - Starting comprehensive import fix
2025-07-10 06:52:27,505 - INFO - Scanning directory: core
2025-07-10 06:52:27,506 - INFO - Processing core\acceleration_enhancement.py
2025-07-10 06:52:27,508 - INFO - Backed up core\acceleration_enhancement.py to backup_before_import_fix\acceleration_enhancement.py
2025-07-10 06:52:27,514 - INFO - Found undefined names in core\acceleration_enhancement.py: {'timeout', 'timestamp', 'e', 'self', 'config', 'success', 'error', 'retries', 'debug', 'enabled'}
2025-07-10 06:52:27,517 - INFO - Fixed imports in core\acceleration_enhancement.py: ['import logging', 'import logging']
2025-07-10 06:52:27,517 - INFO - Processing core\advanced_dualistic_trading_execution_system.py
2025-07-10 06:52:27,518 - INFO - Backed up core\advanced_dualistic_trading_execution_system.py to backup_before_import_fix\advanced_dualistic_trading_execution_system.py
2025-07-10 06:52:27,523 - INFO - Found undefined names in core\advanced_dualistic_tradin
... (truncated)
```

### update_requirements_comprehensive.py: [SUCCESS]
STDOUT:
```
2025-07-10 06:52:33,818 - INFO - Starting comprehensive requirements update
2025-07-10 06:52:33,818 - INFO - Updating requirements.txt
2025-07-10 06:52:33,828 - INFO - Backed up requirements.txt to requirements.txt.backup
2025-07-10 06:52:33,898 - INFO - Updated requirements.txt with 61 dependencies
2025-07-10 06:52:33,899 - INFO - Created requirements-windows.txt
2025-07-10 06:52:33,900 - INFO - Created requirements-linux.txt
2025-07-10 06:52:33,900 - INFO - Created requirements-darwin.txt
2025-07-10 06:52:33,902 - INFO - Created setup scripts: setup_windows.bat, setup_unix.sh
2025-07-10 06:52:33,902 - INFO - Report generated: requirements_update_report.md
2025-07-10 06:52:33,903 - INFO - Requirements update completed

```

### test_comprehensive_fixes.py: [FAILED]
STDOUT:
```
Testing comprehensive fixes...
==================================================
[PASS] core.strategy_bit_mapper
[FAIL] core.matrix_mapper: No module named 'core.matrix_mapper'
[FAIL] core.trading_strategy_executor: cannot import name 'np' from 'numpy' (C:\Users\maxde\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\__init__.py)
[PASS] core.schwabot_rheology_integration
[FAIL] core.orbital_shell_brain_system: cannot import name 'np' from 'numpy' (C:\Users\maxde\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\__init__.py)
[FAIL] core.zpe_core: cannot import name 'np' from 'numpy' (C:\Users\maxde\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\__init__.py)
[FAIL] core.zbe_core: cannot import name 'np' from 'numpy' (C:\Users\maxde\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\__init__.py)
[PASS] core\acceleration_enhancement.py
[PASS] core\advanced_dualistic_trading_execution_system.py
[PASS] core\advanced_risk_manager.py
[
... (truncated)
```
STDERR:
```
2025-07-10 06:52:35.914611: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-10 06:52:37.390337: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Loss anticipation curve not available
Flip switch logic lattice not available

```

## Recommendations

[WARNING] Some fixes failed. Please review the detailed results above.
- Check the logs in the `logs/` directory for more details
- Manual intervention may be required for failed scripts
- Consider running individual fix scripts to isolate issues

## Next Steps

1. Review the generated reports
2. Test the system functionality
3. Run additional code quality checks if needed
4. Deploy the fixed system

# CLI and Fault Handling Quick Reference

## 🚀 **SCHWABOT v0.42f**

### **Quick Start**

```python
# Import the centralized CLI handler
from core.utils.windows_cli_compatibility import (
    WindowsCliCompatibilityHandler,
    safe_print,
    safe_format_error,
    log_safe,
    cli_handler,
)

# Use safe print for CLI-safe output
safe_print("🚀 System ready")  # Outputs: "[LAUNCH] System ready"

# Use safe error formatting
try:
    # Your code here
    pass
except Exception as e:
    error_msg = safe_format_error(e, "context")
    safe_print(f"❌ Error: {error_msg}")

# Use safe logging
import logging
logger = logging.getLogger("my_module")
log_safe(logger, "info", "📊 Data processed")
```

---

## 🔧 **Core Functions**

### **safe_print(message, use_emoji=True)**
- Converts emojis to ASCII text
- Handles Unicode characters safely
- Works on all platforms including Windows CLI

### **safe_format_error(error, context="")**
- Safely formats error messages
- Adds context information
- Prevents information leakage

### **log_safe(logger, level, message)**
- Safe logging with Unicode handling
- Supports all logging levels
- CLI-compatible output

### **WindowsCliCompatibilityHandler.is_windows_cli()**
- Detects Windows CLI environment
- Returns True/False for conditional logic

---

## 🎯 **Common Patterns**

### **Entry Vector Stabilization**
```python
safe_print("🚀 ENTRY VECTOR STABILIZED - echo safe")
safe_print("📊 Market data received")
safe_print("🎯 Target price calculated")
```

### **Fault Bus Integration**
```python
from core.fault_bus import FaultBus, FaultType, FaultBusEvent

fault_bus = FaultBus()
event = FaultBusEvent(
    tick=1,
    module="my_module",
    type=FaultType.THERMAL_HIGH,
    severity=0.6,
    metadata={"message": "🚀 Test event"},
    profit_context=15.0,
)
fault_bus.push(event)
```

### **Error Handling**
```python
try:
    # Risky operation
    result = complex_calculation()
except Exception as e:
    error_msg = safe_format_error(e, "complex_calculation")
    safe_print(f"❌ Calculation failed: {error_msg}")
    
    # Report to fault bus
    fault_event = FaultBusEvent(
        tick=current_tick,
        module="calculation_module",
        type=FaultType.PROFIT_ANOMALY,
        severity=0.8,
        metadata={"error": error_msg},
        profit_context=0.0,
    )
    fault_bus.push(fault_event)
```

---

## 📊 **Emoji Mappings**

| Emoji | ASCII | Usage |
|-------|-------|-------|
| ✅ | [SUCCESS] | Successful operations |
| ❌ | [ERROR] | Errors and failures |
| 🚀 | [LAUNCH] | System startup, launches |
| 📊 | [DATA] | Data processing, analytics |
| 🎯 | [TARGET] | Targets, goals, objectives |
| 🔥 | [HOT] | High activity, thermal |
| ⚡ | [FAST] | Fast operations, speed |
| 💰 | [PROFIT] | Profit, financial data |
| 🔧 | [TOOL] | Tools, utilities, fixes |
| ⚠️ | [WARNING] | Warnings, cautions |

---

## 🔍 **Validation Commands**

### **Quick Validation**
```bash
python tools/final_validation_check.py
```

### **CLI Echo Integration Test**
```bash
python tools/cli_echo_integration_test.py
```

### **Injection Points Validation**
```bash
python tools/validate_cli_injection_points.py
```

### **Simple CLI Test**
```bash
python tools/simple_cli_test.py
```

---

## ⚙️ **Configuration**

### **YAML Settings**
```yaml
# config/settings.yaml
cli:
  emoji: false                    # Enable/disable emoji usage
  verbose: true                   # Enable verbose output
  mode: "safe"                    # CLI mode: "safe", "full", "minimal"
  unicode_fallback: true          # Enable Unicode fallbacks
  windows_detection: true         # Enable Windows CLI detection
```

### **Dynamic Control**
```python
import yaml

with open("config/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

if settings["cli"]["emoji"]:
    safe_print("🚀 TRADE READY")
else:
    safe_print("[TRADE READY]")
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Import Error: CLI handler not available**
   ```python
   # Check if file exists
   import os
   if os.path.exists("core/utils/windows_cli_compatibility.py"):
       # File exists, check import path
       from core.utils.windows_cli_compatibility import safe_print
   ```

2. **Unicode Error on Windows CLI**
   ```python
   # Use safe_print instead of print
   safe_print("🚀 Message with emoji")  # ✅ Safe
   print("🚀 Message with emoji")       # ❌ May fail
   ```

3. **Fault Bus Integration Issues**
   ```python
   # Ensure proper import
   from core.fault_bus import FaultBus, FaultType, FaultBusEvent
   
   # Initialize fault bus
   fault_bus = FaultBus()
   
   # Create events with CLI-safe metadata
   event = FaultBusEvent(
       tick=1,
       module="test",
       type=FaultType.THERMAL_HIGH,
       severity=0.5,
       metadata={"message": "🚀 Test"},  # Use safe_print for CLI safety
       profit_context=0.0,
   )
   ```

---

## 📈 **Performance Tips**

### **Optimization**
- Use `safe_print` only when needed
- Cache CLI handler instance
- Use appropriate logging levels
- Minimize Unicode operations in hot paths

### **Monitoring**
```python
# Monitor CLI performance
import time

start_time = time.time()
safe_print("🚀 Performance test")
end_time = time.time()
print(f"CLI operation took: {end_time - start_time:.4f}s")
```

---

## 🔐 **Security Considerations**

### **Error Handling**
- Never expose raw exceptions in production
- Use `safe_format_error` for all error messages
- Add context to error messages
- Log errors appropriately

### **CLI Safety**
- Always use `safe_print` for user-facing output
- Test on Windows CLI environment
- Validate Unicode handling
- Use fallback mechanisms

---

## 🎉 **Success Indicators**

### **Validation Checklist**
- ✅ All imports work correctly
- ✅ No Unicode errors on Windows CLI
- ✅ Fault bus integration functional
- ✅ Error handling standardized
- ✅ Emoji conversion working
- ✅ Cross-platform compatibility verified

### **Deployment Ready**
- ✅ CLI compatibility verified
- ✅ Fault handling standardized
- ✅ Error formatting consistent
- ✅ Cross-platform support confirmed
- ✅ Code quality maintained
- ✅ Testing framework complete

---

## 📞 **Support**

### **Quick Help**
1. Run `python tools/final_validation_check.py`
2. Check the validation report
3. Review any failed tests
4. Fix issues and re-run validation

### **Files to Check**
- `core/utils/windows_cli_compatibility.py` - Main CLI handler
- `core/fault_bus.py` - Fault bus with CLI integration
- `config/settings.yaml` - Configuration settings
- `tools/` - Validation and testing tools

---

**🔥 Schwabot v0.42f is CLI-battle-tested, fault-verified, and deployment-rigged.** 
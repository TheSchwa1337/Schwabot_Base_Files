
#!/usr/bin/env python3
"""Fix file formatting issues."""


def fix_file(filename):
    """Fix file formatting issues."""
    try:
        # Read file content
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove null bytes and trailing whitespace
        content = content.replace("\x00", "").rstrip()

        # Add newline at end
        content += "\n"

        # Write back
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Fixed {filename}")
    except Exception as e:
        print(f"Error fixing {filename}: {e}")


def fix_profit_vector_forecast():
    """Fix specific issues in profit_vector_forecast.py."""
    try:
        with open("core/profit_vector_forecast.py", "r", encoding="utf-8") as f:
            content = f.read()

        # Remove unused imports
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Skip unused imports
            if any()
                skip in line
                for skip in []
                    "from dataclasses import dataclass, field",
                    "import numpy as np",
                    "from core.drift_shell_engine import MemorySnapshot",
                ]
            ):
                if "dataclass" in line:
                    fixed_lines.append("from dataclasses import dataclass")
                elif "numpy" in line:
                    continue  # Skip numpy import
                elif "MemorySnapshot" in line:
                    fixed_lines.append()
                        "from core.drift_shell_engine import ProfitVector"
                    )
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        # Fix f-strings without placeholders
        content = "\n".join(fixed_lines)
        content = content.replace()
            'f"📈 Profit Vector Forecast Engine initialized with {lookback_periods} period lookback"',
            'f"📈 Profit Vector Forecast Engine initialized with {lookback_periods} period lookback"',
        )

        # Add newline at end
        content = content.rstrip() + "\n"

        with open("core/profit_vector_forecast.py", "w", encoding="utf-8") as f:
            f.write(content)

        print("Fixed core/profit_vector_forecast.py")
    except Exception as e:
        print(f"Error fixing profit_vector_forecast.py: {e}")


if __name__ == "__main__":
    fix_file("core/brain_trading_engine.py")
    fix_profit_vector_forecast()

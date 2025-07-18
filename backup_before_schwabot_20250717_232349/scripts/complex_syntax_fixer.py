import re
import subprocess
import sys
from pathlib import Path

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex Syntax Fixer for Schwabot.

Targets specific complex syntax issues:
- Missing try: statements
- Indentation errors
- Unterminated strings
- Malformed function definitions
"""


class ComplexSyntaxFixer:
    """Fixes complex syntax errors while preserving mathematical content."""

    def __init__(self):
        """Initialize the fixer."""
        self.fixed_files = []

    def fix_missing_try_statements(): -> str:
        """Fix missing try: statements in functions."""
        lines = content.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check for function definitions followed by try without try:
            if ()
                line.startswith("def ")
                and i + 1 < len(lines)
                and lines[i + 1].strip().startswith("try")
            ):
                # Add missing try: statement
                fixed_lines.append(lines[i])
                fixed_lines.append("        try:")
                i += 1
                continue

            # Check for other missing try: patterns
            if line.startswith("try") and not line.endswith(":"):
                fixed_lines.append(line + ":")
            else:
                fixed_lines.append(lines[i])

            i += 1

        return "\n".join(fixed_lines)

    def fix_indentation_errors(): -> str:
        """Fix indentation errors in the code."""
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Fix common indentation issues
            if line.strip().startswith("return ") and not line.startswith("        "):
                line = "        " + line.strip()
            elif line.strip().startswith("except ") and not line.startswith("        "):
                line = "        " + line.strip()
            elif line.strip().startswith("logger.") and not line.startswith()
                "            "
            ):
                line = "            " + line.strip()

            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def fix_unterminated_strings(): -> str:
        """Fix unterminated string literals."""
        # Fix common unterminated string patterns
        content = re.sub(r"'([^']*)$", r"'\1'", content, flags=re.MULTILINE)
        content = re.sub(r'"([^"]*)$', r'"\1"', content, flags=re.MULTILINE)
        return content

    def fix_function_definitions(): -> str:
        """Fix malformed function definitions."""
        # Fix triple colons
        content = re.sub(r":::", ":", content)
        # Fix missing colons
        content = re.sub()
            r"def [^:]+$", lambda m: m.group(0) + ":", content, flags=re.MULTILINE
        )
        return content

    def fix_specific_file(): -> bool:
        """Fix a specific file with complex syntax errors."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply fixes
            content = self.fix_missing_try_statements(content)
            content = self.fix_indentation_errors(content)
            content = self.fix_unterminated_strings(content)
            content = self.fix_function_definitions(content)

            # Check if content changed
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # Verify fix
                result = subprocess.run()
                    [sys.executable, "-m", "py_compile", file_path],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    print(f"✅ Fixed: {file_path}")
                    self.fixed_files.append(file_path)
                    return True
                else:
                    print(f"⚠️  Partial fix: {file_path}")
                    return False

            return False

        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
            return False

    def run_fixes(self):
        """Run fixes on all Python files in core directory."""
        print("🔧 Complex Syntax Fixer - Targeting Advanced Issues")
        print("=" * 60)

        core_files = list(Path("core").rglob("*.py"))
        total_files = len(core_files)
        fixed_count = 0

        for file_path in core_files:
            if self.fix_specific_file(str(file_path)):
                fixed_count += 1

        print("\n📊 Results:")
        print(f"   Total files processed: {total_files}")
        print(f"   Files fixed: {fixed_count}")
        print(f"   Success rate: {(fixed_count / total_files) * 100:.1f}%")

        if self.fixed_files:
            print("\n✅ Successfully fixed files:")
            for f in self.fixed_files[:10]:  # Show first 10
                print(f"   - {f}")
            if len(self.fixed_files) > 10:
                print(f"   ... and {len(self.fixed_files) - 10} more")


if __name__ == "__main__":
    fixer = ComplexSyntaxFixer()
    fixer.run_fixes()

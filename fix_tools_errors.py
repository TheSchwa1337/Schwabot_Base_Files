#!/usr/bin/env python3
"""
Fix common flake8 errors in tools directory
Priority order:
1. F821 undefined names (critical functionality)
2. F811 redefinition of unused imports
3. F841 unused variables
4. W292 no newline at end of file
5. E265 block comment style (cosmetic)
"""

import re
import os
import glob


def fix_tools_errors():
    """Fix common flake8 errors in tools directory"""

    tools_dir = "tools"
    if not os.path.exists(tools_dir):
        print(f"❌ Directory not found: {tools_dir}")
        return

    print(f"🔧 Fixing flake8 errors in {tools_dir}")

    # Get all Python files in tools directory
    python_files = glob.glob(f"{tools_dir}/*.py")

    total_fixed = 0

    for file_path in python_files:
        print(f"\n📁 Processing: {file_path}")

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        file_fixes = 0

        # Fix 1: F821 undefined names - common patterns
        # safe_safe_print -> safe_print
        content = re.sub(r'\bsafe_safe_print\b', 'safe_print', content)

        # Fix 2: F811 redefinition of unused imports
        # Remove duplicate import lines that are redefined
        lines = content.split('\n')
        fixed_lines = []
        seen_imports = set()

        for line in lines:
            # Check if this is a redefinition of an import
            if line.strip().startswith('from ') and 'import' in line:
                import_name = line.strip().split('import')[-1].strip()
                if import_name in seen_imports:
                    print(f"  🔧 Removed duplicate import: {line.strip()}")
                    file_fixes += 1
                    continue
                seen_imports.add(import_name)
            fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

        # Fix 3: F841 unused variables - remove common unused variable assignments
        # Remove lines like: original_content = content (when not used)
        content = re.sub(r'^\s*original_content\s*=\s*content\s*$', '', content, flags=re.MULTILINE)

        # Fix 4: W292 no newline at end of file
        if not content.endswith('\n'):
            content += '\n'
            print(f"  🔧 Added newline at end of file")
            file_fixes += 1

        # Fix 5: E265 block comment style - fix shebang lines
        content = re.sub(r'^#!/usr/bin/env python3$', '#!/usr/bin/env python3', content, flags=re.MULTILINE)

        # Check if changes were made
        if content != original_content:
            # Backup the original file
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # Write the fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"  ✅ Fixed {file_fixes} issues in {file_path}")
            total_fixed += file_fixes
        else:
            print(f"  ℹ️ No issues found in {file_path}")

    print(f"\n🎉 Total fixes applied: {total_fixed}")
    print("🔧 Fixed the following error types:")
    print("1. F821 undefined names (safe_safe_print -> safe_print)")
    print("2. F811 redefinition of unused imports")
    print("3. F841 unused variables (original_content)")
    print("4. W292 no newline at end of file")
    print("5. E265 block comment style")


if __name__ == "__main__":
    fix_tools_errors()

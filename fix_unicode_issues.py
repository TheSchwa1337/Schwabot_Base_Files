from pathlib import Path

#!/usr/bin/env python3
"""
Fix Unicode character issues in Python files.
Replaces Unicode characters with ASCII equivalents to prevent syntax errors.
"""


# Unicode to ASCII character mappings
UNICODE_REPLACEMENTS = {}
    "∈": "in",
    "√": "sqrt",
    "π": "pi",
    "Σ": "sum",
    "λ": "lambda",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "∧": "and",
    "∨": "or",
    "¬": "not",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "∀": "forall",
    "∃": "exists",
    "∞": "infinity",
    "±": "+/-",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "≡": "==",
    "≅": "cong",
    "⊂": "subset",
    "⊃": "superset",
    "⊆": "subseteq",
    "⊇": "superseteq",
    "∪": "union",
    "∩": "intersection",
    "∅": "empty",
    "ℕ": "N",
    "ℤ": "Z",
    "ℚ": "Q",
    "ℝ": "R",
    "ℂ": "C",
}


def fix_unicode_in_file(): -> bool:
    """Fix Unicode characters in a single file."""
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        original_content = content

        # Replace Unicode characters
        for unicode_char, ascii_replacement in UNICODE_REPLACEMENTS.items():
            content = content.replace(unicode_char, ascii_replacement)

        # Only write if changes were made
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix Unicode issues in all Python files."""
    print("🔧 Fixing Unicode character issues...")
    print("=" * 50)

    # Find all Python files in core directory
    core_dir = Path("core")
    python_files = list(core_dir.rglob("*.py"))

    if not python_files:
        print("No Python files found in core directory!")
        return

    print(f"Found {len(python_files)} Python files to check")

    fixed_count = 0
    for file_path in python_files:
        print(f"Checking: {file_path}")
        if fix_unicode_in_file(file_path):
            print(f"✅ Fixed Unicode issues in: {file_path}")
            fixed_count += 1

    print()
        f"\n🎉 Fixed Unicode issues in {fixed_count} out of {len(python_files)} files!"
    )

    if fixed_count > 0:
        print("\n📋 Next steps:")
        print("1. Test imports: python -c 'import core.strategy_loader'")
        print("2. Run: flake8 core/ to check for remaining errors")
        print("3. Verify functionality is preserved")


if __name__ == "__main__":
    main()

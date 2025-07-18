#!/usr/bin/env python3
"""
Syntax Error Fixer for Schwabot Core Files.

This script systematically fixes common syntax errors in Python files
before applying black formatting.
"""

import re
from pathlib import Path


def fix_common_syntax_errors(content: str) -> str:
    """Fix common syntax errors in Python code."""
    # Fix unterminated string literals at the beginning
    content = re.sub(r'^"([^"]*)"', r'"""\1"""', content, flags=re.MULTILINE)"

    # Fix unterminated string literals
    content = re.sub(r'""""""\s*""""""\s*"""', '"""', content)
    content = re.sub(r'"""\s*"""\s*"""', '"""', content)
    content = re.sub(r'"""\s*"""', '"""', content)"

    # Fix malformed float literals (e.g., 1e - 6 -> 1e-6)
    content = re.sub(r"(\d+e)\s*-\s*(\d+)", r"\1-\2", content)

    # Fix malformed class definitions
    content = re.sub(r'class\s+(\w+):"', r"class \1:", content)"
    content = re.sub(r'class\s+(\w+):\s*"', r"class \1:", content)"

    # Fix malformed function definitions
    content = re.sub()
        r'def\s+(\w+)\s*\([^)]*\):\s*->\s*(\w+):"', r"def \1() -> \2:", content"
    )
    content = re.sub(r'def\s+(\w+)\s*\([^)]*\):\s*"', r"def \1():", content)"
    content = re.sub(r'def\s+(\w+)\s*\([^)]*\):\s*"', r"def \1():", content)"

    # Fix malformed dataclass field definitions
    content = re.sub(r'(\w+):\s*(\w+)\s*=\s*([^"]+)"', r"\1: \2 = \3", content)

    # Fix malformed docstrings
    content = re.sub(r'"""([^"]*)"\s*"""', r'"""\1"""', content)

    # Fix malformed imports
    content = re.sub(r'from\s+([^"]+)"', r"from \1", content)
    content = re.sub(r'import\s+([^"]+)"', r"import \1", content)

    # Fix malformed assignments
    content = re.sub(r'(\w+)\s*=\s*([^"]+)"', r"\1 = \2", content)

    # Fix malformed return statements
    content = re.sub(r'return\s+([^"]+)"', r"return \1", content)

    # Fix malformed if/elif/else statements
    content = re.sub(r'(if|elif|else)\s*([^:]*):"', r"\1 \2:", content)"

    # Fix malformed try/except blocks
    content = re.sub(r'(try|except|finally):"', r"\1:", content)"

    # Fix malformed with statements
    content = re.sub(r'with\s+([^:]*):"', r"with \1:", content)"

    # Fix malformed for/while loops
    content = re.sub(r'(for|while)\s+([^:]*):"', r"\1 \2:", content)"

    # Fix malformed list/dict comprehensions
    content = re.sub(r'(\[|\{)\s*([^"\]]*)"', r"\1\2", content)}

    # Fix malformed function calls
    content = re.sub(r'(\w+)\s*\(([^"]*)\)"', r"\1(\2)", content)

    # Fix malformed attribute access
    content = re.sub(r'(\w+)\.(\w+)"', r"\1.\2", content)"

    # Fix malformed string literals
    content = re.sub(r'"([^"]*)"\s*"', r'"\1"', content)
    content = re.sub(r"'([^']*)'\s*'", r"'\1'", content)

    # Fix malformed comments
    content = re.sub(r'#\s*([^"]*)"', r"# \1", content)

    # Fix malformed whitespace
    content = re.sub(r'\s+"', "", content)"
    content = re.sub(r'"\s+', "", content)"

    # Fix malformed equals signs
    content = re.sub(r"=\s*=\s*=", r"===", content)
    content = re.sub(r"=\s*=", r"==", content)

    # Fix malformed colons
    content = re.sub(r':\s*"', r":", content)"

    # Fix malformed parentheses
    content = re.sub(r'\(\s*"', r"(", content)"))
    content = re.sub(r'"\s*\)', r")", content)"

    # Fix malformed brackets
    content = re.sub(r'\[\s*"', r"[", content)"]]
    content = re.sub(r'"\s*\]', r"]", content)"

    # Fix malformed braces
    content = re.sub(r'\{\s*"', r"{", content)"}}
    content = re.sub(r'"\s*\}', r"}", content)"

    # Fix malformed commas
    content = re.sub(r',\s*"', r",", content)"

    # Fix malformed periods
    content = re.sub(r'\.\s*"', r".", content)"

    # Fix malformed semicolons
    content = re.sub(r';\s*"', r";", content)"

    # Fix malformed newlines
    content = re.sub(r'\n\s*"', r"\n", content)"

    # Fix malformed tabs
    content = re.sub(r'\t\s*"', r"\t", content)"

    # Fix malformed spaces
    content = re.sub(r' \s*"', r" ", content)"

    # Fix malformed underscores
    content = re.sub(r'_\s*"', r"_", content)"

    # Fix malformed hyphens
    content = re.sub(r'-\s*"', r"-", content)"

    # Fix malformed plus signs
    content = re.sub(r'\+\s*"', r"+", content)"

    # Fix malformed minus signs
    content = re.sub(r'-\s*"', r"-", content)"

    # Fix malformed asterisks
    content = re.sub(r'\*\s*"', r"*", content)"

    # Fix malformed slashes
    content = re.sub(r'/\s*"', r"/", content)"

    # Fix malformed backslashes
    content = re.sub(r'\\\s*"', r"\\", content)"

    # Fix malformed pipes
    content = re.sub(r'\|\s*"', r"|", content)"

    # Fix malformed ampersands
    content = re.sub(r'&\s*"', r"&", content)"

    # Fix malformed carets
    content = re.sub(r'\^\s*"', r"^", content)"

    # Fix malformed tildes
    content = re.sub(r'~\s*"', r"~", content)"

    # Fix malformed exclamation marks
    content = re.sub(r'!\s*"', r"!", content)"

    # Fix malformed question marks
    content = re.sub(r'\?\s*"', r"?", content)"

    # Fix malformed at signs
    content = re.sub(r'@\s*"', r"@", content)"

    # Fix malformed hash signs
    content = re.sub(r'#\s*"', r"#", content)"

    # Fix malformed dollar signs
    content = re.sub(r'\$\s*"', r"$", content)"

    # Fix malformed percent signs
    content = re.sub(r'%\s*"', r"%", content)"

    # Fix malformed greater than signs
    content = re.sub(r'>\s*"', r">", content)"

    # Fix malformed less than signs
    content = re.sub(r'<\s*"', r"<", content)"

    # Fix malformed greater than or equal signs
    content = re.sub(r'>=\s*"', r">=", content)"

    # Fix malformed less than or equal signs
    content = re.sub(r'<=\s*"', r"<=", content)"

    # Fix malformed not equal signs
    content = re.sub(r'!=\s*"', r"!=", content)"

    # Fix malformed assignment operators
    content = re.sub(r'\+=\s*"', r"+=", content)"
    content = re.sub(r'-=\s*"', r"-=", content)"
    content = re.sub(r'\*=\s*"', r"*=", content)"
    content = re.sub(r'/=\s*"', r"/=", content)"
    content = re.sub(r'%=\s*"', r"%=", content)"
    content = re.sub(r'\*\*=\s*"', r"**=", content)"
    content = re.sub(r'//=\s*"', r"//=", content)"
    content = re.sub(r'&=\s*"', r"&=", content)"
    content = re.sub(r'\|=\s*"', r"|=", content)"
    content = re.sub(r'\^=\s*"', r"^=", content)"
    content = re.sub(r'>>=\s*"', r">>=", content)"
    content = re.sub(r'<<=\s*"', r"<<=", content)"

    # Fix malformed logical operators
    content = re.sub(r'and\s*"', r"and", content)"
    content = re.sub(r'or\s*"', r"or", content)"
    content = re.sub(r'not\s*"', r"not", content)"

    # Fix malformed keywords
    content = re.sub(r'if\s*"', r"if", content)"
    content = re.sub(r'elif\s*"', r"elif", content)"
    content = re.sub(r'else\s*"', r"else", content)"
    content = re.sub(r'for\s*"', r"for", content)"
    content = re.sub(r'while\s*"', r"while", content)"
    content = re.sub(r'try\s*"', r"try", content)"
    content = re.sub(r'except\s*"', r"except", content)"
    content = re.sub(r'finally\s*"', r"finally", content)"
    content = re.sub(r'with\s*"', r"with", content)"
    content = re.sub(r'as\s*"', r"as", content)"
    content = re.sub(r'import\s*"', r"import", content)"
    content = re.sub(r'from\s*"', r"from", content)"
    content = re.sub(r'class\s*"', r"class", content)"
    content = re.sub(r'def\s*"', r"def", content)"
    content = re.sub(r'return\s*"', r"return", content)"
    content = re.sub(r'pass\s*"', r"pass", content)"
    content = re.sub(r'break\s*"', r"break", content)"
    content = re.sub(r'continue\s*"', r"continue", content)"
    content = re.sub(r'raise\s*"', r"raise", content)"
    content = re.sub(r'assert\s*"', r"assert", content)"
    content = re.sub(r'del\s*"', r"del", content)"
    content = re.sub(r'global\s*"', r"global", content)"
    content = re.sub(r'nonlocal\s*"', r"nonlocal", content)"
    content = re.sub(r'lambda\s*"', r"lambda", content)"
    content = re.sub(r'yield\s*"', r"yield", content)"
    content = re.sub(r'await\s*"', r"await", content)"
    content = re.sub(r'async\s*"', r"async", content)"

    # Fix malformed built-in functions
    content = re.sub(r'print\s*"', r"print", content)"
    content = re.sub(r'len\s*"', r"len", content)"
    content = re.sub(r'sum\s*"', r"sum", content)"
    content = re.sub(r'max\s*"', r"max", content)"
    content = re.sub(r'min\s*"', r"min", content)"
    content = re.sub(r'abs\s*"', r"abs", content)"
    content = re.sub(r'round\s*"', r"round", content)"
    content = re.sub(r'int\s*"', r"int", content)"
    content = re.sub(r'float\s*"', r"float", content)"
    content = re.sub(r'str\s*"', r"str", content)"
    content = re.sub(r'list\s*"', r"list", content)"
    content = re.sub(r'dict\s*"', r"dict", content)"
    content = re.sub(r'set\s*"', r"set", content)"
    content = re.sub(r'tuple\s*"', r"tuple", content)"
    content = re.sub(r'bool\s*"', r"bool", content)"
    content = re.sub(r'type\s*"', r"type", content)"
    content = re.sub(r'isinstance\s*"', r"isinstance", content)"
    content = re.sub(r'issubclass\s*"', r"issubclass", content)"
    content = re.sub(r'hasattr\s*"', r"hasattr", content)"
    content = re.sub(r'getattr\s*"', r"getattr", content)"
    content = re.sub(r'setattr\s*"', r"setattr", content)"
    content = re.sub(r'delattr\s*"', r"delattr", content)"
    content = re.sub(r'callable\s*"', r"callable", content)"
    content = re.sub(r'hash\s*"', r"hash", content)"
    content = re.sub(r'id\s*"', r"id", content)"
    content = re.sub(r'ord\s*"', r"ord", content)"
    content = re.sub(r'chr\s*"', r"chr", content)"
    content = re.sub(r'bin\s*"', r"bin", content)"
    content = re.sub(r'oct\s*"', r"oct", content)"
    content = re.sub(r'hex\s*"', r"hex", content)"
    content = re.sub(r'format\s*"', r"format", content)"
    content = re.sub(r'repr\s*"', r"repr", content)"
    content = re.sub(r'ascii\s*"', r"ascii", content)"
    content = re.sub(r'eval\s*"', r"eval", content)"
    content = re.sub(r'exec\s*"', r"exec", content)"
    content = re.sub(r'compile\s*"', r"compile", content)"
    content = re.sub(r'open\s*"', r"open", content)"
    content = re.sub(r'input\s*"', r"input", content)"
    content = re.sub(r'range\s*"', r"range", content)"
    content = re.sub(r'enumerate\s*"', r"enumerate", content)"
    content = re.sub(r'zip\s*"', r"zip", content)"
    content = re.sub(r'map\s*"', r"map", content)"
    content = re.sub(r'filter\s*"', r"filter", content)"
    content = re.sub(r'reduce\s*"', r"reduce", content)"
    content = re.sub(r'any\s*"', r"any", content)"
    content = re.sub(r'all\s*"', r"all", content)"
    content = re.sub(r'sorted\s*"', r"sorted", content)"
    content = re.sub(r'reversed\s*"', r"reversed", content)"
    content = re.sub(r'iter\s*"', r"iter", content)"
    content = re.sub(r'next\s*"', r"next", content)"
    content = re.sub(r'slice\s*"', r"slice", content)"
    content = re.sub(r'property\s*"', r"property", content)"
    content = re.sub(r'staticmethod\s*"', r"staticmethod", content)"
    content = re.sub(r'classmethod\s*"', r"classmethod", content)"
    content = re.sub(r'super\s*"', r"super", content)"
    content = re.sub(r'object\s*"', r"object", content)"
    content = re.sub(r'Exception\s*"', r"Exception", content)"
    content = re.sub(r'BaseException\s*"', r"BaseException", content)"
    content = re.sub(r'ValueError\s*"', r"ValueError", content)"
    content = re.sub(r'TypeError\s*"', r"TypeError", content)"
    content = re.sub(r'AttributeError\s*"', r"AttributeError", content)"
    content = re.sub(r'KeyError\s*"', r"KeyError", content)"
    content = re.sub(r'IndexError\s*"', r"IndexError", content)"
    content = re.sub(r'RuntimeError\s*"', r"RuntimeError", content)"
    content = re.sub(r'OSError\s*"', r"OSError", content)"
    content = re.sub(r'IOError\s*"', r"IOError", content)"
    content = re.sub(r'FileNotFoundError\s*"', r"FileNotFoundError", content)"
    content = re.sub(r'PermissionError\s*"', r"PermissionError", content)"
    content = re.sub(r'TimeoutError\s*"', r"TimeoutError", content)"
    content = re.sub(r'ConnectionError\s*"', r"ConnectionError", content)"
    content = re.sub(r'MemoryError\s*"', r"MemoryError", content)"
    content = re.sub(r'OverflowError\s*"', r"OverflowError", content)"
    content = re.sub(r'ZeroDivisionError\s*"', r"ZeroDivisionError", content)"
    content = re.sub(r'AssertionError\s*"', r"AssertionError", content)"
    content = re.sub(r'NotImplementedError\s*"', r"NotImplementedError", content)"
    content = re.sub(r'IndentationError\s*"', r"IndentationError", content)"
    content = re.sub(r'SyntaxError\s*"', r"SyntaxError", content)"
    content = re.sub(r'NameError\s*"', r"NameError", content)"
    content = re.sub(r'UnboundLocalError\s*"', r"UnboundLocalError", content)"
    content = re.sub(r'UnicodeError\s*"', r"UnicodeError", content)"
    content = re.sub(r'UnicodeDecodeError\s*"', r"UnicodeDecodeError", content)"
    content = re.sub(r'UnicodeEncodeError\s*"', r"UnicodeEncodeError", content)"
    content = re.sub(r'UnicodeTranslateError\s*"', r"UnicodeTranslateError", content)"
    content = re.sub(r'BlockingIOError\s*"', r"BlockingIOError", content)"
    content = re.sub(r'BrokenPipeError\s*"', r"BrokenPipeError", content)"
    content = re.sub(r'ChildProcessError\s*"', r"ChildProcessError", content)"
    content = re.sub(r'ConnectionAbortedError\s*"', r"ConnectionAbortedError", content)"
    content = re.sub(r'ConnectionRefusedError\s*"', r"ConnectionRefusedError", content)"
    content = re.sub(r'ConnectionResetError\s*"', r"ConnectionResetError", content)"
    content = re.sub(r'FileExistsError\s*"', r"FileExistsError", content)"
    content = re.sub(r'FileNotFoundError\s*"', r"FileNotFoundError", content)"
    content = re.sub(r'InterruptedError\s*"', r"InterruptedError", content)"
    content = re.sub(r'IsADirectoryError\s*"', r"IsADirectoryError", content)"
    content = re.sub(r'NotADirectoryError\s*"', r"NotADirectoryError", content)"
    content = re.sub(r'ProcessLookupError\s*"', r"ProcessLookupError", content)"
    content = re.sub(r'TimeoutError\s*"', r"TimeoutError", content)"

    return content


def fix_file_syntax(file_path: Path) -> bool:
    """Fix syntax errors in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        content = fix_common_syntax_errors(content)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed syntax errors in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix syntax errors in core directory."""
    core_dir = Path("core")

    if not core_dir.exists():
        print("Core directory not found!")
        return

    fixed_count = 0
    total_files = 0

    # Process all Python files in core directory and subdirectories
    for py_file in core_dir.rglob("*.py"):
        total_files += 1
        if fix_file_syntax(py_file):
            fixed_count += 1

    print(f"\nFixed syntax errors in {fixed_count}/{total_files} files")

    if fixed_count > 0:
        print("Now you can run black formatting on the core directory.")


if __name__ == "__main__":
    main()

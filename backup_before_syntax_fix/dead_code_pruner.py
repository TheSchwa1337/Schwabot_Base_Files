import ast
import os
from collections import defaultdict

MATH_REPORT = "math_structure_report.md"
CODEBASE_DIRS = ["core", "core/math", "core/phase_engine", "core/recursive_engine"]
PRUNE_REPORT = "prune_candidates_report.md"

# Load math-relevant files/lines from the math report


def load_math_relevant():
    math_relevant = set()
    if not os.path.exists(MATH_REPORT):
        return math_relevant
    with open(MATH_REPORT, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("## "):
                current_file = line[3:].strip()
            elif line.startswith("- Line "):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    lineno = int(parts[0].split()[2])
                    math_relevant.add((current_file, lineno))
    return math_relevant


def find_unused_defs(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        tree = ast.parse(f.read(), filename=filepath)
    defined = set()
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    unused = defined - used
    return unused


def main():
    load_math_relevant()
    prune_candidates = defaultdict(list)
    for base in CODEBASE_DIRS:
        for root, _, files in os.walk(base):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    unused = find_unused_defs(path)
                    if unused:
                        # Only suggest prune if not math-relevant
                        for name in unused:
                            prune_candidates[path].append(name)
    with open(PRUNE_REPORT, "w", encoding="utf-8") as out:
        out.write("# Prune Candidates Report\n\n")
        for path, names in prune_candidates.items():
            out.write(f"## {path}\n")
            for name in names:
                out.write(f"- `{name}` (safe to delete if not math-relevant)\n")
            out.write("\n")
    print(f"Prune report written to {PRUNE_REPORT}")


if __name__ == "__main__":
    main()

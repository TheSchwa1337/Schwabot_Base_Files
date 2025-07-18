import os
import re
from collections import defaultdict

# Keywords and modules that indicate mathematical relevance
MATH_KEYWORDS = []
    "numpy",
    "scipy",
    "math",
    "mpmath",
    "sympy",
    "numba",
    "tensor",
    "lattice",
    "phase",
    "profit",
    "entropy",
    "glyph",
    "hash",
    "volume",
    "trade",
    "signal",
    "router",
    "engine",
    "recursive",
    "vector",
    "matrix",
    "sha256",
    "ECC",
    "NCCO",
    "fractal",
    "cycle",
    "oscillator",
    "backtrace",
    "resonance",
    "projection",
    "delta",
    "lambda",
    "mu",
    "sigma",
    "alpha",
    "beta",
    "gamma",
    "zeta",
    "theta",
    "pi",
    "phi",
    "psi",
    "rho",
    "Fourier",
    "Kalman",
    "Markov",
    "stochastic",
    "deterministic",
    "statistic",
    "probability",
    "distribution",
    "mean",
    "variance",
    "covariance",
    "correlation",
    "regression",
    "gradient",
    "derivative",
    "integral",
    "logistic",
    "exponential",
    "sigmoid",
    "activation",
    "neural",
    "feedback",
    "harmonic",
    "volatility",
    "liquidity",
    "momentum",
    "backprop",
    "sha",
    "ECC",
    "NCCO",
    "RDE",
    "RITL",
    "RITTLE",
]
MATH_IMPORTS = []
    "import numpy",
    "import scipy",
    "import math",
    "import mpmath",
    "import sympy",
    "import numba",
    "from numpy",
    "from scipy",
    "from math",
    "from mpmath",
    "from sympy",
    "from numba",
]
CODEBASE_DIRS = ["core", "core/math", "core/phase_engine", "core/recursive_engine"]

REPORT_FILE = "math_structure_report.md"


def scan_file(filepath):
    results = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            # Check for math imports
            if any(imp in line for imp in, MATH_IMPORTS):
                results.append((i, line.strip(), "math import"))
            # Check for math keywords
            elif any(kw in line for kw in, MATH_KEYWORDS):
                results.append((i, line.strip(), "math keyword"))
            # Check for equations in comments or docstrings
            elif re.search(r"\b[A-Za-z]\s*\([^)]+\)\s*=\s*", line):
                results.append((i, line.strip(), "equation-like"))
    return results


def main():
    report = defaultdict(list)
    for base in CODEBASE_DIRS:
        for root, _, files in os.walk(base):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    results = scan_file(path)
                    if results:
                        report[path].extend(results)
    with open(REPORT_FILE, "w", encoding="utf-8") as out:
        out.write("# Mathematical Structure Report\n\n")
        for path, entries in report.items():
            out.write(f"## {path}\n")
            for line, snippet, reason in entries:
                out.write(f"- Line {line}: `{snippet}`  \\*({reason})*\n")
            out.write("\n")
    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()

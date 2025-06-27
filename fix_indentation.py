# -*- coding: utf-8 -*-
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
""""""
"""


from utils.safe_print import safe_print, info, warn, error, success, debug
Fix indentation issues in newmath files"""
""""""
""""""
""""""
""""""
"""

import re


def fix_entropy_calc():"""
    """Fix indentation issues in entropy_calc.py""""""
""""""
""""""
""""""
"""
with open('newmath / entropy_calc.py', 'r') as f:
        content = f.read()

# Fix function signature indentation
content = re.sub(
        r'def entropy_filtering\\(entropy_values: np\\.ndarray, filter_type: str = \'moving_average\',\\s*\\n\\s + window: int = 5\\) -> np\\.ndarray:',
        'def entropy_filtering(entropy_values: np.ndarray, filter_type: str = \'moving_average\',\\n                     window: int = 5) -> np.ndarray:',
        content
)

# Fix continuation line indentation
content = re.sub(
        r'filtered\\[i\\] = \\(alpha \\* entropy_values\\[i\\] \\+\\s*\\n\\s+\\(1 - alpha\\) \\* filtered\\[i - 1\\]\\)',
        'filtered[i] = (alpha * entropy_values[i] +\\n                              (1 - alpha) * filtered[i - 1])',
        content
)

# Fix function signature for adaptive_entropy
content = re.sub(
        r'def adaptive_entropy\\(prices: np\\.ndarray, volumes: np\\.ndarray,\\s*\\n\\s + adaptation_rate: float = 0\\.1\\) -> np\\.ndarray:',
        'def adaptive_entropy(prices: np.ndarray, volumes: np.ndarray,\\n                    adaptation_rate: float = 0.1) -> np.ndarray:',
        content
)

# Fix continuation lines in adaptive_entropy
content = re.sub(
        r'adaptation = \\(adaptation_rate \\*\\s*\\n\\s+\\(current_entropy - adaptive_entropy_series\\[i - 1\\]\\)\\)',
        'adaptation = (adaptation_rate *\\n                            (current_entropy - adaptive_entropy_series[i - 1]))',
        content
)

content = re.sub(
        r'adaptive_entropy_series\\[i\\] = \\(adaptive_entropy_series\\[i - 1\\] \\+\\s*\\n\\s + adaptation\\)',
        'adaptive_entropy_series[i] = (adaptive_entropy_series[i - 1] +\\n                                             adaptation)',
        content
)

# Fix function signature for entropy_divergence
content = re.sub(
        r'def entropy_divergence\\(entropy_a: np\\.ndarray, entropy_b: np\\.ndarray,\\s*\\n\\s + method: str = \'kl\'\\) -> float:',
        'def entropy_divergence(entropy_a: np.ndarray, entropy_b: np.ndarray,\\n                      method: str = \'kl\') -> float:',
        content
)

# Fix continuation lines in entropy_divergence
content = re.sub(
        r'entropy_a = \\(entropy_a / np\\.sum\\(entropy_a\\)\\s*\\n\\s + if np\\.sum\\(entropy_a\\) > 0 else entropy_a\\)',
        'entropy_a = (entropy_a / np.sum(entropy_a)\\n                    if np.sum(entropy_a) > 0 else entropy_a)',
        content
)

content = re.sub(
        r'entropy_b = \\(entropy_b / np\\.sum\\(entropy_b\\)\\s*\\n\\s + if np\\.sum\\(entropy_b\\) > 0 else entropy_b\\)',
        'entropy_b = (entropy_b / np.sum(entropy_b)\\n                    if np.sum(entropy_b) > 0 else entropy_b)',
        content
)

with open('newmath / entropy_calc.py', 'w') as f:
        f.write(content)

"""
if __name__ == "__main__":
    fix_entropy_calc()
    safe_print("Fixed indentation issues in entropy_calc.py")

""""""
""""""
""""""
""""""
""""""
"""
"""
def encode_shell_hash(input_hash):
    """Return a short DNA hash for a given input."""
    return f"DNA_{input_hash[:8]}" 
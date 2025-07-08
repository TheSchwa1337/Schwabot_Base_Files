def encode_shell_hash(input_hash):
    """Return a short DNA hash for a given input."""
    return "DNA_{0}".format(input_hash[:8])

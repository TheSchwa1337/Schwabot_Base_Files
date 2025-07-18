def validate_key(key_id: str) -> bool:
    return isinstance(key_id, str) and len(key_id) > 0 
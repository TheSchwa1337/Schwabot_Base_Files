import hashlib

def generate_key_hash(key_id: str, bit_level: str, phase: str) -> str:
    hash_string = f"{key_id}_{bit_level}_{phase}"
    return hashlib.sha256(hash_string.encode()).hexdigest()[:16] 
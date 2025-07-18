from typing import Dict, Any

class KeyMapper:
    def __init__(self):
        self.key_map: Dict[str, Any] = {}

    def map_key(self, key_id: str, data: Any) -> None:
        self.key_map[key_id] = data

    def get_mapping(self, key_id: str) -> Any:
        return self.key_map.get(key_id)

    def get_all_mappings(self) -> Dict[str, Any]:
        return self.key_map 
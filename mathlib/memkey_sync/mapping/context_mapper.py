from typing import Dict, Any

class ContextMapper:
    def __init__(self):
        self.context_map: Dict[str, Any] = {}

    def map_context(self, context_id: str, data: Any) -> None:
        self.context_map[context_id] = data

    def get_context(self, context_id: str) -> Any:
        return self.context_map.get(context_id)

    def get_all_contexts(self) -> Dict[str, Any]:
        return self.context_map 
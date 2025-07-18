from typing import Dict, Any

class DataPointMapper:
    def __init__(self):
        self.data_point_map: Dict[str, Any] = {}

    def map_data_point(self, point_id: str, data: Any) -> None:
        self.data_point_map[point_id] = data

    def get_data_point(self, point_id: str) -> Any:
        return self.data_point_map.get(point_id)

    def get_all_data_points(self) -> Dict[str, Any]:
        return self.data_point_map 
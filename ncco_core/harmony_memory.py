class HarmonyMemory:
    def __init__(self):
        self.patterns = {}

    def add_pattern(self, pattern_id, pattern_data):
        self.patterns[pattern_id] = pattern_data

    def get_pattern(self, pattern_id):
        return self.patterns.get(pattern_id) 
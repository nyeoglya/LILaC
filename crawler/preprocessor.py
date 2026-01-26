
class ProcessedComponent:
    def __init__(self, component_data) -> None:
        self.component_data = component_data
        self.embed = None # embed vector
        self.heading_path = []
        self.subcomponent = [] # (subcomp, embed) list
        self.edge = [] # list(str)

class Preprocessor:
    def __init__(self, components: list) -> None:
        self.components = components
        self.processed = []

    def preprocess(self) -> bool:
        return False

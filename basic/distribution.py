from enum import Enum

class Distribution(Enum):
    X_PX = "x_px"
    Y_PX = "y_px"
    X_PY = "x_py"
    Y_PY = "y_py"

    def space(self):
        return self.value[0].upper()

    def momentum(self):
        return f"P{self.value[-1]}"

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return self.value.__hash__()

    def __str__(self) -> str:
        return self.value
from enum import Enum

class Species(Enum):
    ELECTRON = "Electrons"
    PROTON   = "Protons"
    ALPHA    = "Alphas"

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return self.value.__hash__()

from enum import Enum

class Species(Enum):
    ELECTRON = "Electrons"
    PROTON   = "Protons"
    ALPHA    = "Alphas"

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return self.value.__hash__()

    def __str__(self) -> str:
        return self.value

    def symbol(self) -> str:
        """Get superscript/subscript symbol for LaTeX.
        """
        SYMBOLS = {
            Species.ELECTRON: "\\text{e}",
            Species.PROTON  : "\\text{p}",
            Species.ALPHA   : "\\alpha",
        }
        assert self in SYMBOLS, "What is going on?"
        return SYMBOLS.get(self)

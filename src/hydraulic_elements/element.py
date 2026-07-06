class Element:
    """
    Base class for hydraulic elements.

    Elements encapsulate physics (from SIMSEN DAT files).
    They are connected into a network separately (topology).
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

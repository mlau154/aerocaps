from abc import ABC


class Unit(ABC):
    def __init__(self, primary_unit: str):
        self.primary_unit = primary_unit

    def __add__(self, other):
        if self.__class__ == other.__class__:
            new_primary_value = getattr(self, self.primary_unit) + getattr(other, other.primary_unit)
            return self.__class__(**{self.primary_unit: new_primary_value})
        elif isinstance(other, (int, float)):
            new_primary_value = getattr(self, self.primary_unit) + other
            return self.__class__(**{self.primary_unit: new_primary_value})
        else:
            return NotImplemented

    def __sub__(self, other):
        if self.__class__ == other.__class__:
            new_primary_value = getattr(self, self.primary_unit) - getattr(other, other.primary_unit)
            return self.__class__(**{self.primary_unit: new_primary_value})
        elif isinstance(other, (int, float)):
            new_primary_value = getattr(self, self.primary_unit) - other
            return self.__class__(**{self.primary_unit: new_primary_value})
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_primary_value = getattr(self, self.primary_unit) * other
            return self.__class__(**{self.primary_unit: new_primary_value})
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            new_primary_value = getattr(self, self.primary_unit) / other
            return self.__class__(**{self.primary_unit: new_primary_value})
        elif self.__class__ == other.__class__:
            new_primary_value = getattr(self, self.primary_unit) / getattr(other, other.primary_unit)
            return new_primary_value
        else:
            return NotImplemented

    def __abs__(self):
        new_primary_value = abs(getattr(self, self.primary_unit))
        return self.__class__(**{self.primary_unit: new_primary_value})

    def __neg__(self):
        new_primary_value = -(getattr(self, self.primary_unit))
        return self.__class__(**{self.primary_unit: new_primary_value})

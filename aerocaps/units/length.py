from aerocaps.units.unit import Unit
from aerocaps.units.area import Area


__all__ = [
    "Length"
]


class Length(Unit):
    """Base-level class for a length dimension with various available units."""
    def __init__(self, ft=None, m=None, inch=None, mm=None, mi=None, nmi=None, km=None, cm=None):
        self._ft, self._m, self._inch, self._mm, self._mi, self._nmi, self._km, self._cm = \
            None, None, None, None, None, None, None, None
        self.convert_from_feet_map = {
            'm': 0.3048,
            'inch': 12,
            'mm': 304.8,
            'mi': 1 / 5280,
            'nmi': 1 / 5280 / 1.150779448,
            'km': 0.0003048,
            'cm': 30.48
        }
        if ft is not None:
            self.ft = ft
        elif m is not None:
            self.m = m
        elif inch is not None:
            self.inch = inch
        elif mm is not None:
            self.mm = mm
        elif mi is not None:
            self.mi = mi
        elif nmi is not None:
            self.nmi = nmi
        elif km is not None:
            self.km = km
        elif cm is not None:
            self.cm = cm
        super().__init__(primary_unit="ft")

    def set_all(self):
        """After setting the length in feet, set the length for all other units"""
        for k, v in self.convert_from_feet_map.items():
            setattr(self, f'_{k}', self._ft * v)

    @property
    def ft(self):
        return self._ft

    @ft.setter
    def ft(self, ft):
        self._ft = ft
        self.set_all()

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self._ft = m / self.convert_from_feet_map['m']
        self.set_all()

    @property
    def inch(self):
        return self._inch

    @inch.setter
    def inch(self, inch):
        self._ft = inch / self.convert_from_feet_map['inch']
        self.set_all()

    @property
    def mm(self):
        return self._mm

    @mm.setter
    def mm(self, mm):
        self._ft = mm / self.convert_from_feet_map['mm']
        self.set_all()

    @property
    def mi(self):
        return self._mi

    @mi.setter
    def mi(self, mi):
        self._ft = mi / self.convert_from_feet_map['mi']
        self.set_all()

    @property
    def nmi(self):
        return self._nmi

    @nmi.setter
    def nmi(self, nmi):
        self._ft = nmi / self.convert_from_feet_map['nmi']
        self.set_all()

    @property
    def km(self):
        return self._km

    @km.setter
    def km(self, km):
        self._ft = km / self.convert_from_feet_map['km']
        self.set_all()

    @property 
    def cm(self): 
        return self._cm
    
    @cm.setter
    def cm(self, cm):
        self._ft = cm / self.convert_from_feet_map['cm']
        self.set_all()

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_primary_value = getattr(self, self.primary_unit) * other
            return self.__class__(**{self.primary_unit: new_primary_value})
        elif isinstance(other, Length):
            return Area(m2=self.m * other.m)
        else:
            return NotImplemented

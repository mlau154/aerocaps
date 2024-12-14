from aerocaps.units.unit import Unit


__all__ = [
    "Area"
]


area_conversions_from_m2 = {
    'ft2': (1 / 0.3048) ** 2,
    'mm2': 1000 ** 2,
    'cm2': 100 ** 2,
    'in2': (12 / 0.3048) ** 2,
}


class Area(Unit):
    def __init__(self, m2=None, mm2=None, cm2=None, ft2=None, in2=None):
        self._m2, self._mm2, self._cm2, self._ft2, self._in2 = None, None, None, None, None
        if m2 is not None:
            self.m2 = m2
        elif mm2 is not None:
            self.mm2 = mm2
        elif cm2 is not None:
            self.cm2 = cm2
        elif ft2 is not None:
            self.ft2 = ft2
        elif in2 is not None:
            self.in2 = in2
        super().__init__(primary_unit="m2")

    def set_all(self):
        """After setting the area in m2, set the area for all other units"""
        for k, v in area_conversions_from_m2.items():
            setattr(self, f'_{k}', self._m2 * v)

    @property
    def m2(self):
        return self._m2

    @m2.setter
    def m2(self, value):
        self._m2 = value
        self.set_all()

    @property
    def mm2(self):
        return self._mm2

    @mm2.setter
    def mm2(self, value):
        self._m2 = value / area_conversions_from_m2['mm2']
        self.set_all()

    @property
    def cm2(self):
        return self._cm2

    @cm2.setter
    def cm2(self, value):
        self._m2 = value / area_conversions_from_m2['cm2']
        self.set_all()

    @property
    def ft2(self):
        return self._ft2

    @ft2.setter
    def ft2(self, value):
        self._m2 = value / area_conversions_from_m2['ft2']
        self.set_all()

    @property
    def in2(self):
        return self._in2

    @in2.setter
    def in2(self, value):
        self._m2 = value / area_conversions_from_m2['in2']
        self.set_all()

import numpy as np

from aerocaps.units.unit import Unit


__all__ = [
    "Angle"
]


angle_conversions_from_rad = {
    'deg': lambda r: np.rad2deg(r),
}

angle_conversions_to_rad = {
    'deg': lambda d: np.deg2rad(d),
}


class Angle(Unit):
    def __init__(self, rad=None, deg=None):
        self._rad, self._deg = None, None
        if rad is not None:
            self.rad = rad
        elif deg is not None:
            self.deg = deg
        super().__init__(primary_unit="rad")

    def set_all(self):
        """After setting the temperature in rad, set the temperature for all other units"""
        for k, v in angle_conversions_from_rad.items():
            setattr(self, f'_{k}', v(self._rad))

    @property
    def rad(self):
        return self._rad

    @rad.setter
    def rad(self, value):
        self._rad = value
        self.set_all()

    @property
    def deg(self):
        return self._deg

    @deg.setter
    def deg(self, value):
        self._rad = angle_conversions_to_rad['deg'](value)
        self.set_all()

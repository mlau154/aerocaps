import numpy as np

from aerocaps.iges.entity import IGESEntity
from aerocaps.iges.iges_param import IGESParam
from aerocaps.units.angle import Angle
from aerocaps.units.length import Length


class TransformationMatrixIGES(IGESEntity):
    def __init__(self,
                 yaw_z: Angle = Angle(deg=0.0),
                 pitch_y: Angle = Angle(deg=0.0),
                 roll_x: Angle = Angle(deg=0.0),
                 tx: Length = Length(m=0.0),
                 ty: Length = Length(m=0.0),
                 tz: Length = Length(m=0.0)
                 ):
        R_mat = self.generate_roll_pitch_yaw_matrix(yaw_z, pitch_y, roll_x)
        parameter_data = [
            IGESParam(R_mat[0, 0], "real"),
            IGESParam(R_mat[0, 1], "real"),
            IGESParam(R_mat[0, 2], "real"),
            IGESParam(tx.m, "real"),
            IGESParam(R_mat[1, 0], "real"),
            IGESParam(R_mat[1, 1], "real"),
            IGESParam(R_mat[1, 2], "real"),
            IGESParam(ty.m, "real"),
            IGESParam(R_mat[2, 0], "real"),
            IGESParam(R_mat[2, 1], "real"),
            IGESParam(R_mat[2, 2], "real"),
            IGESParam(tz.m, "real")
        ]
        super().__init__(124, parameter_data)

    @staticmethod
    def _generate_matrix_z(alpha: Angle):
        """
        Generates the yaw matrix
        """
        return np.array([
            [np.cos(alpha.rad), -np.sin(alpha.rad), 0.0],
            [np.sin(alpha.rad), np.cos(alpha.rad), 0.0],
            [0.0, 0.0, 1.0]
        ])

    @staticmethod
    def _generate_matrix_y(beta: Angle):
        """
        Generates the pitch matrix
        """
        return np.array([
            [np.cos(beta.rad), 0.0, np.sin(beta.rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(beta.rad), 0.0, np.cos(beta.rad)]
        ])

    @staticmethod
    def _generate_matrix_x(gamma: Angle):
        """
        Generates the roll matrix
        """
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(gamma.rad), -np.sin(gamma.rad)],
            [0.0, np.sin(gamma.rad), np.cos(gamma.rad)]
        ])

    def generate_roll_pitch_yaw_matrix(self, alpha: Angle, beta: Angle, gamma: Angle):
        yaw_matrix = self._generate_matrix_z(alpha)
        pitch_matrix = self._generate_matrix_y(beta)
        roll_matrix = self._generate_matrix_x(gamma)
        return (yaw_matrix @ pitch_matrix) @ roll_matrix

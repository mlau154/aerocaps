import os

from .geom.curves import *
from .geom.geometry_container import *
from .geom.intersection import *
from .geom.plane import *
from .geom.point import *
from .geom.surfaces import *
from .geom.tools import *
from .geom.transformation import *
from .geom.vector import *
from .units.area import *
from .units.length import *
from .units.angle import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "tests")

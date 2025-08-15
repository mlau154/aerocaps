__version__ = "0.3.2"


def get_major_project_version() -> str:
    """
    Gets the major project version for AeroCAPS using semantic versioning

    Returns
    -------
    str
        The major project version
    """
    split_version = __version__.split(".")
    return ".".join(split_version[:2])

from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from typing import Callable
from torch.nn import Module
from torch import save


def get_l1_zenith_radec(gps_time):
    """
    Get the RA and Dec of the point directly overhead LIGO Livingston at a given GPS time.
    This is the optimal location to detect a gravitational wave from at a specific time.

    Args:
        gps_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    # LIGO Livingston (L1) geodetic coordinates
    l1_lat = 30.5630 * u.deg
    l1_lon = -90.7742 * u.deg  # West is negative
    l1_height = 0 * u.m  # Approximate sea level height

    location = EarthLocation(lat=l1_lat, lon=l1_lon, height=l1_height)

    # Convert GPS time to an astropy Time object
    t = Time(gps_time, format="gps", location=location)

    # RA at zenith is the Local Sidereal Time (LST)
    ra_zenith = t.sidereal_time("mean").to(u.radian)

    # Dec at zenith is exactly the latitude
    dec_zenith = l1_lat.to(u.radian)

    return ra_zenith.value, dec_zenith.value

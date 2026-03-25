"""Unit tests for the OpenDRIVE elevation profile evaluation."""

from log_viewer.xodr_parser import ElevationEntry, ElevationProfile
import pytest


def test_elevation_profile_evaluation():
    """Verify that elevation profile correctly interpolates and evaluates Z.
    
    Tests flat, linear, and multi-segment profiles.
    """
    # Test a simple flat elevation
    flat_entry = ElevationEntry(s=0.0, a=10.0, b=0.0, c=0.0, d=0.0)
    profile = ElevationProfile([flat_entry])
    assert profile.get_z(0.0) == 10.0
    assert profile.get_z(100.0) == 10.0

    # Test linear profile z(ds) = 10 + 2*ds
    linear_entry = ElevationEntry(s=0.0, a=10.0, b=2.0, c=0.0, d=0.0)
    profile = ElevationProfile([linear_entry])
    assert profile.get_z(0.0) == 10.0
    assert profile.get_z(5.0) == 20.0  # 10 + 2*5

    # Test multi-entry profile
    # 0.0 - 10.0: z = 0
    # 10.0 - 20.0: z = 10 + 5*(s-10)
    e1 = ElevationEntry(s=0.0, a=0.0, b=0.0, c=0.0, d=0.0)
    e2 = ElevationEntry(s=10.0, a=10.0, b=5.0, c=0.0, d=0.0)
    profile = ElevationProfile([e1, e2])

    assert profile.get_z(5.0) == 0.0
    assert profile.get_z(10.0) == 10.0
    assert profile.get_z(12.0) == 20.0  # 10 + 5*2


def test_elevation_profile_cubic():
    """Verify evaluation of cubic polynomial elevation profiles."""
    # Test cubic profile z(ds) = ds^3
    cubic_entry = ElevationEntry(s=0.0, a=0.0, b=0.0, c=0.0, d=1.0)
    profile = ElevationProfile([cubic_entry])
    assert profile.get_z(0.0) == 0.0
    assert profile.get_z(2.0) == 8.0  # 2^3
    assert profile.get_z(3.0) == 27.0  # 3^3


def test_elevation_profile_defaults_to_zero_before_first_entry():
    """Verify that the profile stays at zero before the first explicit record."""
    profile = ElevationProfile(
        [ElevationEntry(s=10.0, a=5.0, b=1.0, c=0.0, d=0.0)]
    )
    assert profile.get_z(0.0) == 0.0
    assert profile.get_z(5.0) == 0.0
    assert profile.get_z(10.0) == 5.0


if __name__ == "__main__":
    pytest.main([__file__])

"""Tests for marker_asterisk_count: one asterisk per started 0.1 s of a
marker, without float error adding an asterisk to exact tenths.
"""

import pytest

from transcript_render.markers import marker_asterisk_count


@pytest.mark.parametrize('duration, expected', [
    (3.62 - 3.42, 2),   # 0.20000000000000018
    (1.26 - 0.8, 5),    # 0.45999999999999996
    (0.21, 3),
    (0.05, 1),
    (0.0, 1),
    (-0.1, 1),
    (12.0, 50),
])
def test_marker_asterisk_count(duration, expected):
    assert marker_asterisk_count(duration) == expected

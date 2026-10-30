import numpy as np
from compare import net_value


def test_net_value_by_hand():
    # 3 customers, monthly charge 100. Threshold 0.5 contacts customers 0 and 1 (2 offers).
    # Only customer 0 actually churns: saved value = 100 * 12 * 0.25 = 300. Cost = 2 * 60 = 120.
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.6, 0.1])
    m = np.array([100.0, 100.0, 100.0])
    assert net_value(y, p, m, 0.5) == 300 - 120


def test_contacting_nobody_is_zero():
    assert net_value(np.array([1, 0]), np.array([0.2, 0.1]), np.array([50.0, 50.0]), 0.99) == 0

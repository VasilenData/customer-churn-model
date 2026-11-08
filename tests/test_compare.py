import numpy as np
import pytest
from compare import net_value, select_threshold


def test_net_value_by_hand():
    # 3 customers, monthly charge 100. Threshold 0.5 contacts customers 0 and 1 (2 offers).
    # Only customer 0 actually churns: saved value = 100 * 12 * 0.25 = 300. Cost = 2 * 60 = 120.
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.6, 0.1])
    m = np.array([100.0, 100.0, 100.0])
    assert net_value(y, p, m, 0.5) == 300 - 120


def test_contacting_nobody_is_zero():
    assert net_value(np.array([1, 0]), np.array([0.2, 0.1]), np.array([50.0, 50.0]), 0.99) == 0


def test_utility_optimizer_can_decline_all_loss_making_offers():
    threshold, value = select_threshold([0, 1], [1.0, 1.0], [1., 1.])
    assert threshold > 1 and value == 0
    assert net_value([0, 1], [1., 1.], [1., 1.], threshold) == 0


def test_utility_optimizer_can_choose_contact_everyone():
    threshold, value = select_threshold([1, 1], [0.01, 0.02], [100., 100.])
    assert threshold == 0 and value == 480


def test_misaligned_arrays_cannot_broadcast_into_false_savings():
    with pytest.raises(ValueError, match="aligned"):
        net_value([1], [0.9, 0.8], [100, 100], 0.5)


@pytest.mark.parametrize("proba", [[float("nan")], [1.1], [-0.1]])
def test_invalid_probabilities_fail_clearly(proba):
    with pytest.raises(ValueError):
        net_value([1], proba, [100], 0.5)

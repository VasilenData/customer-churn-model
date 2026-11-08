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
    assert threshold == 0.01 and value == 480


def test_optimizer_finds_profitable_policy_between_old_grid_points():
    threshold, value = select_threshold([1, 0], [0.957, 0.951], [100., 100.])
    assert threshold == 0.957 and value == 240


def test_equal_scores_are_an_indivisible_contact_group():
    threshold, value = select_threshold([1, 0, 0], [0.9, 0.9, 0.9], [50., 50., 50.])
    assert threshold > 1 and value == 0


def test_zero_gain_prefers_fewer_contacts():
    threshold, value = select_threshold([1, 1], [0.8, 0.3], [100., 20.])
    assert threshold == 0.8 and value == 240


def test_threshold_optimizer_matches_exhaustive_contact_mask_oracle():
    """Small seeded cases enumerate all attainable masks independently of sorting/cumulative sums."""
    rng = np.random.default_rng(47)
    for size in range(1, 30):
        for _ in range(10):
            y = rng.integers(0, 2, size)
            p = rng.choice([0., 0.013, 0.051, 0.503, 0.951, 0.957, 1.], size=size)
            monthly = rng.integers(1, 180, size)
            candidates = [np.nextafter(1.0, 2.0), *set(p)]
            def oracle(t):
                return sum((float(monthly[i]) * 12 * 0.25 if y[i] else 0) - 60
                           for i in range(size) if p[i] >= t)
            expected_t = max(candidates, key=lambda t: (oracle(t), t))
            threshold, value = select_threshold(y, p, monthly)
            assert threshold == expected_t and value == oracle(expected_t)
            # Row order must not change the chosen contact policy.
            shuffled = rng.permutation(size)
            assert select_threshold(y[shuffled], p[shuffled], monthly[shuffled]) == (threshold, value)


def test_misaligned_arrays_cannot_broadcast_into_false_savings():
    with pytest.raises(ValueError, match="aligned"):
        net_value([1], [0.9, 0.8], [100, 100], 0.5)


@pytest.mark.parametrize("proba", [[float("nan")], [1.1], [-0.1]])
def test_invalid_probabilities_fail_clearly(proba):
    with pytest.raises(ValueError):
        net_value([1], proba, [100], 0.5)

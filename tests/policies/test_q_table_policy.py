
from gym.spaces import Discrete, MultiBinary
import numpy as np
import pytest

from abmarl.policies.q_table_policy import QTablePolicy, GreedyPolicy, EpsilonSoftPolicy


class QPolicyTester(QTablePolicy):
    def reset(self):
        self.first_guess = True

    def compute_action(self, obs, **kwargs):
        pass

    def probability(self, obs, action, **kwargs):
        pass


def test_q_policy_init_and_properties():
    with pytest.raises(AssertionError):
        QPolicyTester(observation_space = Discrete(3), action_space=MultiBinary(4))
    with pytest.raises(AssertionError):
        QPolicyTester(observation_space=MultiBinary(2), action_space=Discrete(10))
    policy = QPolicyTester(observation_space=Discrete(3), action_space=(Discrete(10)))
    assert policy.q_table.shape == (3, 10)
    assert policy.action_space == Discrete(10)
    assert policy.observation_space == Discrete(3)

    with pytest.raises(AssertionError):
        policy.q_table = Discrete(10)
    with pytest.raises(AssertionError):
        policy.action_space = MultiBinary(10)
    with pytest.raises(AssertionError):
        policy.observation_space = MultiBinary(10)


def test_q_policy_build():
    q = np.random.normal(0, 1, size=(14, 4))
    policy = QPolicyTester.build(q)
    assert policy.observation_space == Discrete(14)
    assert policy.action_space == Discrete(4)
    np.testing.assert_array_equal(policy.q_table, q)

    policy2 = QPolicyTester.build(policy)
    np.testing.assert_array_equal(policy2.q_table, q)
    assert policy2.observation_space == Discrete(14)
    assert policy2.action_space == Discrete(4)


def test_q_policy_update():
    q = np.array([
        [ 0,  1,  2],
        [-1,  1, -2],
        [ 0,  1,  0],
        [ 2, -2, -2]])
    policy = QPolicyTester.build(q)
    policy.update(0, 1, 3)
    policy.update(2, 2, -1)
    policy.update(3, 0, -2)
    np.testing.assert_array_equal(
        policy.q_table,
        np.array([
            [ 0,  3,  2],
            [-1,  1, -2],
            [ 0,  1, -1],
            [-2, -2, -2]
        ])
    )


def test_greedy_policy():
    q = np.array([
        [ 0,  1,  2],
        [-1,  1, -2],
        [ 0,  1,  0],
        [ 2, -2, -2]])
    policy = GreedyPolicy.build(q)
    assert policy.compute_action(0) == 2
    assert policy.probability(0, 0) == 0
    assert policy.probability(0, 1) == 0
    assert policy.probability(0, 2) == 1

    assert policy.compute_action(1) == 1
    assert policy.probability(1, 0) == 0
    assert policy.probability(1, 1) == 1
    assert policy.probability(1, 2) == 0

    assert policy.compute_action(2) == 1
    assert policy.probability(2, 0) == 0
    assert policy.probability(2, 1) == 1
    assert policy.probability(2, 2) == 0

    assert policy.compute_action(3) == 0
    assert policy.probability(3, 0) == 1
    assert policy.probability(3, 1) == 0
    assert policy.probability(3, 2) == 0


def test_epsilon_soft_init_and_build():
    policy = EpsilonSoftPolicy(
        observation_space=Discrete(3),
        action_space=Discrete(5),
        epsilon=0.6
    )
    assert policy.epsilon == 0.6

    policy = EpsilonSoftPolicy.build(policy, epsilon=0.2)
    assert policy.epsilon == 0.2

    with pytest.raises(AssertionError):
        EpsilonSoftPolicy(
            observation_space=Discrete(3),
            action_space=Discrete(5),
            epsilon=-0.6
        )

    with pytest.raises(AssertionError):
        EpsilonSoftPolicy.build(policy, epsilon=1.6)


def test_epsilon_soft_compute_action_and_probability():
    np.random.seed(24)
    q = np.array([
        [ 0,  1,  2],
        [-1,  1, -2],
        [ 0,  1,  0],
        [ 2, -2, -2]])
    policy = EpsilonSoftPolicy.build(q, epsilon=0.5)
    assert policy.compute_action(0) == 2 # random
    assert policy.probability(0, 0) == 0.5 / 3
    assert policy.probability(0, 1) == 0.5 / 3
    assert policy.probability(0, 2) == 1 - 0.5 + 0.5 / 3

    assert policy.compute_action(1) == 1 # random
    assert policy.probability(1, 0) == 0.5 / 3
    assert policy.probability(1, 1) == 1 - 0.5 + 0.5 / 3
    assert policy.probability(1, 2) == 0.5 / 3

    assert policy.compute_action(2) == 1 # random
    assert policy.probability(2, 0) == 0.5 / 3
    assert policy.probability(2, 1) == 1 - 0.5 + 0.5 / 3
    assert policy.probability(2, 2) == 0.5 / 3

    assert policy.compute_action(3) == 1 # random
    assert policy.probability(3, 0) == 1 - 0.5 + 0.5 / 3
    assert policy.probability(3, 1) == 0.5 / 3
    assert policy.probability(3, 2) == 0.5 / 3

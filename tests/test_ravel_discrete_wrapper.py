
from gym.spaces import MultiDiscrete, Discrete, MultiBinary, Dict, Tuple
from gym.spaces import Box as GymBox
import numpy as np
import pytest

from abmarl.tools import Box
from abmarl.sim.wrappers.ravel_discrete_wrapper import ravel, unravel
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.sim import Agent
from abmarl.examples import EmptyABS, MultiAgentGymSpacesSim


def test_ravel():
    my_space = Dict({
        'a': MultiDiscrete([5, 3]),
        'b': MultiBinary(4),
        'c': GymBox(np.array([[-2, 6, 3],[0, 0, 1]]), np.array([[2, 12, 5],[2, 4, 2]]), dtype=int),
        'd': Dict({
            1: Discrete(3),
            2: Box(1, 3, (2,), int)
        }),
        'e': Tuple((
            MultiDiscrete([4, 1, 5]),
            MultiBinary(2),
            Dict({
                'my_dict': Discrete(11)
            })
        )),
        'f': Discrete(6),
    })
    point = {
        'a': [3, 1],
        'b': [0, 1, 1, 0],
        'c': np.array([[0, 7, 5],[1, 3, 1]]),
        'd': {1: 2, 2: np.array([1, 3])},
        'e': ([1,0,4], [1, 1], {'my_dict': 5}),
        'f': 1
    }
    ravelled_point = ravel(my_space, point)
    unravelled_point = unravel(my_space, ravelled_point)

    assert ravelled_point == 74748022765
    np.testing.assert_array_equal(unravelled_point['a'], point['a'])
    np.testing.assert_array_equal(unravelled_point['b'], point['b'])
    np.testing.assert_array_equal(unravelled_point['c'], point['c'])
    assert unravelled_point['d'][1] == point['d'][1]
    np.testing.assert_array_equal(unravelled_point['d'][2], point['d'][2])
    np.testing.assert_array_equal(unravelled_point['e'][0], point['e'][0])
    np.testing.assert_array_equal(unravelled_point['e'][1], point['e'][1])
    np.testing.assert_array_equal(unravelled_point['e'][2], point['e'][2])
    assert unravelled_point['f'] == point['f']


# Observations that we can't ravel
class FloatObservation(EmptyABS):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(-1.0, 1.0, shape=(4,)), action_space=Discrete(3)
        )}


class UnboundedBelowObservation(EmptyABS):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(
                np.array([0, 13, -3, -np.inf]),
                np.array([0, 20, 0, 0]),
                dtype=int
            ),
            action_space=Discrete(3)
        )}


class UnboundedAboveObservation(EmptyABS):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(
                np.array([0, 12, 20, 0]),
                np.array([np.inf, 20, 24, np.inf]),
                dtype=int
            ),
            action_space=Discrete(2)
        )}


# Actions that we can't ravel
class FloatAction(EmptyABS):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(-1.0, 1.0, shape=(4,)), action_space=Discrete(3)
        )}


class UnboundedBelowAction(EmptyABS):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0',
            observation_space=Box(
                np.array([0, 13, -3, -np.inf]),
                np.array([0, 20, 0, 0]),
                dtype=int
            ),
            action_space=Discrete(3)
        )}


class UnboundedAboveAction(EmptyABS):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0',
            observation_space=Box(
                np.array([0, 12, 20, 0]),
                np.array([np.inf, 20, 24, np.inf]),
                dtype=int
            ),
            action_space=Discrete(2)
        )}


def test_exceptions():
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(FloatObservation())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedBelowObservation())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedAboveObservation())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(FloatAction())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedBelowAction())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedAboveAction())


def test_ravel_wrapper():
    sim = MultiAgentGymSpacesSim()
    wrapped_sim = RavelDiscreteWrapper(sim)
    assert wrapped_sim.unwrapped == sim
    for agent_id in wrapped_sim.agents:
        if not isinstance(wrapped_sim.agents[agent_id], Agent): continue
        assert isinstance(wrapped_sim.agents[agent_id].observation_space, Discrete)
        assert isinstance(wrapped_sim.agents[agent_id].action_space, Discrete)
    sim = wrapped_sim

    sim.reset()
    assert sim.get_obs('agent0') == 1
    assert sim.get_obs('agent1') == 0
    assert sim.get_obs('agent2') == 2
    assert sim.get_obs('agent3') == 47

    action_0 = {
        'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 1]),
        'agent1': [3, 2, 0],
        'agent2': {'alpha': [1, 1, 0]},
        'agent3': (2, np.array([0, 6]), 1)
    }
    action_0_wrapped = {
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_0.items()
    }

    action_1 = {
        'agent0': ({'first': 0, 'second': [3, 3]}, [1, 1, 1]),
        'agent1': [1, 5, 1],
        'agent2': {'alpha': [1, 0, 0]},
        'agent3': (1, np.array([9, 4]), 0)
    }
    action_1_wrapped = {
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_1.items()
    }

    action_2 = {
        'agent0': ({'first': 1, 'second': [1, 0]}, [0, 0, 1]),
        'agent1': [2, 0, 1],
        'agent2': {'alpha': [0, 0, 0]},
        'agent3': (0, np.array([7, 7]), 0)
    }
    action_2_wrapped = {
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_2.items()
    }

    sim.step(action_0_wrapped)
    assert sim.get_obs('agent0') == 1
    assert sim.get_obs('agent1') == 0
    assert sim.get_obs('agent2') == 2
    assert sim.get_obs('agent3') == 47

    assert sim.get_reward('agent0') == 2
    assert sim.get_reward('agent1') == 3
    assert sim.get_reward('agent2') == 5
    assert sim.get_reward('agent3') == 7

    assert not sim.get_done('agent0')
    assert not sim.get_done('agent1')
    assert not sim.get_done('agent2')
    assert not sim.get_done('agent3')

    assert sim.get_info('agent0')[0]['first'] == action_0['agent0'][0]['first']
    np.testing.assert_array_equal(
        sim.get_info('agent0')[0]['second'], action_0['agent0'][0]['second']
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_0['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_0['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_0['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_0['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_0['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_0['agent3'][2])

    sim.step(action_1_wrapped)
    assert sim.get_info('agent0')[0]['first'] == action_1['agent0'][0]['first']
    np.testing.assert_array_equal(
        sim.get_info('agent0')[0]['second'], action_1['agent0'][0]['second']
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_1['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_1['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_1['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_1['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_1['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_1['agent3'][2])

    sim.step(action_2_wrapped)
    assert sim.get_info('agent0')[0]['first'] == action_2['agent0'][0]['first']
    np.testing.assert_array_equal(
        sim.get_info('agent0')[0]['second'], action_2['agent0'][0]['second']
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_2['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_2['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_2['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_2['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_2['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_2['agent3'][2])


def test_ravel_null_points():
    abs = MultiAgentGymSpacesSim()
    agents = abs.agents
    agents['agent0'].null_observation = [0, 0, 0, 0]
    assert agents['agent0'].null_observation in agents['agent0'].observation_space
    agents['agent0'].null_action = ({'first': 0, 'second': [0, 0]}, [0, 0, 0])
    assert agents['agent0'].null_action in agents['agent0'].action_space
    agents['agent1'].null_observation = [0]
    assert agents['agent1'].null_observation in agents['agent1'].observation_space
    agents['agent1'].null_action = [0, 0, 0]
    assert agents['agent1'].null_action in agents['agent1'].action_space
    agents['agent2'].null_observation = [0, 0]
    assert agents['agent2'].null_observation in agents['agent2'].observation_space
    agents['agent2'].null_action = {'alpha': [0, 0, 0]}
    assert agents['agent2'].null_action in agents['agent2'].action_space
    agents['agent3'].null_observation = {'first': 0, 'second': [0, 0]}
    assert agents['agent3'].null_observation in agents['agent3'].observation_space
    agents['agent3'].null_action = (0, [0, 0], 0)
    assert agents['agent3'].null_action in agents['agent3'].action_space

    sim = RavelDiscreteWrapper(abs)
    agents = sim.agents
    assert agents['agent0'].null_observation == 0
    assert agents['agent0'].null_action == 48
    assert agents['agent1'].null_observation == 0
    assert agents['agent1'].null_action == 0
    assert agents['agent2'].null_observation == 0
    assert agents['agent2'].null_action == 0
    assert agents['agent3'].null_observation == 6
    assert agents['agent3'].null_action == 0

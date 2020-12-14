
from gym.spaces import Box

import numpy as np

from admiral.envs.components.position import PositionAgent, TeamAgent, ObservingAgent
from admiral.envs.components.position import PositionState, PositionObserver, GridPositionBasedObserver, GridPositionTeamBasedObserver

class PositionTestAgent(PositionAgent, ObservingAgent): pass
class PositionTeamTestAgent(PositionAgent, ObservingAgent, TeamAgent): pass
class PositionTeamNoViewTestAgent(PositionAgent, TeamAgent): pass

def test_position_observer():
    pass # TODO: Implement position based observer of the agents

def test_grid_position_observer():
    agents = {
        'agent0': PositionTestAgent(id='agent0', starting_position=np.array([0, 0]), view=1),
        'agent1': PositionTestAgent(id='agent1', starting_position=np.array([2, 2]), view=2),
        'agent2': PositionTestAgent(id='agent2', starting_position=np.array([3, 2]), view=3),
        'agent3': PositionTestAgent(id='agent3', starting_position=np.array([1, 4]), view=4),
        'agent4': PositionAgent(id='agent4', starting_position=np.array([1, 4])),
    }
    
    state = PositionState(agents=agents, region=5)
    observer = GridPositionBasedObserver(position=state, agents=agents)
    for agent in agents.values():
        state.reset(agent)

    np.testing.assert_array_equal(observer.get_obs(agents['agent0']), np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent1']), np.array([
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2']), np.array([
        [-1.,  1.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  1., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3']), np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 1.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  1., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))


def test_grid_team_position_observer():
    agents = {
        'agent0': PositionTeamTestAgent      (id='agent0', team=0, starting_position=np.array([0, 0]), view=1),
        'agent1': PositionTeamNoViewTestAgent(id='agent1', team=0, starting_position=np.array([0, 0])),
        'agent2': PositionTeamTestAgent      (id='agent2', team=0, starting_position=np.array([2, 2]), view=2),
        'agent3': PositionTeamTestAgent      (id='agent3', team=1, starting_position=np.array([3, 2]), view=3),
        'agent4': PositionTeamTestAgent      (id='agent4', team=1, starting_position=np.array([1, 4]), view=4),
        'agent5': PositionTeamNoViewTestAgent(id='agent5', team=1, starting_position=np.array([1, 4])),
        'agent6': PositionTeamNoViewTestAgent(id='agent6', team=1, starting_position=np.array([1, 4])),
        'agent7': PositionTeamTestAgent      (id='agent7', team=2, starting_position=np.array([1, 4]), view=2),
    }
    for agent in agents.values():
        agent.position = agent.starting_position
    
    state = PositionState(agents=agents, region=5)
    observer = GridPositionTeamBasedObserver(position=state, agents=agents, number_of_teams=3)
    for agent in agents.values():
        state.reset(agent)

    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])[:,:,0], np.array([
        [-1., -1., -1.],
        [-1.,  1.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])[:,:,1], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])[:,:,2], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])[:,:,0], np.array([
        [2., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])[:,:,1], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 3.],
        [0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])[:,:,2], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])[:,:,0], np.array([
        [-1.,  2.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])[:,:,1], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  3., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])[:,:,2], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])[:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 2.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])[:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  2., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])[:,:,2], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])[:,:,0], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])[:,:,1], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  3., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])[:,:,2], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))

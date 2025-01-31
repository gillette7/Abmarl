from gym.spaces import Dict, Tuple, Discrete, MultiDiscrete, MultiBinary
from gym.spaces import Box as GymBox
import numpy as np

from abmarl.tools import Box
from abmarl.sim.wrappers import FlattenWrapper, FlattenActionWrapper
from abmarl.sim.wrappers.flatten_wrapper import flatdim, flatten, unflatten, flatten_space
from abmarl.examples import MultiAgentContinuousGymSpaceSim

# --- Test flatten helper commands --- #

box = Box(2, 16, (3,4), int)
box2 = GymBox(2.4, 16.1, (3,4))
discrete = Discrete(11)
multi_binary = MultiBinary(7)
multi_discrete = MultiDiscrete([2, 6, 4])
d = Dict({'first': box2, 'second': multi_binary})
t = Tuple((discrete, box2, multi_discrete))
combo = Tuple((Dict({'first': discrete, 'second': box}), multi_binary))


def test_integer_sample():
    flat_combo = flatten_space(combo)
    samp = flat_combo.sample()
    for i in samp:
        assert type(i) is np.int64


def test_flatdim():
    assert flatdim(box) == 12
    assert flatdim(discrete) == 1
    assert flatdim(multi_binary) == 7
    assert flatdim(multi_discrete) == 3
    assert flatdim(d) == 19
    assert flatdim(t) == 16
    assert flatdim(combo) == 20


def test_flatten_and_unflatten():
    box_sample = np.array([
        [2, 12, 6, 6],
        [2, 8, 5, 2],
        [2, 16, 4, 10]
    ])
    flattened_box_sample = np.array([
        2, 12, 6, 6, 2, 8, 5, 2, 2, 16, 4, 10
    ])

    box2_sample = np.array([
        [1.570356,   8.745239,  15.145958,   6.7516246],
        [ 4.981189,   8.553591,   4.252773,  13.892432 ],
        [ 9.568261,  13.409188,   3.5828538,  3.1473527]
    ])
    flattened_box2_sample = np.array([
        1.570356, 8.745239, 15.145958, 6.7516246, 4.981189, 8.553591, 4.252773,
        13.892432, 9.568261, 13.409188, 3.5828538, 3.1473527
    ])

    d_sample = {
        'first': np.array([
            [6.7937455, 14.242623 ,  8.434648 ,  3.721371 ],
            [13.182184 ,  2.9882407, 11.972553 , 10.094532 ],
            [ 7.314138 ,  8.059702 , 13.132411 ,  6.1159306]]
        ),
        'second': np.array([1, 1, 0, 0, 0, 0, 1])
    }
    flattened_d_sample = np.array([
        6.79374552, 14.24262333, 8.43464756, 3.72137094, 13.18218422, 2.98824072,
        11.97255325, 10.09453201, 7.31413794, 8.05970192, 13.132411, 6.11593056,
        1., 1., 0., 0., 0., 0., 1.
    ])

    t_sample = (
        6,
        np.array([
            [2.8780882, 9.45528, 7.886346, 5.2954407],
            [15.226127, 11.98857, 3.2817354, 7.7292967],
            [12.403905, 8.250312, 10.714847, 9.248549 ]
        ]),
        np.array([1, 1, 3])
    )
    flattened_t_sample = np.array([
        6., 2.87808824, 9.4552803, 7.88634586,
        5.29544067, 15.22612667, 11.98857021, 3.28173542, 7.72929668, 12.40390491,
        8.25031185, 10.71484661, 9.24854946, 1., 1., 3.
    ])

    combo_sample = (
        {
            'first': 2,
            'second': np.array([
                [15, 8, 10, 3],
                [14, 10, 7, 7],
                [7, 14, 4, 10]
            ])
        },
        np.array([1, 1, 1, 0, 1, 1, 0])
    )
    flattened_combo_sample = np.array([
        2, 15, 8, 10, 3, 14, 10, 7, 7, 7, 14, 4,
        10, 1, 1, 1, 0, 1, 1, 0
    ])

    # Test flattens
    np.testing.assert_array_equal(flatten(box, box_sample), flattened_box_sample)
    assert np.allclose(flatten(box2, box2_sample), flattened_box2_sample, atol=1.e-6)
    np.testing.assert_array_equal(
        flatten(discrete, 8), np.array([8])
    )
    np.testing.assert_array_equal(
        flatten(multi_binary, [0, 1, 0, 0, 1, 1, 1]), np.array([0, 1, 0, 0, 1, 1, 1])
    )
    np.testing.assert_array_equal(flatten(multi_discrete, [0, 3, 1]), np.array([0, 3, 1]))
    assert np.allclose(flatten(d, d_sample), flattened_d_sample, atol=1.e-8)
    assert np.allclose(flatten(t, t_sample), flattened_t_sample, atol=1.e-8)
    np.testing.assert_array_equal(flatten(combo, combo_sample), flattened_combo_sample)

    # Test unflattens
    assert np.allclose(unflatten(box2, flattened_box2_sample), box2_sample, atol=1.e-6)
    np.testing.assert_array_equal(
        unflatten(discrete, np.array([8])), 8
    )
    np.testing.assert_array_equal(
        unflatten(multi_binary, np.array([0, 1, 0, 0, 1, 1, 1])), [0, 1, 0, 0, 1, 1, 1]
    )
    np.testing.assert_array_equal(unflatten(multi_discrete, np.array([0, 3, 1])), [0, 3, 1])
    assert np.allclose(unflatten(d, flattened_d_sample)['first'], d_sample['first'], atol=1.e-8)
    assert np.allclose(unflatten(d, flattened_d_sample)['second'], d_sample['second'], atol=1.e-8)
    assert unflatten(t, flattened_t_sample)[0] == 6
    assert np.allclose(unflatten(t, flattened_t_sample)[1], t_sample[1], atol=1.e-8)
    assert np.allclose(unflatten(t, flattened_t_sample)[2], t_sample[2], atol=1.e-8)
    assert unflatten(combo, flattened_combo_sample)[0]['first'] == 2
    np.testing.assert_array_equal(
        unflatten(combo, flattened_combo_sample)[0]['second'], combo_sample[0]['second']
    )
    np.testing.assert_array_equal(unflatten(combo, flattened_combo_sample)[1], combo_sample[1])


def test_flatten_space():
    flattened_box_space = flatten_space(box)
    assert flattened_box_space == Box(2, 16, (12,), int)
    flattened_box2_space = flatten_space(box2)
    assert flattened_box2_space == Box(2.4, 16.1, (12,))
    flattened_discrete_space = flatten_space(discrete)
    assert flattened_discrete_space == Box(0, 10, (1,), int)
    flattened_multi_binary_space = flatten_space(multi_binary)
    assert flattened_multi_binary_space == Box(0, 1, (7,), int)
    flattened_multi_discrete_space = flatten_space(multi_discrete)
    assert flattened_multi_discrete_space == Box(
        np.array([0, 0, 0]), np.array([1, 5, 3]), (3,), int
    )
    flattened_d_space = flatten_space(d)
    assert flattened_d_space == Box(
        np.array([
            2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 0, 0, 0, 0, 0, 0, 0
        ]),
        np.array([
            16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 1, 1, 1, 1, 1,
            1, 1
        ]),
        (19,)
    )
    flattened_t_space = flatten_space(t)
    assert flattened_t_space == Box(
        np.array([
            0, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4,
            2.4, 0, 0, 0
        ]),
        np.array([
            10, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1, 16.1,
            16.1, 16.1, 16.1, 1, 5, 3
        ]),
        (16,)
    )
    flattened_combo_space = flatten_space(combo)
    assert flattened_combo_space == Box(
        np.array([
            0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
            0
        ]),
        np.array([
            10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16,  1,  1,  1,  1,  1,  1,  1
        ]),
        (20,),
        int
    )


# --- Test flatten wrappers --- #
def test_flatten_wrapper():
    sim = MultiAgentContinuousGymSpaceSim()
    wrapped_sim = FlattenWrapper(sim)
    assert wrapped_sim.unwrapped == sim
    for agent_id in wrapped_sim.agents:
        assert isinstance(wrapped_sim.agents[agent_id].observation_space, Box)
        assert isinstance(wrapped_sim.agents[agent_id].action_space, Box)
    sim = wrapped_sim
    sim.reset()

    action_0 = {
        'agent0': ({'first': 2, 'second': [-0.24, 1.9]}, [0, 1, 1]),
        'agent1': [3, 2, 0],
        'agent2': {'alpha': [1, 1, 0]},
        'agent3': (2, np.array([0., 6.]), 1)
    }
    sim.step({
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_0.items()
    })
    np.testing.assert_array_equal(sim.get_obs('agent0'), [0, 1, 1, 0])
    assert sim.get_obs('agent1') == 0.98
    np.testing.assert_array_equal(sim.get_obs('agent2'), [1, 0])
    np.testing.assert_array_equal(sim.get_obs('agent3'), np.array([1, -1, 1]))

    assert sim.get_reward('agent0') == 'Reward from agent0'
    assert sim.get_reward('agent1') == 'Reward from agent1'
    assert sim.get_reward('agent2') == 'Reward from agent2'
    assert sim.get_reward('agent3') == 'Reward from agent3'

    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_done('agent3') == 'Done from agent3'
    assert sim.get_all_done() == "Done from all agents and/or simulation."

    assert sim.get_info('agent0')[0]['first'] == action_0['agent0'][0]['first']
    assert np.allclose(
        sim.get_info('agent0')[0]['second'], action_0['agent0'][0]['second'], atol=1.0e-7
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_0['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_0['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_0['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_0['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_0['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_0['agent3'][2])


    action_1 = {
        'agent0': ({'first': 0, 'second': [0.6, -0.2]}, [1, 1, 1]),
        'agent1': [1, 5, 1],
        'agent2': {'alpha': [1, 0, 0]},
        'agent3': (1, np.array([9., 4.]), 0)
    }
    sim.step({
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_1.items()
    })
    assert sim.get_info('agent0')[0]['first'] == action_1['agent0'][0]['first']
    assert np.allclose(
        sim.get_info('agent0')[0]['second'], action_1['agent0'][0]['second'], atol=1.0e-7
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_1['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_1['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_1['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_1['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_1['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_1['agent3'][2])


    action_2 = {
        'agent0': ({'first': 1, 'second': [2.2, 0.98]}, [0, 0, 1]),
        'agent1': [2, 0, 1],
        'agent2': {'alpha': [0, 0, 0]},
        'agent3': (0, np.array([7., 7.]), 0)
    }
    sim.step({
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_2.items()
    })
    assert sim.get_info('agent0')[0]['first'] == action_2['agent0'][0]['first']
    assert np.allclose(
        sim.get_info('agent0')[0]['second'], action_2['agent0'][0]['second'], atol=1.0e-7)

    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_2['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_2['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_2['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_2['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_2['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_2['agent3'][2])


def test_flatten_null_points():
    abs = MultiAgentContinuousGymSpaceSim()
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

    sim = FlattenWrapper(abs)
    agents = sim.agents

    np.testing.assert_array_equal(agents['agent0'].null_observation, [0, 0, 0, 0])
    np.testing.assert_array_equal(agents['agent0'].null_action, [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(agents['agent1'].null_observation, [0])
    np.testing.assert_array_equal(agents['agent1'].null_action, [0, 0, 0])
    np.testing.assert_array_equal(agents['agent2'].null_observation, [0, 0])
    np.testing.assert_array_equal(agents['agent2'].null_action, [0, 0, 0])
    np.testing.assert_array_equal(agents['agent3'].null_observation, [0, 0, 0])
    np.testing.assert_array_equal(agents['agent3'].null_action, [0, 0, 0, 0])


def test_flatten_action_null_points():
    abs = MultiAgentContinuousGymSpaceSim()
    agents = abs.agents
    agents['agent0'].null_action = ({'first': 0, 'second': [0, 0]}, [0, 0, 0])
    assert agents['agent0'].null_action in agents['agent0'].action_space
    agents['agent1'].null_action = [0, 0, 0]
    assert agents['agent1'].null_action in agents['agent1'].action_space
    agents['agent2'].null_action = {'alpha': [0, 0, 0]}
    assert agents['agent2'].null_action in agents['agent2'].action_space
    agents['agent3'].null_action = (0, [0, 0], 0)
    assert agents['agent3'].null_action in agents['agent3'].action_space

    sim = FlattenActionWrapper(abs)
    agents = sim.agents

    np.testing.assert_array_equal(agents['agent0'].null_action, [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(agents['agent1'].null_action, [0, 0, 0])
    np.testing.assert_array_equal(agents['agent2'].null_action, [0, 0, 0])
    np.testing.assert_array_equal(agents['agent3'].null_action, [0, 0, 0, 0])


def test_flatten_sample_in_space():
    flattened_combo_sample = flatten_space(combo).sample()
    unflattened_combo_sample = unflatten(combo, flattened_combo_sample)
    assert unflattened_combo_sample in combo

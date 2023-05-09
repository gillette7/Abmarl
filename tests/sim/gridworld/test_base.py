
import os

import numpy as np
import pytest

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.examples.sim import MultiAgentGridSim


def test_build():
    sim = MultiAgentGridSim.build_sim(3, 4, agents={})
    assert sim.agents == {}
    assert isinstance(sim.grid, Grid)
    assert sim.grid.rows == 3
    assert sim.grid.cols == 4
    np.testing.assert_array_equal(sim.grid._internal, np.empty((3, 4), dtype=object))

    sim.reset()
    np.testing.assert_array_equal(
        sim.grid._internal,  np.array([
            [{}, {}, {}, {}],
            [{}, {}, {}, {}],
            [{}, {}, {}, {}]
        ])
    )

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(3.0, 4)

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(0, 4)

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(3, -4)

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(3, '4')


def test_build_from_grid():
    grid = Grid(2, 2)
    grid.reset()
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0])),
        'agent1': GridWorldAgent(id='agent1', encoding=1, initial_position=np.array([0, 1])),
        'agent2': GridWorldAgent(id='agent2', encoding=1, initial_position=np.array([1, 0])),
        'agent3': GridWorldAgent(id='agent3', encoding=1, initial_position=np.array([1, 1])),
    }
    grid.place(agents['agent0'], (0, 0))
    grid.place(agents['agent1'], (0, 1))
    grid.place(agents['agent2'], (1, 0))
    grid.place(agents['agent3'], (1, 1))

    sim = MultiAgentGridSim.build_sim_from_grid(grid)
    sim.reset()
    assert sim.agents == {
        'agent0': agents['agent0'],
        'agent1': agents['agent1'],
        'agent2': agents['agent2'],
        'agent3': agents['agent3'],
    }
    np.testing.assert_array_equal(
        sim.agents['agent0'].initial_position,
        np.array([0, 0])
    )
    np.testing.assert_array_equal(
        sim.agents['agent1'].initial_position,
        np.array([0, 1])
    )
    np.testing.assert_array_equal(
        sim.agents['agent2'].initial_position,
        np.array([1, 0])
    )
    np.testing.assert_array_equal(
        sim.agents['agent3'].initial_position,
        np.array([1, 1])
    )
    assert next(iter(sim.grid[0, 0].values())) == agents['agent0']
    assert next(iter(sim.grid[0, 1].values())) == agents['agent1']
    assert next(iter(sim.grid[1, 0].values())) == agents['agent2']
    assert next(iter(sim.grid[1, 1].values())) == agents['agent3']

    with pytest.raises(AssertionError):
        # This fails because the grid must be a grid object, not an array
        MultiAgentGridSim.build_sim_from_grid(grid._internal)

    with pytest.raises(AssertionError):
        # This fails becaue the agents' initial positions must match their index
        # within the grid.
        agents['agent1'].initial_position = np.array([1, 0])
        agents['agent2'].initial_position = np.array([0, 1])
        MultiAgentGridSim.build_sim_from_grid(grid)


def test_build_sim_from_array():
    array = np.array([
        ['A', '.', 'B', '0', ''],
        ['B', '_', '', 'C', 'A']
    ])
    obj_registry = {
        'A': lambda n: GridWorldAgent(
            id=f'A-class-barrier{n}',
            encoding=1,
        ),
        'B': lambda n: GridWorldAgent(
            id=f'B-class-barrier{n}',
            encoding=2,
        ),
        'C': lambda n: GridWorldAgent(
            id=f'C-class-barrier{n}',
            encoding=3,
        ),
    }
    sim = MultiAgentGridSim.build_sim_from_array(array, obj_registry)
    assert sim.grid.rows == 2
    assert sim.grid.cols == 5
    np.testing.assert_array_equal(sim.grid._internal, np.empty((2, 5), dtype=object))

    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert sim.grid[0, 4] == {}
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert 'C-class-barrier3' in sim.grid[1, 3]
    assert 'A-class-barrier4' in sim.grid[1, 4]

    assert len(sim.agents) == 5
    assert sim.agents['A-class-barrier0'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier0'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['B-class-barrier1'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier1'].initial_position,
        np.array([0, 2])
    )
    assert sim.agents['B-class-barrier2'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier2'].initial_position,
        np.array([1, 0])
    )
    assert sim.agents['C-class-barrier3'].encoding == 3
    np.testing.assert_array_equal(
        sim.agents['C-class-barrier3'].initial_position,
        np.array([1, 3])
    )
    assert sim.agents['A-class-barrier4'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier4'].initial_position,
        np.array([1, 4])
    )

    # Testin what happens when one of the keys is not in the registry
    del obj_registry['C']
    sim = MultiAgentGridSim.build_sim_from_array(array, obj_registry)
    assert sim.grid.rows == 2
    assert sim.grid.cols == 5

    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert sim.grid[0, 4] == {}
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert sim.grid[1, 3] == {}
    assert 'A-class-barrier3' in sim.grid[1, 4]

    assert len(sim.agents) == 4
    assert 'C-class-barrier3' not in sim.agents
    assert sim.agents['A-class-barrier0'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier0'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['B-class-barrier1'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier1'].initial_position,
        np.array([0, 2])
    )
    assert sim.agents['B-class-barrier2'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier2'].initial_position,
        np.array([1, 0])
    )
    assert sim.agents['A-class-barrier3'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier3'].initial_position,
        np.array([1, 4])
    )

    # Bad array
    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_array(obj_registry, obj_registry)
    # Bad Object Registry
    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_array(array, array)
    # Using reserved key
    with pytest.raises(AssertionError):
        obj_registry.update({
            '_': lambda n: GridWorldAgent(
                id='invalid_underscore!',
                encoding=0,
            ),
        })
        MultiAgentGridSim.build_sim_from_array(array, obj_registry)


def test_build_sim_from_file():
    file_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'grid_file.txt'
    )
    obj_registry = {
        'A': lambda n: GridWorldAgent(
            id=f'A-class-barrier{n}',
            encoding=1,
        ),
        'B': lambda n: GridWorldAgent(
            id=f'B-class-barrier{n}',
            encoding=2,
        ),
        'C': lambda n: GridWorldAgent(
            id=f'C-class-barrier{n}',
            encoding=3,
        ),
    }
    sim = MultiAgentGridSim.build_sim_from_file(file_name, obj_registry)
    assert sim.grid.rows == 2
    assert sim.grid.cols == 5
    np.testing.assert_array_equal(sim.grid._internal, np.empty((2, 5), dtype=object))

    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert sim.grid[0, 4] == {}
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert 'C-class-barrier3' in sim.grid[1, 3]
    assert 'A-class-barrier4' in sim.grid[1, 4]

    assert len(sim.agents) == 5
    assert sim.agents['A-class-barrier0'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier0'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['B-class-barrier1'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier1'].initial_position,
        np.array([0, 2])
    )
    assert sim.agents['B-class-barrier2'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier2'].initial_position,
        np.array([1, 0])
    )
    assert sim.agents['C-class-barrier3'].encoding == 3
    np.testing.assert_array_equal(
        sim.agents['C-class-barrier3'].initial_position,
        np.array([1, 3])
    )
    assert sim.agents['A-class-barrier4'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier4'].initial_position,
        np.array([1, 4])
    )

    # Testin what happens when one of the keys is not in the registry
    del obj_registry['C']
    sim = MultiAgentGridSim.build_sim_from_file(file_name, obj_registry)
    assert sim.grid.rows == 2
    assert sim.grid.cols == 5

    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert sim.grid[0, 4] == {}
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert sim.grid[1, 3] == {}
    assert 'A-class-barrier3' in sim.grid[1, 4]

    assert len(sim.agents) == 4
    assert 'C-class-barrier3' not in sim.agents
    assert sim.agents['A-class-barrier0'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier0'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['B-class-barrier1'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier1'].initial_position,
        np.array([0, 2])
    )
    assert sim.agents['B-class-barrier2'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier2'].initial_position,
        np.array([1, 0])
    )
    assert sim.agents['A-class-barrier3'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier3'].initial_position,
        np.array([1, 4])
    )

    # Bad array
    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_file(obj_registry, obj_registry)
    # Bad Object Registry
    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_file(file_name, file_name)
    # Using reserved key
    with pytest.raises(AssertionError):
        obj_registry.update({
            '_': lambda n: GridWorldAgent(
                id='invalid_underscore!',
                encoding=0,
            ),
        })
        MultiAgentGridSim.build_sim_from_file(file_name, obj_registry)

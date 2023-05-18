
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent, GridWorldAgent
from abmarl.sim.gridworld.agent import HealthAgent
import abmarl.sim.gridworld.utils as gu


class StateBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract State Component base from which all state components will inherit.
    """
    @abstractmethod
    def reset(self, **kwargs):
        """
        Resets the part of the state for which it is responsible.
        """
        pass


class PositionState(StateBaseComponent):
    """
    Manage the agents' positions in the grid.
    """
    @property
    def ravelled_positions_available(self):
        """
        A dictionary mapping the enodings to a list of positions available to
        agents of that encoding at reset. The list should contain cells represented
        in their ravelled form.
        """
        return self._ravelled_positions_available

    @ravelled_positions_available.setter
    def ravelled_positions_available(self, value):
        assert type(value) is dict, "Ravelled Positions available must be a dictionary."
        for encoding, ravelled_positions_available in value.items():
            assert type(encoding) is int, "Ravelled Position keys must be integers."
            assert type(ravelled_positions_available) is list, \
                "Ravelled Position values must be lists."
            for ravelled_cell in ravelled_positions_available:
                assert type(ravelled_cell) is int, "Available cells must be integers. " \
                    "They should be the ravelled presentation of the cell."
        self._ravelled_positions_available = value

    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agents in the grid.
        """
        self.grid.reset()

        # Build lists of available positions
        self._build_available_positions()

        # Place agents with initial positions.
        for agent in self.agents.values():
            if agent.initial_position is not None:
                self._place_initial_position_agent(agent)

        # Now place agents with variable positions
        for agent in self.agents.values():
            if agent.initial_position is None:
                self._place_variable_position_agent(agent)

    def _build_available_positions(self, **kwargs):
        """
        Define the positions that are available per encoding.
        """
        max_encoding = max([agent.encoding for agent in self.agents.values()])
        self.ravelled_positions_available = {
            encoding: [i for i in range(self.rows * self.cols)]
            for encoding in range(1, max_encoding + 1)
        }

    def _update_available_positions(self, agent_just_placed, **kwargs):
        """
        Update the available positions based on the agent that was just placed.
        """
        for encoding, positions_available in self.ravelled_positions_available.items():
            # Remove this cell from any encoding where overlapping is False
            if encoding not in self.grid.overlapping.get(agent_just_placed.encoding, {}):
                try:
                    positions_available.remove(
                        np.ravel_multi_index(agent_just_placed.position, (self.rows, self.cols))
                    )
                except KeyError:
                    # Catch a key error because this cell might have already
                    # been removed from this encoding
                    continue

    def _place_initial_position_agent(self, ip_agent_to_place, **kwargs):
        """
        Place an agent with an initial position.
        """
        ravelled_initial_position = np.ravel_multi_index(
            ip_agent_to_place.initial_position,
            (self.rows, self.cols)
        )
        assert ravelled_initial_position in \
            self.ravelled_positions_available[ip_agent_to_place.encoding], \
            f"Cell {ip_agent_to_place.initial_position} is not available for " \
            f"{ip_agent_to_place.id}."
        assert self.grid.place(ip_agent_to_place, ip_agent_to_place.initial_position)
        self._update_available_positions(ip_agent_to_place)

    def _place_variable_position_agent(self, var_agent_to_place, **kwargs):
        """
        Place an agent with a variable position.

        This implementation randomly places the agent in one of its available cells.
        """
        try:
            ravelled_position = np.random.choice(
                [*self.ravelled_positions_available[var_agent_to_place.encoding]], 1)
        except ValueError:
            raise RuntimeError(f"Could not find a cell for {var_agent_to_place.id}") from None
        else:
            r, c = np.unravel_index(ravelled_position.item(), shape=(self.rows, self.cols))
            assert self.grid.place(var_agent_to_place, (r, c))
            self._update_available_positions(var_agent_to_place)


class MazePlacementState(PositionState):
    """
    Place agents in the grid based on a maze generated around a target.
    # TODO: Add support for target agent?

    Partition the cells into two categories, either a free cell or a wall, based
    on a maze. If one of the agents is specified as a target, then use that agent
    as the starting point of the maze. Specify available positions as follows: barrier-encoded
    agents will be placed at the maze walls, free-encoded agents will be placed at free positions.

    Note: Because the maze is randomly generated at the beginning of each episode
    and because the agents must be placed in either a free cell or barrier cell
    according to their encodings, it is highly recommended that none of your agents
    be given initial positions, except for the target agent.
    # TODO: If we don't force a complete division into two categories, then we can
    # better support agents with initial positions that are neither free nor barrier.

    Args:
        target_agent: Start the maze generation at this agent's position and place
            the target agent here.
        barrier_encodings: A set of encodings corresponding to the maze's barrier cells.
        free_encodings: A set of encodigns corresponding to the maze's free cells.
        barriers_near_target: Prioritize the placement of barriers near the target.
        free_agents_far_from_target: Prioritize the placement of free agents away from
            the target.
    """
    def __init__(self,
                 target_agent=None,
                 barrier_encodings=None,
                 free_encodings=None,
                 barriers_near_target=False,
                 free_agents_far_from_target=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_agent=target_agent
        self.barrier_encodings=barrier_encodings
        self.free_encodings=free_encodings
        self.barriers_near_target = barriers_near_target
        self.free_agents_far_from_target = free_agents_far_from_target

    @property
    def target_agent(self):
        """
        The target agent is the place from which to start the maze generation.
        """
        return self._target_agent
    
    @target_agent.setter
    def target_agent(self, value):
        assert isinstance(value, GridWorldAgent), "Target agent must be a GridWorld agent."
        assert value in self.agents.values(), "The target agent must be an agent in the simulation."
        self._target_agent = value

    @property
    def barrier_encodings(self):
        """
        A set of encodings corresponding to the maze's barrier cells.
        """
        return self._barrier_encodings

    @barrier_encodings.setter
    def barrier_encodings(self, value):
        if value is not None:
            assert type(value) is set, "Barrier encodings must be a set."
            for encoding in value:
                assert type(encoding) is int, "Each barrier encoding must be an integer."
            self._barrier_encodings = value
        else:
            self._barrier_encodings = set()

    @property
    def free_encodings(self):
        """
        A set of encodings corresponding to the maze's free cells.
        """
        return self._free_encodings

    @free_encodings.setter
    def free_encodings(self, value):
        if value is not None:
            assert type(value) is set, "Free encodings must be a set."
            for encoding in value:
                assert type(encoding) is int, "Each free encoding must be an integer."
            self._free_encodings = value
        else:
            self._free_encodings = set()


    # TODO: Change options to "cluster barriers" and "scatter free agents"
    @property
    def barriers_near_target(self):
        """
        If True, then prioritize placing barriers near the target agent.
        """
        return self._barriers_near_target

    @barriers_near_target.setter
    def barriers_near_target(self, value):
        assert type(value) is bool, "Barriers near target must be a boolean."
        self._barriers_near_target = value

    @property
    def free_agents_far_from_target(self):
        """
        If True, then prioritize placing free agents away from the target agent.
        """
        return self._free_agents_far_from_target

    @free_agents_far_from_target.setter
    def free_agents_far_from_target(self, value):
        assert type(value) is bool, "Free agents far from target must be a boolean."
        self._free_agents_far_from_target = value

    def reset(self, **kwargs):
        """
        Give the agents their starting positions.
        """
        self.grid.reset()

        # Build lists of available encodings
        self._build_available_positions()

        # Manually place target agent at the maze start
        assert self.grid.place(self.target_agent, self._maze_start), "Unable to place target agent."
        self._update_available_positions(self.target)

        # Place agents with initial positions.
        for agent in self.agents.values():
            if agent == self.target_agent:
                continue
            if agent.initial_position is not None:
                self._place_initial_position_agent(agent)

        # Now place barrier + free agents with variable positions
        for agent in self.agents.values():
            if agent.encoding not in {**self.free_encodings, **self.barrier_encodings}:
                continue
            if agent == self.target_agent:
                continue
            if agent.initial_position is None:
                self._place_variable_position_agent(agent)

        # Now place all other agents with variable positions
        for agent in self.agents.values():
            if agent.encoding in {**self.free_encodings, **self.barrier_encodings}:
                continue
            if agent == self.target_agent:
                continue
            if agent.initial_position is None:
                self._place_variable_position_agent(agent)

    def _build_available_positions(self, **kwargs):
        """
        Define the positions available per encoding.

        The avaiable positions is based on maze generated
        starting from the target's position, if it exists. This maze divides the
        cells into two categories: free and barrier. If an agent has a barrier encoding,
        then it can only be placed at a barrier cell. If an agent has a free encoding,
        then it can only be placed at a free cell.
        """
        # Start by setting the the whole grid as available for all the agents
        # TODO: Support agents that are not just barrier or free
        # super()._build_available_positions()

        if self.target_agent.initial_position:
            self._maze_start = self.target_agent.initial_position
        else:
            n = np.random.randint(0, self.rows * self.cols)
            r, c = np.unravel_index(n, shape=(self.rows, self.cols))
            self._maze_start = (r, c)
        ravelled_maze_start = np.ravel_multi_index(self._maze_start, (self.rows, self.cols))
        maze = gu.generate_maze(self.rows, self.cols, self._maze_start)

        ravelled_barrier_positions = set()
        barrier_positions = np.where(maze == 1)
        for ndx in range(barrier_positions[0]):
            r, c = barrier_positions[0][ndx], barrier_positions[1][ndx]
            n = np.ravel_multi_index((r, c), (self.rows, self.cols))
            ravelled_barrier_positions.append(n)
        ravelled_barrier_positions.sort(key=lambda x: abs(x - ravelled_maze_start), reverse=True)

        ravelled_free_positions = set()
        free_positions = np.where(maze == 0)
        for ndx in range(free_positions[0]):
            r, c = free_positions[0][ndx], free_positions[1][ndx]
            n = np.ravel_multi_index((r, c), (self.rows, self.cols))
            ravelled_free_positions.append(n)
        ravelled_free_positions.sort(key=lambda x: abs(x - ravelled_maze_start))

        # TODO: Update so that we only modify the encodings, don't build it from scratch.
        self.ravelled_positions_available = {
            **{encoding: ravelled_barrier_positions for encoding in self.barrier_encodings},
            **{encoding: ravelled_free_positions for encoding in self.free_encoding}
        }

    def _place_variable_position_agent(self, var_agent_to_place, **kwargs):
        """
        Place an agent with a variable position.

        This implementation places the agents according to their available positions,
        either a free cell or a barrier cell. Barriers agents will be clustered
        around the maze's starting position if barriers_near_target is True. Free
        agents will be scattered far from the maze's starting position if
        free_agents_far_from_target is True.
        """
        if (var_agent_to_place.encoding in self.barrier_encodings and self.barriers_near_target) \
            or (var_agent_to_place.encoding in self.free_encodings \
                and self.free_agents_far_from_target):
            try:
                ravelled_position = self.ravelled_positions_available[var_agent_to_place.encoding].pop()
            except ValueError:
                raise RuntimeError(f"Could not find a cell for {var_agent_to_place.id}") from None
            else:
                r, c = np.unravel_index(ravelled_position, shape=(self.rows, self.cols))
                assert self.grid.place(var_agent_to_place, (r, c))
                self._update_available_positions(var_agent_to_place)
        else:
            super()._place_variable_position_agent(var_agent_to_place)


class HealthState(StateBaseComponent):
    """
    Manage the state of the agents' healths.

    Every HealthAgent has a health. If that health falls to zero, that agent dies
    and is remove from the grid.
    """
    def reset(self, **kwargs):
        """
        Give HealthAgents their starting healths.

        We use the agent's initial health if it exists. Otherwise, we randomly
        assign a value between 0 and 1.
        """
        for agent in self.agents.values():
            if isinstance(agent, HealthAgent):
                if agent.initial_health is not None:
                    agent.health = agent.initial_health
                else:
                    agent.health = np.random.uniform(0, 1)

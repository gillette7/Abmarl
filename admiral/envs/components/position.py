
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.agent import PositionAgent
from admiral.envs.components.agent import AgentObservingAgent
from admiral.envs.components.agent import TeamAgent

class PositionState:
    """
    Manages the agents' positions. All position updates must be within the region.

    region (int):
        The size of the environment.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the agents' positions. If the agents were created with a starting
        position, then use that. Otherwise, randomly assign a position in the region.
        """
        for agent in self.agents.values():
            if isinstance(agent, PositionAgent):
                if agent.starting_position is not None:
                    agent.position = agent.starting_position
                else:
                    agent.position = np.random.randint(0, self.region, 2)
    
    def set_position(self, agent, _position, **kwargs):
        """
        Set the agent's position to the incoming value only if the new position
        is within the region.
        """
        if isinstance(agent, PositionAgent):
            if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
                agent.position = _position
    
    def modify_position(self, agent, value, **kwargs):
        """
        Add some value to the position of the agent.
        """
        if isinstance(agent, PositionAgent):
            self.set_position(agent, agent.position + value)

class PositionObserver:
    """
    Observe the positions of all the agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['position'] = Dict({
                other.id: Box(-1, self.position.region, (2,), np.int) for other in agents.values() if isinstance(other, PositionAgent)
            })

    def get_obs(self, agent, **kwargs):
        """
        Get the positions of all the agents in the simulator.
        """
        return {'position': {other.id: other.position for other in self.agents.values() if isinstance(other, PositionAgent)}}
    
    @property
    def null_value(self):
        return np.array([-1, -1])

class RelativePositionObserver:
    """
    Observe the relative positions of agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, PositionAgent):
                agent.observation_space['position'] = Dict({
                    other.id: Box(-position.region, position.region, (2,), np.int) for other in agents.values() if (other.id != agent.id and isinstance(other, PositionAgent))
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the relative positions of all the agents in the simulator.
        """
        if isinstance(agent, PositionAgent):
            obs = {}
            for other in self.agents.values():
                if other.id == agent.id: continue # Don't observe your own position
                if not isinstance(other, PositionAgent): continue # Can't observe relative position from agents who do not have a position.
                r_diff = other.position[0] - agent.position[0]
                c_diff = other.position[1] - agent.position[1]
                obs[other.id] = np.array([r_diff, c_diff])
            return {'position': obs}
        else:
            return {}
    
    @property
    def null_value(self):
        return np.array([-self.position.region, -self.position.region])

class GridPositionBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
    position. The values of the cells are as such:
        Out of bounds  : -1
        Empty          :  0
        Agent occupied : 1
    
    position (PositionState):
        The position state handler, which contains the region.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionAgent):
                agent.observation_space['position'] = Box(-1, 1, (agent.agent_view*2+1, agent.agent_view*2+1), np.int)

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position.
        """
        if isinstance(my_agent, AgentObservingAgent) and isinstance(my_agent, PositionAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.agent_view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.agent_view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.agent_view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.agent_view - 1:] = -1

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not isinstance(other_agent, PositionAgent): continue # Can only observer position of PositionAgents
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff] = 1 # There is an agent at this location.

            return {'position': signal}
        else:
            return {}

class GridPositionTeamBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
    position. The observation contains one channel per team, where the value of
    the cell is the number of agents on that team that occupy that square. -1
    indicates out of bounds.
    
    position (PositionState):
        The position state handler, which contains the region.

    team_state (TeamState):
        The team state handler, which contains the number of teams.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, team_state=None, agents=None, **kwargs):
        self.position = position
        self.team_state = team_state
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, TeamAgent)
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, AgentObservingAgent) and isinstance(agent, PositionAgent):
                agent.observation_space['position'] = Box(-1, np.inf, (agent.agent_view*2+1, agent.agent_view*2+1, self.team_state.number_of_teams), np.int)
    
    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position. Each team has its own channel and the value represents the number
        of agents of that team occupying the same square.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, TeamAgent) and \
           isinstance(my_agent, PositionAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.agent_view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.agent_view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.agent_view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.agent_view - 1:] = -1

            # Repeat the boundaries signal for all teams
            signal = np.repeat(signal[:, :, np.newaxis], self.team_state.number_of_teams, axis=2)

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not isinstance(other_agent, PositionAgent): continue # Cannot observe agent without position
                if not isinstance(other_agent, TeamAgent): continue # Cannot observe agent without team.
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff, other_agent.team] += 1

            return {'position': signal}
        else:
            return {}

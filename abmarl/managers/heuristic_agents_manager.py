
from abmarl.sim import Agent
from abmarl.policies.policy import Policy


from .simulation_manager import SimulationManager


class ManagerWrapper(SimulationManager):
    """
    Wrap a simulation manager.
    """
    def __init__(self, manager, **kwargs):
        self.manager = manager
        super().__init__(self.manager.sim, **kwargs)

    @property
    def manager(self):
        """
        The internal, wrapped manager.
        """
        return self._manager
    
    @manager.setter
    def manager(self, value):
        assert isinstance(value, SimulationManager), \
            "Wrapped manager must be a Simulation Manager."

    def reset(self, **kwargs):
        return self.manager.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        return self.manager.step(action_dict, **kwargs)
    

class InternallyHeuristicPoliciesManager(ManagerWrapper):
    """
    This manager internally takes care of data generation for agents with heuristic policies.
    """
    def __init__(self, manager, heuristic_agents_mapping=None, **kwargs):
        super().__init__(self, manager, **kwargs)
        self.heuristic_agents_mapping = heuristic_agents_mapping

    @property
    def heuristic_agents_mapping(self):
        """
        Agents with heuristic policies mapped to those policies.
        """
        return self._heuristic_agents_mapping

    @heuristic_agents_mapping.setter
    def heuristic_agents_mapping(self, value):
        assert type(value) is dict, "Heuristic Agents Mapping must be a dictionary."
        for agent, policy in value.items():
            assert type(agent) is str, "Heuristic Agents Mapping must map agent id to its policy."
            assert agent in self.agents, f"{agent} is not in the simulation."
            assert isinstance(policy, Policy), "Policy must be a Policy."
        self._heuristic_agents_mapping = value

    def reset(self, **kwargs):
        """
        Reset simulation and acquire first observation.

        Store any observations of heuristic agents separately without outputting
        them so that the trainer will not attempt to create an action for those
        agents.

        Returns:
            The first observations of the agent(s), minus any observation generated
                for agents using a heuristic policy.
        """
        self.heuristic_agent_obs = {}
        obs = self.manager.reset(**kwargs)
        for agent_id, agent_obs in obs.items().copy():
            if agent_id in self.heuristic_agents_mapping:
                self.heuristic_agent_obs[agent_id] = agent_obs
                del obs[agent_id]

    def step(self, action_dict, **kwargs):
        """
        Step the simulation forward one discrete time-step.

        Generate an action for any heuristic agent whose observation we are currently
        storing using the policying from the heuristic agents policy mapping.
        This mimics what the trainer would do if it had received those agents'
        observations. Add those to any other actions in the dictionary and pass
        it along to the wrapped manager. Any resulting observations from agents'
        with heuristic policies will be stored and used in the next step. Those
        observations, rewards, dones, and infos are removed from the output so
        that the trainer does not receive them.

        Args:
            action_dict: Dictionary mapping agent_ids to actions for any non-heurisitc
                agent.

        Returns:
            The observations, rewards, dones, and info for the agent(s) whose actions
            we expect to receive next, minus any data from heuristic agents.
        """
        for agent_id, agent_obs in self.heuristic_agent_obs.items():
            action_dict[agent_id] = self.heuristic_agents_mapping[agent_id].compute_action(agent_obs)

        obs, rewards, dones, infos = self.manager.step(action_dict, **kwargs)

        self.heuristic_agent_obs = {}
        for agent_id, agent_obs in obs.items().copy():
            if agent_id in self.heuristic_agents_mapping:
                self.heuristic_agent_obs[agent_id] = agent_obs
                del obs[agent_id]
                del rewards[agent_id]
                del dones[agent_id]
                del infos[agent_id]

        return obs, rewards, dones, infos
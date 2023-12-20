
import copy

from abmarl.sim import AgentBasedSimulation
from abmarl.sim.wrappers.wrapper import Wrapper
from abmarl.policies.policy import Policy


class InternallyHeuristicPoliciesWrapper(Wrapper):
    """
    Internally manage data processing for agents with heuristic policies.
    """
    def __init__(self, sim, heuristic_agents_policy=None, **kwargs):
        super.__init__(self, sim, **kwargs)
        self.heuristic_agents_policy = heuristic_agents_policy

    @property
    def heuristic_agents_policy(self):
        """
        Agents with heuristic policies mapped to those policies.
        """
        return self._heuristic_agents_policy

    @heuristic_agents_policy.setter
    def heuristic_agents_policy(self, value):
        assert type(value) is dict, "Heuristic Agents Mapping must be a dictionary."
        for agent, policy in value.items():
            assert type(agent) is str, "Heuristic Agents Mapping must map agent id to its policy."
            assert agent in self.sim.agents, f"{agent} is not in the simulation."
            assert isinstance(policy, Policy), "Policy must be a Policy."
        self._heuristic_agents_policy = value

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.heuristic_agent_obs = {}

    def step(self, action_dict, **kwargs):
        for agent_id in action_dict:
            if agent_id in self.heuristic_agents_policy:
                action_dict[agent_id] = self.heuristic_agents_policy[agent_id].compute_action(
                    self.heuristic_agent_obs[agent_id]
                )
        self.sim.step(action_dict, **kwargs)

    def get_obs(self, agent_id, **kwargs):
        obs = super().get_obs(agent_id, **kwargs)
        if agent_id in self.heuristic_agents_policy:
            self.heuristic_agent_obs[agent_id] = obs
        return obs

    def get_reward(self, agent_id, **kwargs):
        return self.sim.get_reward(agent_id, **kwargs)

    def get_done(self, agent_id, **kwargs):
        return self.sim.get_done(agent_id, **kwargs)

    def get_all_done(self, **kwargs):
        return self.sim.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        return self.sim.get_info(agent_id, **kwargs)


class Number:
    def __init__(self, num=None, **kwags):
        self.num=num

    def __repr__(self) -> str:
        return str(self.num)

a = {i: Number(i) for i in range(4)}
evens_list = [0, 2]
evens_dict = {}
for num in evens_list:
    evens_dict[num] = a[num]
    del a[num]

print(a)
print(evens_dict)
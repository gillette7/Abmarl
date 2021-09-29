
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent
from abmarl.sim.gridworld.state import HealthState, PositionState
from abmarl.sim.gridworld.actor import MoveActor, AttackActor
from abmarl.sim.gridworld.observer import SingleGridObserver
from abmarl.sim.gridworld.done import OneTeamRemainingDone
from abmarl.tools.matplotlib_utils import mscatter

class BattleAgent(GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent):
    pass

class TeamBattleSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State Components
        self.position_state = PositionState(**kwargs)
        self.health_state = HealthState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(**kwargs)
        self.attack_actor = AttackActor(**kwargs)

        # Observation Components
        self.grid_observer = SingleGridObserver(**kwargs)

        # Done Compoennts
        self.done = OneTeamRemainingDone(**kwargs)
        
        self.finalize()
    
    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.health_state.reset(**kwargs)

        # Track the rewards
        self.rewards = {agent.id: 0 for agent in self.agents.values()}

    def step(self, action_dict, **kwargs):
        # Process attacks:
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                self.attack_actor.process_action(agent, action, **kwargs)

        # Process moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                self.move_actor.process_action(agent, action, **kwargs)

    def render(self, fig=None, **kwargs):
        fig.clear()
        ax = fig.gca()

        # Draw the gridlines
        ax.set(xlim=(0, self.position_state.cols), ylim=(0, self.position_state.rows))
        ax.set_xticks(np.arange(0, self.position_state.cols, 1))
        ax.set_yticks(np.arange(0, self.position_state.rows, 1))
        ax.grid()

        # Draw the agents
        agents_x = [
            agent.position[1] + 0.5 for agent in self.agents.values() if agent.active
        ]
        agents_y = [
            self.position_state.rows - 0.5 - agent.position[0]
            for agent in self.agents.values() if agent.active
        ]
        shape = [agent.render_shape for agent in self.agents.values() if agent.active]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')
        # TODO: Color by team

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.grid_observer.get_obs(agent, **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):
        reward = self.rewards[agent_id]
        self.rewards[agent_id] = 0
        return reward

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id])

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        return {}

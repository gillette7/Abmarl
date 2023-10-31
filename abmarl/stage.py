import os

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ray
import ray.rllib
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

from abmarl.tools import utils as adu
from abmarl.managers import SimulationManager



def run_analysis(full_trained_directory, full_subscript, parameters):
    """Analyze MARL policies from a saved policy through an analysis script"""
    # Load the experiment as a module
    # First, we must find the .py file in the directory
    py_files = [file for file in os.listdir(full_trained_directory) if file.endswith('.py')]
    assert len(py_files) == 1
    full_path_to_config = os.path.join(full_trained_directory, py_files[0])
    experiment_mod = adu.custom_import_module(full_path_to_config)
    # Modify the number of workers in the configuration
    experiment_mod.params['ray_tune']['config']['num_workers'] = 1
    experiment_mod.params['ray_tune']['config']['num_envs_per_worker'] = 1
    experiment_mod.params['ray_tune']['config']['seed'] = parameters.seed

    checkpoint_dir, checkpoint_value = adu.checkpoint_from_trained_directory(
        full_trained_directory, parameters.requested_checkpoint
    )
    print(checkpoint_dir)

    # Setup ray
    ray.init()

    # Get the trainer
    alg = get_trainable_cls(experiment_mod.params['ray_tune']['run_or_experiment'])
    trainer = alg(
        env=experiment_mod.params['ray_tune']['config']['env'],
        config=experiment_mod.params['ray_tune']['config']
    )
    trainer.restore(os.path.join(checkpoint_dir, 'checkpoint-' + str(checkpoint_value)))

    # Get the simulation
    sim = experiment_mod.params['experiment']['sim_creator'](
        experiment_mod.params['ray_tune']['config']['env_config']
    )

    # The sim may be wrapped by an external wrapper, which we support, but we need
    # to unwrap it.
    if not isinstance(sim, SimulationManager):
        sim = sim.unwrapped

    # Load the analysis module and run it
    analysis_mod = adu.custom_import_module(full_subscript)
    analysis_mod.run(sim, trainer)

    ray.shutdown()


def visualize(
        params,
        checkpoint=None,
        episodes=1,
        steps_per_episode=200,
        record=False,
        record_only=False,
        frame_delay=200,
        explore=True,
        seed=None
    ):
    full_trained_directory = params['ray_tune']['local_dir']

    if record_only:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("macosx")
        except ImportError:
            matplotlib.use('TkAgg')

    # Modify the number of workers in the configuration
    params['ray_tune']['config']['num_workers'] = 1
    params['ray_tune']['config']['num_envs_per_worker'] = 1
    params['ray_tune']['config']['seed'] = seed

    # Find the checkpoint
    checkpoint_dir, checkpoint_value = adu.checkpoint_from_trained_directory(
        full_trained_directory, checkpoint
    )
    print(checkpoint_dir)

    # Setup ray
    ray.init()

    # Get the trainer
    alg = get_trainable_cls(params['ray_tune']['run_or_experiment'])
    trainer = alg(
        env=params['ray_tune']['config']['env'],
        config=params['ray_tune']['config']
    )
    trainer.restore(os.path.join(checkpoint_dir, 'checkpoint-' + str(checkpoint_value)))

    # Get the simulation
    sim = params['experiment']['sim_creator'](
        params['ray_tune']['config']['env_config']
    )

    # Determine if we are single- or multi-agent case.
    def _multi_get_action(obs, done=None, sim=None, policy_agent_mapping=None, **kwargs):
        joint_action = {}
        if done is None:
            done = {agent: False for agent in obs}
        for agent_id, agent_obs in obs.items():
            if done[agent_id]: continue # Don't get actions for done agents
            policy_id = policy_agent_mapping(agent_id)
            action = trainer.compute_action(
                agent_obs, policy_id=policy_id, explore=explore
            )
            joint_action[agent_id] = action
        return joint_action

    def _single_get_action(obs, trainer=None, **kwargs):
        return trainer.compute_action(obs, explore=explore)

    def _multi_get_done(done):
        return done['__all__']

    def _single_get_done(done):
        return done

    policy_agent_mapping = None
    if isinstance(sim, MultiAgentEnv):
        policy_agent_mapping = trainer.config['multiagent']['policy_mapping_fn']
        _get_action = _multi_get_action
        _get_done = _multi_get_done
    else:
        _get_action = _single_get_action
        _get_done = _single_get_done

    for episode in range(episodes):
        print('Episode: {}'.format(episode))
        obs = sim.reset()
        done = None
        all_done = False
        fig = plt.figure()

        def gen_frame_until_done():
            nonlocal all_done
            i = 0
            while not all_done:
                i += 1
                yield i

        def animate(i):
            nonlocal obs, done
            sim.render(fig=fig)
            if not record_only:
                plt.pause(1e-16)
            action = _get_action(
                obs, done=done, sim=sim, trainer=trainer, policy_agent_mapping=policy_agent_mapping
            )
            obs, _, done, _ = sim.step(action)
            if _get_done(done) or i >= steps_per_episode:
                nonlocal all_done
                all_done = True
                sim.render(fig=fig)
                if not record_only:
                    plt.pause(1e-16)

        anim = FuncAnimation(
            fig, animate, frames=gen_frame_until_done, repeat=False,
            interval=frame_delay, cache_frame_data=False
        )
        if record:
            anim.save(os.path.join(full_trained_directory, 'Episode_{}.gif'.format(episode)))
            plt.show(block=False)
        elif record_only:
            anim.save(os.path.join(full_trained_directory, 'Episode_{}.gif'.format(episode)))

        while not all_done:
            plt.pause(1)
        plt.close(fig)

    ray.shutdown()

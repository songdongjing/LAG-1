import os
import sys
import wandb
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.buffer import ReplayBuffer
import logging

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.render_mode = config['render_mode']

        try:
            self.writer = config['writer']
        except:
            self.writer = None
        
        # Tacview render obj
        self.tacview = None
        if self.render_mode == "real_time":
            from runner.tacview import Tacview
            self.tacview = Tacview()

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.buffer_size = self.all_args.buffer_size
        self.use_wandb = self.all_args.use_wandb

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.save_dir = str(self.run_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.load()

    def load(self):
        # algorithm
        if self.algorithm_name == "ppo":
            from ..algorithms.ppo.ppo_trainer_backup import PPOTrainer as Trainer
            from ..algorithms.ppo.ppo_policy_backup import PPOPolicy as Policy
        elif self.algorithm_name == "dsac":
            from ..algorithms.dsac.dsac_trainer import DSACTrainer as Trainer
            from ..algorithms.dsac.dsac_policy import DSACPolicy as Policy
            from ..algorithms.dsac.dsac_policy import DSACCritic as Critic
        else:
            raise NotImplementedError
        
        if self.algorithm_name == "dsac":
            self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
            self.critic1 = Critic(self.all_args, self.envs.observation_space, self.device)
            self.critic2 = Critic(self.all_args, self.envs.observation_space, self.device)
            self.target_critic1 = Critic(self.all_args, self.envs.observation_space, self.device)
            self.target_critic2 = Critic(self.all_args, self.envs.observation_space, self.device)
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
            self.trainer = Trainer(self.all_args, device=self.device)
        elif self.algorithm_name == "ppo":
            self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
            self.trainer = Trainer(self.all_args, device=self.device)
        else:
            raise NotImplementedError

        # buffer
        self.buffer = ReplayBuffer(self.all_args,
                                   self.envs.observation_space,
                                   self.envs.action_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def rollout(self):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        policy_actor = self.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        policy_critic = self.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_info(self, infos, total_num_steps):
        if self.use_wandb:
            for k, v in infos.items():
                wandb.log({k: v}, step=total_num_steps)
        else:
            for k, v in infos.items():#use tensorboard
                self.writer.add_scalar(k, v, total_num_steps)

    def render_with_tacview(self, data):
        """
        Send data to Tacview for real-time rendering.
        :param data: The data to be rendered, which can be a string or a specific structure.
        """
        if self.tacview:
            try:
                self.tacview.send_data_to_client(data)
            except Exception as e:
                logging.error(f"Tacview rendering error: {e}")

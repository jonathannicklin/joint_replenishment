import torch
import os
import json
import sys
import tianshou as ts

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from torch.optim.lr_scheduler import ExponentialLR

from dp import dynaplex
from dp.utils.tianshou.network_wrapper import TianshouModuleWrapper
from dp.gym.base_env import BaseEnv
from scripts.networks.joint_replenishment_actor_critic import CriticMLP, ActorMLP

load_mdp_from_file = True

if load_mdp_from_file:
    folder_name = "joint_replenishment"  # the name of the folder where the json file is located
    mdp_version_number = 0
    path_to_json = dynaplex.filepath("mdp_config_examples", folder_name, f"mdp_config_{mdp_version_number}.json")

    try:
        with open(path_to_json, "r") as input_file:
            vars = json.load(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path_to_json} not found. Please make sure the file exists and try again.")
    except:
        raise Exception("Something went wrong when loading the json file. Have you checked the json file does not contain any comment?")

mdp = dynaplex.get_mdp(**vars)

# Training parameters
train_args = {"hidden_dim": 512,
              "n_layers": 3,
              "lr": 1e-3,
              "entropy_coef": 0.1,
              "discount_factor": 0.99,
              "batch_size": 1040,
              "max_batch_size": 0,
              "nr_train_envs": 48,  # Increased from 24 to 48
              "nr_test_envs": 48,   # Increased from 24 to 48
              "max_epoch": 500,
              "step_per_collect": 1040,
              "step_per_epoch": 1040,
              "repeat_per_collect": 4,
              "replay_buffer_size": 1040,
              "max_batchsize": 1040,
              "num_actions_until_done": 0,
              "num_steps_per_test_episode": 1040
              }

def policy_path():
    path = os.path.normpath(dynaplex.filepath(mdp.identifier(), "ppo_policy"))
    return path

def save_best_fn(policy):
    save_path = policy_path()
    dynaplex.save_policy(policy.actor.wrapped_module,
                         {'input_type': 'dict', 'num_inputs': mdp.num_flat_features(), 'num_outputs': mdp.num_valid_actions()},
                         save_path)

def get_env():
    return BaseEnv(mdp, train_args["num_actions_until_done"])

def get_test_env():    
    return BaseEnv(mdp, train_args["num_steps_per_test_episode"])

def preprocess_function(**kwargs):
    if "obs" in kwargs:
        obs_with_tensors = [
            {"obs": torch.from_numpy(obs['obs']).to(torch.float).to(device=device),
             "mask": torch.from_numpy(obs['mask']).to(torch.bool).to(device=device)}
            for obs in kwargs["obs"]]
        kwargs["obs"] = obs_with_tensors
    if "obs_next" in kwargs:
        obs_with_tensors = [
            {"obs": torch.from_numpy(obs['obs']).to(torch.float).to(device=device),
             "mask": torch.from_numpy(obs['mask']).to(torch.bool).to(device=device)}
            for obs in kwargs["obs_next"]]
        kwargs["obs_next"] = obs_with_tensors
    return kwargs

if __name__ == '__main__':

    train = True
    if train:
        model_name = "ppo_model_dict.pt"     
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # define actor network structure
        actor_net = ActorMLP(
            input_dim=mdp.num_flat_features(),
            hidden_dim=train_args["hidden_dim"],
            n_layers=train_args["n_layers"],
            output_dim=mdp.num_valid_actions(),
            min_val=torch.finfo(torch.float).min
        ).to(device)

        # define critic network structure
        critic_net = CriticMLP(
            input_dim=mdp.num_flat_features(),
            hidden_dim=train_args["hidden_dim"],
            n_layers=train_args["n_layers"],
            min_val=torch.finfo(torch.float).min
        ).to(device).share_memory()

        # define optimizer
        optim = torch.optim.Adam(
            params=list(actor_net.parameters()) + list(critic_net.parameters()),
            lr=train_args["lr"]
        )

        # define scheduler
        scheduler = ExponentialLR(optim, 0.99)

        # Entropy coefficient
        entropy_coef = train_args["entropy_coef"]

        # define PPO policy with entropy regularization
        class PPOWithEntropy(ts.policy.PPOPolicy):
            def __init__(self, *args, entropy_coef=entropy_coef, **kwargs):
                super().__init__(*args, **kwargs)
                self.entropy_coef = entropy_coef

            def get_loss(self, data: ts.data.Batch, dist, value, return_critic=False):
                # Get the base PPO loss
                loss = super().get_loss(data, dist, value, return_critic)

                # Add entropy term
                entropy = dist.entropy().mean()
                loss -= self.entropy_coef * entropy  # Minimize negative entropy to encourage exploration

                # Value function clipping: Clip the value predictions to the range [value - epsilon, value + epsilon]
                value_clip_range = 0.2  # Clip the value predictions with epsilon
                value_clipped = torch.clamp(value, -value_clip_range, value_clip_range)
                loss += torch.mean((value - value_clipped) ** 2)  # Add the value clipping term to the loss

                return loss

        # Define PPO policy with added entropy
        policy = PPOWithEntropy(
            TianshouModuleWrapper(actor_net), critic_net, optim,
            discount_factor=train_args["discount_factor"],
            max_batchsize=train_args["max_batchsize"],
            dist_fn=torch.distributions.categorical.Categorical,
            deterministic_eval=True,
            lr_scheduler=scheduler,
            reward_normalization=False,
            entropy_coef=train_args["entropy_coef"],
            gae_lambda=0.95  # Set GAE lambda value for advantage estimation
        )
        policy.action_type = "discrete"

        # TensorBoard logging
        log_path = dynaplex.filepath(mdp.identifier(), "tensorboard_logs", model_name)
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        # Create nr_envs train environments
        train_envs = ts.env.DummyVectorEnv(
            [get_env for _ in range(train_args["nr_train_envs"])]
        )
        collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(train_args["replay_buffer_size"], train_args["nr_train_envs"]), exploration_noise=True, preprocess_fn=preprocess_function)
        collector.reset()

        # Create nr_envs test environments
        test_envs = ts.env.DummyVectorEnv(
            [get_test_env for _ in range(train_args["nr_test_envs"])]
        )
        test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False, preprocess_fn=preprocess_function)
        test_collector.reset()

        # Train the policy
        print("Starting training")
        policy.train()
        trainer = ts.trainer.OnpolicyTrainer(
            policy, collector, test_collector=test_collector,
            max_epoch=train_args["max_epoch"],
            step_per_epoch=train_args["step_per_epoch"],
            step_per_collect=train_args["step_per_collect"],
            episode_per_test=10, batch_size=train_args["batch_size"],
            repeat_per_collect=train_args["repeat_per_collect"],
            logger=logger, test_in_train=True,
            save_best_fn=save_best_fn)
        print(f'save location:{policy_path()}')
        result = trainer.run()
        print(f'Finished training!')

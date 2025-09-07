import argparse
import os
from typing import Callable, Tuple, Union

import gymnasium as gym
import torch
import torch.nn as nn
import wandb
from gymnasium import spaces
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from wandb.integration.sb3 import WandbCallback

# Our
import sys
import os
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from algorithms.wrappers import StableBaselineWrapperNum
from environment import load_LLM
from environment.movies.configs import get_base_parser, get_enviroment_from_args


# Define arguments
def parse_args():
    parser = get_base_parser()
    parser.add_argument("--model-device", type=str, default="cpu")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--path-ckpt", type=str, default=None)
    args = parser.parse_args()
    return args


# Define model
class Net(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.latent_dim_pi = embedding_dim * 2
        self.latent_dim_vf = embedding_dim * 2

        ## Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        
        # 初始化权重以确保数值稳定性
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.user_bias.weight)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(
                self.user_embedding.embedding_dim,
                num_items,
            )
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(
                self.user_embedding.embedding_dim + num_items, self.latent_dim_vf * 2
            ),
            nn.ReLU(),
            nn.Linear(self.latent_dim_vf * 2, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        user_id = features["user_id"].squeeze(1)
        film_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, film_seen], dim=1)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        
        # 确保logits的数值稳定性
        logits = torch.clamp(logits, min=-1e8, max=1e8)
        
        # 应用mask，但确保至少有一个动作可用
        logits[mask] = -1e8
        
        # 检查是否所有动作都被mask，如果是则取消一个mask
        all_masked = mask.all(dim=1, keepdim=True)
        if all_masked.any():
            # 对于完全被mask的样本，取消第一个动作的mask
            logits[all_masked.squeeze(), 0] = 0.0
        
        return logits, self.value_net(user_embedding_value)

    def forward_actor(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        user_embedding = self.user_embedding(user_id)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        
        # 确保logits的数值稳定性
        logits = torch.clamp(logits, min=-1e8, max=1e8)
        
        # 应用mask，但确保至少有一个动作可用
        logits[mask] = -1e8
        
        # 检查是否所有动作都被mask，如果是则取消一个mask
        all_masked = mask.all(dim=1, keepdim=True)
        if all_masked.any():
            # 对于完全被mask的样本，取消第一个动作的mask
            logits[all_masked.squeeze(), 0] = 0.0
        
        return logits

    def forward_critic(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        film_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, film_seen], dim=1)
        return self.value_net(user_embedding_value)


class DistributionUseLogitsDirectly(CategoricalDistribution):
    def __init__(self, action_dim: int):
        super().__init__(action_dim)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity(latent_dim)
    
    def proba_distribution(self, action_logits: torch.Tensor) -> "DistributionUseLogitsDirectly":
        # 确保logits中没有NaN或inf值
        action_logits = torch.clamp(action_logits, min=-1e8, max=1e8)
        # 检查并替换任何NaN值
        action_logits = torch.where(torch.isnan(action_logits), torch.zeros_like(action_logits), action_logits)
        return super().proba_distribution(action_logits)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        self.action_dist = DistributionUseLogitsDirectly(action_space.n)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Net(
            self.observation_space,
            train_env.num_users,
            train_env.num_items,
            embedding_dim=args.embedding_dim,
        )


class ExtractPass(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations["user_id"] = observations["user_id"].int()
        return observations


class TrainingRewardPrinter(BaseCallback):
    """在每步累积算法接收的 reward，并在每个 episode 结束时打印该 episode 的总回报。"""

    def __init__(self, prefix: str = "train", verbose: int = 0, save_to_file: bool = True):
        super().__init__(verbose)
        self.prefix = prefix
        self._episode_returns = None
        self._episode_lengths = None
        self._global_step = 0
        self.save_to_file = save_to_file
        self._csv_file = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        n_envs = self.model.n_envs if hasattr(self.model, "n_envs") else 1
        self._episode_returns = [0.0 for _ in range(n_envs)]
        self._episode_lengths = [0 for _ in range(n_envs)]
        
        # 初始化 CSV 文件
        if self.save_to_file:
            import csv
            import os
            os.makedirs("./tmp/rewards", exist_ok=True)
            self._csv_file = open(f"./tmp/rewards/{self.prefix}_rewards.csv", "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["global_step", "env_id", "reward", "episode_return", "episode_length"])

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()

    def _on_step(self) -> bool:
        # SB3 在 _on_step 提供 locals: rewards, dones
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        if rewards is None or dones is None:
            return True

        # 兼容单环境与向量化环境
        if not hasattr(rewards, "__len__"):
            rewards = [float(rewards)]
            dones = [bool(dones)]

        for i, r in enumerate(rewards):
            self._global_step += 1
            self._episode_returns[i] += float(r)
            self._episode_lengths[i] += 1
            
            # 写入 CSV
            if self._csv_writer:
                self._csv_writer.writerow([
                    self._global_step,
                    i,
                    float(r),
                    self._episode_returns[i],
                    self._episode_lengths[i]
                ])
                self._csv_file.flush()  # 确保实时写入
            
            if bool(dones[i]):
                print(f"[{self.prefix}] episode_return={self._episode_returns[i]}")
                self._episode_returns[i] = 0.0
                self._episode_lengths[i] = 0
        return True


if __name__ == "__main__":
    args = parse_args()
    llm = load_LLM(args.llm_model)

    if args.path_ckpt is not None:
        args.seed += 1200000

    train_env = get_enviroment_from_args(llm, args)

    test_env = get_enviroment_from_args(
        llm,
        args,
        seed=args.seed + 600,
    )

    # Create the custom actor-critic policy
    policy_kwargs = dict(
        features_extractor_class=ExtractPass,
    )

    train_env = StableBaselineWrapperNum(train_env)
    test_env = Monitor(StableBaselineWrapperNum(test_env))
    # 跳过环境检查以避免卡住
    # check_env(train_env)
    # check_env(test_env)

    # Initialize wandb
    run = wandb.init(
        project="MPR",
        config=args,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,
        mode="disabled",  # 禁用wandb日志记录
        dir="./tmp/wandb",
    )

    model = TRPO(
        policy=CustomActorCriticPolicy,
        env=train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=args.model_device,
        tensorboard_log=f"./tmp/runs/{run.id}",
        gamma=args.gamma,
        # 优化参数以适应CPU和内存限制
        n_steps=2048,  # 减少步数以降低内存使用
        batch_size=64,  # 减小批次大小
        learning_rate=3e-4,  # 使用较小的学习率
    )

    wandb_callback = WandbCallback(
        model_save_path=f"./tmp/models/{run.id}",
        verbose=2,
        gradient_save_freq=100,
    )

    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=f"./tmp/models/{run.id}",
        log_path=f"./tmp/models/{run.id}",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 * 20,
        save_path=f"./tmp/models/{run.id}",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 添加奖励记录回调
    reward_callback = TrainingRewardPrinter(prefix="trpo_train", verbose=1, save_to_file=True)

    callback = CallbackList([wandb_callback, eval_callback, checkpoint_callback, reward_callback])

    print(model.policy)
    print(args)

    if args.path_ckpt is not None:
        model.policy.load_state_dict(torch.load(f"{args.path_ckpt}/policy.pth"))
        model.policy.to(args.model_device)
        model.policy.optimizer.load_state_dict(
            torch.load(f"{args.path_ckpt}/policy.optimizer.pth")
        )

        model.learn(total_timesteps=400000, progress_bar=False, callback=callback)
    else:
        model.learn(total_timesteps=5000, progress_bar=True, callback=callback)

    run.finish()

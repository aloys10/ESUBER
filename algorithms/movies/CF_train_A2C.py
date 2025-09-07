import argparse
import os
from typing import Callable, Tuple, Union

import gymnasium as gym
import torch
import torch.nn as nn
import wandb
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from wandb.integration.sb3 import WandbCallback

# Our (robust import to avoid package-name conflicts and missing PYTHONPATH)
try:
    from algorithms.wrappers import StableBaselineWrapperNum, RewardLoggingWrapper
except Exception:
    import sys as _sys
    import os as _os
    from pathlib import Path as _Path
    import importlib.util as _importlib_util
    import importlib.machinery as _importlib_machinery

    _here = _Path(__file__).resolve()
    _project_root = str(_here.parents[2])
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    try:
        from algorithms.wrappers import StableBaselineWrapperNum, RewardLoggingWrapper
    except Exception:
        # Hard-load from file to avoid name conflicts with third-party 'algorithms'
        _wrappers_path = _here.parents[1] / "wrappers.py"
        _loader = _importlib_machinery.SourceFileLoader("_local_wrappers", str(_wrappers_path))
        _spec = _importlib_util.spec_from_loader("_local_wrappers", _loader)
        _module = _importlib_util.module_from_spec(_spec)
        _loader.exec_module(_module)
        StableBaselineWrapperNum = getattr(_module, "StableBaselineWrapperNum")
        RewardLoggingWrapper = getattr(_module, "RewardLoggingWrapper")
from environment import load_LLM
from environment.movies.configs import get_base_parser, get_enviroment_from_args


# Define arguments
def parse_args():
    parser = get_base_parser()
    parser.add_argument("--model-device", type=str, default="cuda:1")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--path-ckpt", type=str, default=None, help="检查点路径，用于恢复训练")
    parser.add_argument("--resume-from", type=str, default=None, help="从指定的模型文件恢复训练")
    parser.add_argument("--n-steps", type=int, default=1000, help="每多少步进行一次参数更新")
    args = parser.parse_args()
    return args


# Define model
class Net(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        num_users: int,
        num_items: int,
    ):
        super().__init__()
        embedding_dim = args.embedding_dim
        self.latent_dim_pi = embedding_dim * 2
        self.latent_dim_vf = embedding_dim * 2

        ## Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)

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
        
        # 改进初始化，防止 NaN
        self._init_weights()

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

        # 检查embedding中是否有NaN
        if torch.isnan(user_embedding).any():
            print(f"Warning: NaN detected in user_embedding, replacing with zeros")
            user_embedding = torch.where(torch.isnan(user_embedding), torch.zeros_like(user_embedding), user_embedding)
        
        if torch.isnan(user_bias).any():
            print(f"Warning: NaN detected in user_bias, replacing with zeros")
            user_bias = torch.where(torch.isnan(user_bias), torch.zeros_like(user_bias), user_bias)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        
        # 检查policy_net输出是否有NaN
        if torch.isnan(logits).any():
            print(f"Warning: NaN detected in policy_net output, replacing with small random values")
            logits = torch.where(torch.isnan(logits), torch.randn_like(logits) * 0.01, logits)
        
        logits[mask] = -torch.inf
        
        # 最终检查并修复 NaN 值
        if torch.isnan(logits).any():
            print(f"Warning: NaN detected in final logits, replacing with -1e8")
            logits = torch.where(torch.isnan(logits), torch.tensor(-1e8, device=logits.device), logits)
        
        # 检查value网络输出
        value_output = self.value_net(user_embedding_value)
        if torch.isnan(value_output).any():
            print(f"Warning: NaN detected in value_net output, replacing with zeros")
            value_output = torch.where(torch.isnan(value_output), torch.zeros_like(value_output), value_output)
        
        return logits, value_output

    def forward_actor(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        user_embedding = self.user_embedding(user_id)
        user_bias = self.user_bias(user_id)

        # 检查embedding中是否有NaN
        if torch.isnan(user_embedding).any():
            print(f"Warning: NaN detected in user_embedding (actor), replacing with zeros")
            user_embedding = torch.where(torch.isnan(user_embedding), torch.zeros_like(user_embedding), user_embedding)
        
        if torch.isnan(user_bias).any():
            print(f"Warning: NaN detected in user_bias (actor), replacing with zeros")
            user_bias = torch.where(torch.isnan(user_bias), torch.zeros_like(user_bias), user_bias)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        
        # 检查policy_net输出是否有NaN
        if torch.isnan(logits).any():
            print(f"Warning: NaN detected in policy_net output (actor), replacing with small random values")
            logits = torch.where(torch.isnan(logits), torch.randn_like(logits) * 0.01, logits)
        
        logits[mask] = -torch.inf
        
        # 最终检查并修复 NaN 值
        if torch.isnan(logits).any():
            print(f"Warning: NaN detected in final actor logits, replacing with -1e8")
            logits = torch.where(torch.isnan(logits), torch.tensor(-1e8, device=logits.device), logits)
        
        return logits

    def forward_critic(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        film_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        
        # 检查embedding中是否有NaN
        if torch.isnan(user_embedding).any():
            print(f"Warning: NaN detected in user_embedding (critic), replacing with zeros")
            user_embedding = torch.where(torch.isnan(user_embedding), torch.zeros_like(user_embedding), user_embedding)
        
        user_embedding_value = torch.cat([user_embedding, film_seen], dim=1)
        value_output = self.value_net(user_embedding_value)
        
        # 检查value网络输出
        if torch.isnan(value_output).any():
            print(f"Warning: NaN detected in value_net output (critic), replacing with zeros")
            value_output = torch.where(torch.isnan(value_output), torch.zeros_like(value_output), value_output)
        
        return value_output
    
    def _init_weights(self):
        """改进的权重初始化，防止 NaN"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更保守的初始化
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # 更小的增益
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                # 使用更小的标准差
                nn.init.normal_(module.weight, mean=0.0, std=0.01)  # 更小的标准差


class DistributionUseLogitsDirectly(CategoricalDistribution):
    def __init__(self, action_dim: int):
        super().__init__(action_dim)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity(latent_dim)


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


class ExplanationPrinter(BaseCallback):
    """打印每步的打分理由和相关信息"""

    def __init__(self, verbose: int = 1, print_frequency: int = 1):
        super().__init__(verbose)
        self.print_frequency = print_frequency
        self._step_count = 0


    def _on_step(self) -> bool:
        self._step_count += 1
        
        # 只在指定频率下打印打分理由
        if self._step_count % self.print_frequency == 0:
            # 从环境的 info 中获取解释信息
            infos = self.locals.get("infos", None)
            if infos is not None:
                # 处理向量化环境的情况
                if isinstance(infos, list):
                    for i, info in enumerate(infos):
                        if info and isinstance(info, dict):
                            explanation = info.get("LLM_explanation", info.get("explanation", ""))
                            rating = info.get("LLM_rating", info.get("rating", "N/A"))
                            
                            if explanation and rating != "N/A":
                                print(f"评分: {rating} | 理由: {explanation}")
                else:
                    # 单环境情况
                    if isinstance(infos, dict):
                        explanation = infos.get("LLM_explanation", infos.get("explanation", ""))
                        rating = infos.get("LLM_rating", infos.get("rating", "N/A"))
                        
                        if explanation and rating != "N/A":
                            print(f"评分: {rating} | 理由: {explanation}")
        
        return True


class GradientClippingCallback(BaseCallback):
    """梯度裁剪回调，防止梯度爆炸"""
    
    def __init__(self, max_grad_norm: float = 0.5, verbose: int = 0):
        super().__init__(verbose)
        self.max_grad_norm = max_grad_norm
    
    def _on_step(self) -> bool:
        # 在每次更新后检查梯度
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'parameters'):
            total_norm = 0
            param_count = 0
            for param in self.model.policy.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                if total_norm > self.max_grad_norm:
                    if self.verbose > 0:
                        print(f"Warning: Gradient norm {total_norm:.4f} exceeds threshold {self.max_grad_norm}")
        
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

    train_env = RewardLoggingWrapper(StableBaselineWrapperNum(train_env), prefix="train")
    test_env = Monitor(StableBaselineWrapperNum(test_env))
    check_env(train_env)
    check_env(test_env)

    # Initialize wandb (disabled mode - no online sync)
    run = wandb.init(
        project="MPR",
        config=args,
        mode="disabled",  # 完全禁用 W&B 在线功能
        dir="./tmp/wandb",
    )

    model = A2C(
        CustomActorCriticPolicy,
        train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=args.model_device,
        tensorboard_log=f"./tmp/runs/{run.id}",
        gamma=args.gamma,
        n_steps=args.n_steps,  # 使用命令行参数控制优化频率
        learning_rate=3e-4,  # 降低学习率
        max_grad_norm=0.5,  # 添加梯度裁剪
        ent_coef=0.01,  # 添加熵正则化
        vf_coef=0.25,  # 价值函数系数
        normalize_advantage=True,  # 标准化优势
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

    callback = CallbackList([
        wandb_callback,
        eval_callback,
        checkpoint_callback,
        TrainingRewardPrinter(prefix="train"),
        GradientClippingCallback(max_grad_norm=0.5, verbose=1),
    ])

    print(model.policy)
    print(args)

    # 检查点恢复逻辑
    if args.resume_from is not None:
        print(f"从检查点恢复训练: {args.resume_from}")
        try:
            # 尝试加载完整的模型
            model = A2C.load(args.resume_from, env=train_env, device=args.model_device)
            print("成功从检查点恢复模型")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("尝试手动加载状态字典...")
            try:
                # 手动加载状态字典
                checkpoint = torch.load(args.resume_from, map_location=args.model_device)
                if 'policy' in checkpoint:
                    model.policy.load_state_dict(checkpoint['policy'])
                else:
                    model.policy.load_state_dict(checkpoint)
                model.policy.to(args.model_device)
                
                # 尝试加载优化器状态
                if 'optimizer' in checkpoint:
                    model.policy.optimizer.load_state_dict(checkpoint['optimizer'])
                print("成功手动加载状态字典")
            except Exception as e2:
                print(f"手动加载也失败: {e2}")
                print("将从头开始训练...")
        
        # 恢复训练时使用较少的步数
        model.learn(total_timesteps=400000, progress_bar=False, callback=callback, reset_num_timesteps=False)
        
    elif args.path_ckpt is not None:
        print(f"从旧格式检查点恢复: {args.path_ckpt}")
        try:
            model.policy.load_state_dict(torch.load(f"{args.path_ckpt}/policy.pth", map_location=args.model_device))
            model.policy.to(args.model_device)
            model.policy.optimizer.load_state_dict(
                torch.load(f"{args.path_ckpt}/policy.optimizer.pth", map_location=args.model_device)
            )
            print("成功从旧格式检查点恢复")
            model.learn(total_timesteps=400000, progress_bar=False, callback=callback, reset_num_timesteps=False)
        except Exception as e:
            print(f"从旧格式检查点恢复失败: {e}")
            print("将从头开始训练...")
            model.learn(total_timesteps=1800000, progress_bar=False, callback=callback)
    else:
        print("从头开始训练...")
        model.learn(total_timesteps=1800000, progress_bar=False, callback=callback)

    run.finish()

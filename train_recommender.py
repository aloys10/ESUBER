import argparse
import os
from typing import Optional

import numpy as np
import gymnasium as gym


class SB3ObsWrapper(gym.ObservationWrapper):
    """
    将环境原始 Dict 观测转为固定长度向量：
    [gender, age_norm, (a1_norm, r1_norm), ..., (aK_norm, rK_norm)]
    其中 K=history_len，交互不足则0填充。
    """

    def __init__(self, env: gym.Env, history_len: int = 10):
        super().__init__(env)
        self.history_len = history_len
        self._num_items = int(self.env.action_space.n)
        # 评分上下界
        self._r_min = float(getattr(self.env.reward_perturbator, "min_rating", 0.0))
        self._r_max = float(getattr(self.env.reward_perturbator, "max_rating", 10.0))
        # obs 维度：2 + 2*K
        low = np.array([0.0, 0.0] + [0.0, 0.0] * self.history_len, dtype=np.float32)
        high = np.array([1.0, 1.0] + [1.0, 1.0] * self.history_len, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):  # type: ignore[override]
        gender = float(observation.get("user_gender", 0))  # 0/1
        age_arr = observation.get("user_age")
        age = float(age_arr[0]) if age_arr is not None and len(age_arr) > 0 else 0.0
        age_norm = np.clip(age / 200.0, 0.0, 1.0)

        items_interact = observation.get("items_interact", tuple())
        # items_interact: tuple of np.array([action, rating])
        hist_pairs = []
        for i in range(min(self.history_len, len(items_interact))):
            a = float(items_interact[-(i + 1)][0])  # 最近优先
            r = float(items_interact[-(i + 1)][1])
            a_norm = np.clip(a / max(1, self._num_items - 1), 0.0, 1.0)
            r_norm = 0.0
            if self._r_max > self._r_min:
                r_norm = np.clip((r - self._r_min) / (self._r_max - self._r_min), 0.0, 1.0)
            hist_pairs.extend([a_norm, r_norm])
        # 填充到固定长度
        while len(hist_pairs) < 2 * self.history_len:
            hist_pairs.extend([0.0, 0.0])

        vec = np.array([gender, age_norm] + hist_pairs, dtype=np.float32)
        return vec


def build_env(domain: str, seed: int, offline_random_rater: bool):
    """
    Assemble environment via environment/*/configs.py factories.
    Optionally disable real LLM calls by enabling random ratings.
    """
    if domain == "movies":
        from environment import LLM
        from environment.movies.configs import (
            get_base_parser as get_movies_parser,
            get_enviroment_from_args as movies_make_env,
        )

        llm = LLM.load_LLM("deepseek-chat")
        parser = get_movies_parser()
        # Provide defaults; allow user CLI to override
        parser.set_defaults(
            llm_model="deepseek-chat",
            llm_rater="0Shot_cotlite_our",
            items_retrieval="decay_emotion_3",
            consider_arousal=True,
            reward_shaping="churn_satisfaction",
            churn_ema_alpha=0.1,
            churn_low_threshold=2.0,
            seed=seed,
        )
        args = parser.parse_args([])
        env = movies_make_env(llm, args, seed=seed)
    elif domain == "books":
        from environment import LLM
        from environment.books.configs import (
            get_base_parser as get_books_parser,
            get_enviroment_from_args as books_make_env,
        )

        llm = LLM.load_LLM("deepseek-chat")
        parser = get_books_parser()
        parser.set_defaults(
            llm_model="deepseek-chat",
            llm_rater="0Shot_cotlite_our",
            items_retrieval="decay_emotion_3",
            consider_arousal=True,
            reward_shaping="churn_satisfaction",
            churn_ema_alpha=0.3,
            churn_low_threshold=2.0,
            book_dataset="books_amazon/postprocessed_books",
            user_dataset="detailed",
            perturbator="none",
            seed=seed,
        )
        args = parser.parse_args([])
        env = books_make_env(llm, args, seed=seed)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Prefer not to generate verbose explanations during training
    try:
        env.rating_prompt.llm_query_explanation = False
    except Exception:
        pass

    # Optional: use random synthetic ratings to avoid LLM cost for smoke tests
    if offline_random_rater:
        try:
            env.rating_prompt.random_rating = True
        except Exception:
            pass

    # 包装为SB3兼容观测
    env = SB3ObsWrapper(env, history_len=10)
    return env


def train(domain: str, algo: str, total_timesteps: int, seed: int, save_path: Optional[str]):
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv

    offline_random_rater = os.environ.get("OFFLINE_RANDOM_RATER", "0") == "1"

    def make_env():
        return build_env(domain, seed, offline_random_rater)

    vec_env = DummyVecEnv([make_env])

    if algo.lower() == "ppo":
        model = PPO("MlpPolicy", vec_env, seed=seed, verbose=1)
    elif algo.lower() == "a2c":
        model = A2C("MlpPolicy", vec_env, seed=seed, verbose=1)
    else:
        raise ValueError("algo must be one of: ppo, a2c")

    model.learn(total_timesteps=total_timesteps)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)

    return model, vec_env


def evaluate(model, vec_env, episodes: int = 3):
    """Simple evaluation loop returning average episodic reward."""
    env = vec_env.envs[0]
    returns = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_r += float(reward)
            done = bool(terminated or truncated)
        returns.append(total_r)
    return float(np.mean(returns)), float(np.std(returns) if len(returns) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["movies", "books"], default="books")
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--total-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--offline-random-rater", action="store_true", help="Use synthetic random ratings instead of LLM calls")

    args = parser.parse_args()

    if args.offline_random_rater:
        os.environ["OFFLINE_RANDOM_RATER"] = "1"

    model, vec_env = train(
        domain=args.domain,
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        save_path=args.save_path,
    )

    avg_r, std_r = evaluate(model, vec_env, episodes=args.eval_episodes)
    print(f"Evaluation: avg_reward={avg_r:.3f}, std={std_r:.3f}, episodes={args.eval_episodes}")


if __name__ == "__main__":
    main()



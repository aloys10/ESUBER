import argparse
import os

from environment import LLM
from environment.movies.movies_loader import MoviesLoader
from ..env import Simulatio4RecSys
from ..users import UsersCSVLoader
from ..items_retrieval import (
    SentenceSimilarityItemsRetrieval,
    SimpleMoviesRetrieval,
    TimeItemsRetrieval,
    DecayEmotionWeightedRetrieval,
)
from ..items_selection import GreedySelector
from ..reward_perturbator import GaussianPerturbator, GreedyPerturbator, NoPerturbator
from .rater_prompts.our_system_prompt import (
    ThirdPersonDescriptive09_2Shot_OurSys,
    ThirdPersonDescriptive09_OurSys,
    ThirdPersonDescriptive09_1Shot_OurSys,
    ThirdPersonDescriptive110_2Shot_OurSys,
    ThirdPersonDescriptive110_1Shot_OurSys,
    ThirdPersonDescriptiveOneTen_2Shot_OurSys,
    ThirdPersonDescriptiveOneTen_1Shot_OurSys,
    ThirdPersonDescriptive09_2Shot_OurSys,
    ThirdPersonDescriptive110_OurSys,
    ThirdPersonDescriptiveOneTen_OurSys,
)
from .rater_prompts.our_system_prompt.third_person_descriptive_cotlite_0_9 import (
    ThirdPersonDescriptive09_CoTLite_OurSys,
)
from .rater_prompts import (
    ThirdPersonDescriptive09,
    ThirdPersonDescriptive09_1Shot,
    ThirdPersonDescriptive09_2Shot,
)
from environment.reward_shaping import (
    RewardReshapingExpDecayTime,
    RewardReshapingRandomWatch,
    IdentityRewardShaping,
    RewardReshapingTerminateIfSeen,
    RewardReshapingChurnSatisfaction,
)

from gymnasium.utils.env_checker import check_env


# Single module loading utils
OPTIONS_LLM_RATER = [
    "2Shot_system_our",
    "1Shot_system_our",
    "0Shot_system_our",
    "0Shot_cotlite_our",
    "2Shot_system_default",
    "1Shot_system_default",
    "0Shot_system_default",
    "0Shot_system_our_1_10",
    "1Shot_system_our_1_10",
    "2Shot_system_our_1_10",
    "2Shot_system_our_one_ten",
    "1Shot_system_our_one_ten",
    "0Shot_system_our_one_ten",
    "2Shot_invert_system_default",
    "2Shot_invert_system_our",
    "1Shot_invert_system_our",
    "1Shot_invert_system_default",
    # 新增的逻辑链增强版本
    "2Shot_cot_enhanced",
    "1Shot_cot_enhanced",
    "0Shot_cot_enhanced",
]
OPTIONS_ITEMS_RETRIEVAL = [
    "last_3",
    "most_similar_3",
    "none",
    "simple_3",
    "decay_emotion_3",
]
OPTIONS_REWARD_PERTURBATOR = ["none", "gaussian", "greedy"]
OPTIONS_USER_DATASET = ["detailed", "basic", "sampled_genres"]
OPTIONS_REWARD_SHAPING = [
    "identity",
    "exp_decay_time",
    "random_watch",
    "same_film_terminate",
    "churn_satisfaction",
]


def get_llm_rater(name, llm, history=True):
    CURRENT_MOVIE_FEATURES_LIST = [
        "title",
        "overview",
        "genres",
        "actors",
        "vote_average",
    ]
    if name == "2Shot_system_our":
        return ThirdPersonDescriptive09_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_invert_system_our":
        return ThirdPersonDescriptive09_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_order=True,
        )
    elif name == "1Shot_system_our":
        return ThirdPersonDescriptive09_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_invert_system_our":
        return ThirdPersonDescriptive09_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_user=True,
        )
    elif name == "0Shot_system_our":
        return ThirdPersonDescriptive09_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "0Shot_cotlite_our":
        return ThirdPersonDescriptive09_CoTLite_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            llm_query_explanation=False,
        )
    # 新增的逻辑链增强版本
    elif name == "2Shot_cot_enhanced":
        from .rater_prompts.our_system_prompt.third_person_descriptive_cot_enhanced import ThirdPersonDescriptive09_2Shot_CotEnhanced_OurSys
        return ThirdPersonDescriptive09_2Shot_CotEnhanced_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_cot_enhanced":
        from .rater_prompts.our_system_prompt.third_person_descriptive_cot_enhanced import ThirdPersonDescriptive09_1Shot_CotEnhanced_OurSys
        return ThirdPersonDescriptive09_1Shot_CotEnhanced_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "0Shot_cot_enhanced":
        from .rater_prompts.our_system_prompt.third_person_descriptive_cot_enhanced import ThirdPersonDescriptive09_CotEnhanced_OurSys
        return ThirdPersonDescriptive09_CotEnhanced_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_system_default":
        return ThirdPersonDescriptive09_2Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_invert_system_default":
        return ThirdPersonDescriptive09_2Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_order=True,
        )
    elif name == "1Shot_system_default":
        return ThirdPersonDescriptive09_1Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_invert_system_default":
        return ThirdPersonDescriptive09_1Shot(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            switch_user=True,
        )
    elif name == "0Shot_system_default":
        return ThirdPersonDescriptive09(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_system_our_1_10":
        return ThirdPersonDescriptive110_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_system_our_1_10":
        return ThirdPersonDescriptive110_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "0Shot_system_our_1_10":
        return ThirdPersonDescriptive110_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "2Shot_system_our_one_ten":
        return ThirdPersonDescriptiveOneTen_2Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "1Shot_system_our_one_ten":
        return ThirdPersonDescriptiveOneTen_1Shot_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
        )
    elif name == "0Shot_system_our_one_ten":
        return ThirdPersonDescriptiveOneTen_OurSys(
            llm,
            current_items_features_list=CURRENT_MOVIE_FEATURES_LIST,
            previous_items_features_list=["title", "rating"] if history else [],
            llm_query_explanation=True,
        )
    else:
        raise ValueError(f"Unknown LLM rater {name}")


def get_items_retrieval(name, args=None):
    if name == "last_3":
        return TimeItemsRetrieval(3)
    elif name == "most_similar_3":
        return SentenceSimilarityItemsRetrieval(3, "overview_embedding")
    elif name == "simple_3":
        return SimpleMoviesRetrieval(3)
    elif name == "none":
        return TimeItemsRetrieval(0)
    elif name == "decay_emotion_3":
        lambda_time = getattr(args, "decay_time_lambda", 0.15) if args else 0.15
        emo_bonus = getattr(args, "emotion_bonus", 0.2) if args else 0.2
        consider_arousal = getattr(args, "consider_arousal", False) if args else False
        return DecayEmotionWeightedRetrieval(
            3,
            name_field_embedding="overview_embedding",
            lambda_time_decay=lambda_time,
            emotion_bonus=emo_bonus,
            consider_arousal=consider_arousal,
        )
    else:
        raise ValueError(f"Unknown item retrieval {name}")


def get_reward_perturbator(name, seed):
    if name == "none":
        return NoPerturbator(seed=seed, stepsize=1.0)
    elif name == "gaussian":
        return GaussianPerturbator(seed=seed, stepsize=1.0)
    elif name == "greedy":
        return GreedyPerturbator(seed=seed, stepsize=1.0)


def get_user_dataset(name):
    base_dir = os.path.join(
        os.path.dirname(__file__),
        "./users_generation/datasets/",
    )

    if name == "detailed":
        return UsersCSVLoader("user_features_hard_600", base_dir)
    elif name == "basic":
        return UsersCSVLoader("user_features_600", base_dir)
    elif name == "sampled_genres":
        return UsersCSVLoader("user_features_sampled_genres_600", base_dir)
    else:
        raise ValueError(f"Unknown user dataset {name}")


def get_reward_shaping(name, seed, args=None):
    if name == "identity":
        return IdentityRewardShaping()
    elif name == "exp_decay_time":
        return RewardReshapingExpDecayTime(q=0.1, seed=seed, stepsize=1.0)
    elif name == "random_watch":
        return RewardReshapingRandomWatch(q=0.1, seed=seed, stepsize=1.0)
    elif name == "same_film_terminate":
        return RewardReshapingTerminateIfSeen(
            q=0.1,
            seed=seed,
            stepsize=1.0,
        )
    elif name == "churn_satisfaction":
        return RewardReshapingChurnSatisfaction(
            ema_alpha=getattr(args, "churn_ema_alpha", 0.2) if args else 0.2,
            low_threshold=getattr(args, "churn_low_threshold", 4.0) if args else 4.0,
            low_streak_threshold=getattr(args, "churn_low_streak_threshold", 2) if args else 2,
            min_steps=getattr(args, "churn_min_steps", 5) if args else 5,
            prob_scale=getattr(args, "churn_prob_scale", 0.3) if args else 0.3,
            ema_min_samples=getattr(args, "churn_ema_min_samples", 1) if args else 1,
            recovery_bonus=getattr(args, "churn_recovery_bonus", 0.0) if args else 0.0,
            stepsize=1.0,
            min_rating=1.0,
            max_rating=10.0,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown reward shaping {name}")


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-model",
        type=str,
        default="TheBloke/Llama-2-7b-Chat-GPTQ",
        choices=LLM.SUPPORTED_MODELS,
    )

    parser.add_argument(
        "--llm-rater",
        type=str,
        default="0Shot_cotlite_our",
        choices=OPTIONS_LLM_RATER,
    )
    parser.add_argument(
        "--items-retrieval",
        type=str,
        default="last_3",
        choices=OPTIONS_ITEMS_RETRIEVAL,
    )

    parser.add_argument(
        "--user-dataset",
        type=str,
        default="sampled_genres",
        choices=OPTIONS_USER_DATASET,
    )
    parser.add_argument(
        "--film-dataset",
        type=str,
        default="movielens_latest-small",
        choices=["movielens_latest-small"],
    )
    parser.add_argument(
        "--perturbator",
        type=str,
        default="none",
        choices=OPTIONS_REWARD_PERTURBATOR,
    )
    parser.add_argument(
        "--reward-shaping",
        type=str,
        default="exp_decay_time",
        choices=OPTIONS_REWARD_SHAPING,
    )

    # Churn satisfaction parameters (1-10 scale defaults)
    parser.add_argument("--churn-ema-alpha", dest="churn_ema_alpha", type=float, default=0.2)
    parser.add_argument("--churn-low-threshold", dest="churn_low_threshold", type=float, default=4.0)
    parser.add_argument("--churn-low-streak-threshold", dest="churn_low_streak_threshold", type=int, default=2)
    parser.add_argument("--churn-min-steps", dest="churn_min_steps", type=int, default=5)
    parser.add_argument("--churn-prob-scale", dest="churn_prob_scale", type=float, default=0.3)
    parser.add_argument("--churn-ema-min-samples", dest="churn_ema_min_samples", type=int, default=1)
    parser.add_argument("--churn-recovery-bonus", dest="churn_recovery_bonus", type=float, default=0.0)

    # Dynamic memory retrieval hyper-parameters
    parser.add_argument("--decay-time-lambda", dest="decay_time_lambda", type=float, default=0.15)
    parser.add_argument("--emotion-bonus", dest="emotion_bonus", type=float, default=0.2)
    parser.add_argument("--consider-arousal", dest="consider_arousal", action="store_true")

    # Emotion thresholds for mapping rating -> (valence, arousal) on 1-10 scale
    parser.add_argument("--valence-positive-threshold", type=float, default=7.0)
    parser.add_argument("--valence-negative-threshold", type=float, default=4.0)
    parser.add_argument("--arousal-high-threshold", type=float, default=8.0)
    parser.add_argument("--arousal-low-threshold", type=float, default=3.0)

    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_enviroment_from_args(
    llm, args, seed=None, render_mode=None, render_path=None, eval_mode=False
):
    """Returns the environment with the configuration specified in args."""
    if seed is None:
        seed = args.seed
    env = Simulatio4RecSys(
        render_mode=render_mode,
        render_path=render_path,
        items_loader=MoviesLoader(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "./datasets/",
                args.film_dataset + ".json",
            )
        ),
        users_loader=get_user_dataset(args.user_dataset),
        items_selector=GreedySelector(seed),
        reward_perturbator=get_reward_perturbator(args.perturbator, seed),
        items_retrieval=get_items_retrieval(args.items_retrieval, args),
        llm_rater=get_llm_rater(
            args.llm_rater, llm, history=args.items_retrieval != "none"
        ),
        reward_shaping=get_reward_shaping(args.reward_shaping, seed, args),
        evaluation=eval_mode,
    )
    env.reset(seed=seed)
    # Apply memory emotion thresholds from args
    env.memory.valence_positive_threshold = args.valence_positive_threshold
    env.memory.valence_negative_threshold = args.valence_negative_threshold
    env.memory.arousal_high_threshold = args.arousal_high_threshold
    env.memory.arousal_low_threshold = args.arousal_low_threshold
    return env

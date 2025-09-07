import typing
from environment.movies.movie import Movie
from environment.memory import UserMovieInteraction
import numpy as np
from abc import ABC, abstractmethod
import math


class RewardShaping(ABC):
    """
    Object that is responsable to reshape the rewards

    Attributes:
        stepsize (float): stepsize describe how the ratings should be rounded after perturbation, for example is stepsize = 0.5 the possible
                          ratings will me {min_rating, min_rating + 0.5, min_rating + 1, ..., max_rating}
        min_rating (integer): smalles rating that can be assigned to a item
        max_rating (integer): largest rating that can be assigned to a item
        seed (integer): seed of the perturbator
    """

    def __init__(self, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.seed(seed)
        self.stepsize = stepsize
        self.min_rating = min_rating
        self.max_rating = max_rating

    @abstractmethod
    def reshape(
        self,
        item_interactions: typing.List[UserMovieInteraction],
        rating: int,
    ) -> typing.Tuple[int, bool]:
        """
        reshapes the reward to the item, the main application is to fix some behaviour of the LLM, which tends
        for example to give always the same rating to the same item

        Args:
            item_interactions (list of UserMovieInteraction): the list of interaction of the current user with the current item
            rating (int): rating (based on the LLM answer)

        Return
            the same list of items and the perturbated ratings
        """
        pass

    def seed(self, seed: int):
        """
        Function used to change the seed of the perturbator

        Args:
            seed (integer): new seed of the object
        """
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """
        Optional reset hook to clear per-episode state in subclasses.
        """
        pass

    def rating_fixing(self, number: float):
        """
        Function used to project a score to the set of feasible ratings, from min_rating to max_rating, spaced linearly every 0.5

        Args:
            number (float), number to project into the set of feasible numbers

        Return:
            projected number
        """
        number = (math.floor(number / self.stepsize)) * self.stepsize
        if number < self.min_rating:
            return self.min_rating
        elif number > self.max_rating:
            return self.max_rating
        else:
            return number


class IdentityRewardShaping(RewardShaping):
    def __init__(self, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        return rating, False


class RewardReshapingExpDecayTime(RewardShaping):
    def __init__(self, q=0.1, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.q = q
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        if len(item_interactions) == 1:
            return float(rating), False
        current_time = item_interactions[-1].timestamp
        last_time = item_interactions[-2].timestamp
        num_watches = item_interactions[-1].num_watches

        return (
            self.rating_fixing(
                float(rating)
                * math.pow(
                    self.q, float(num_watches) / (float(current_time - last_time))
                )
            ),
            False,
        )


class RewardReshapingRandomWatch(RewardShaping):
    def __init__(self, q=0.1, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.q = q
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        if len(item_interactions) == 1:
            return float(rating), False
        current_time = item_interactions[-1].timestamp
        last_time = item_interactions[-2].timestamp
        num_watches = item_interactions[-1].num_watches

        p = math.pow(self.q, float(num_watches) / (float(current_time - last_time)))

        watch = bool(self.rng.choice([True, False], p=[p, 1 - p]))
        if watch:
            return float(rating), False
        else:
            return 0.0, False


class RewardReshapingTerminateIfSeen(RewardShaping):
    def __init__(self, q=0.1, stepsize=0.5, min_rating=1, max_rating=10, seed=42):
        self.q = q
        super().__init__(stepsize, min_rating, max_rating, seed)

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        if len(item_interactions) == 1:
            return float(rating), False
        else:
            return 0.0, True


class RewardReshapingChurnSatisfaction(RewardShaping):
    """
    Dynamic termination based on recent satisfaction.

    - Maintains an EMA of ratings and a consecutive-low-ratings counter.
    - Terminates when:
        * low_streak >= low_streak_threshold (after min_steps), or
        * probabilistically when EMA is below low_threshold (probability grows as EMA drops).

    Parameters are scale-aware via min/max rating and can be tuned via CLI.
    """

    def __init__(
        self,
        ema_alpha: float = 0.2,
        low_threshold: float = 4.0,
        low_streak_threshold: int = 2,
        min_steps: int = 5,
        prob_scale: float = 0.3,
        ema_min_samples: int = 1,
        recovery_bonus: float = 0.0,
        stepsize=0.5,
        min_rating=1,
        max_rating=10,
        seed=42,
    ):
        super().__init__(stepsize, min_rating, max_rating, seed)
        self.ema_alpha = ema_alpha
        self.low_threshold = low_threshold
        self.low_streak_threshold = low_streak_threshold
        self.min_steps = min_steps
        self.prob_scale = prob_scale
        self.ema_min_samples = ema_min_samples
        self.recovery_bonus = recovery_bonus
        self.reset()

    def reset(self):
        self._ema = None
        self._count = 0
        self._low_streak = 0
        self._since_recovery = 0

    def _update_ema(self, rating: float):
        if self._ema is None:
            self._ema = float(rating)
        else:
            self._ema = self.ema_alpha * float(rating) + (1 - self.ema_alpha) * self._ema

    def reshape(
        self, item_interactions: typing.List[UserMovieInteraction], rating: int
    ) -> float:
        # Update counters
        self._count += 1
        self._update_ema(rating)
        if float(rating) <= self.low_threshold:
            self._low_streak += 1
            self._since_recovery = 0
        else:
            self._low_streak = 0
            self._since_recovery += 1

        # Default: do not terminate early in the very first steps
        terminate = False
        if self._count >= self.min_steps:
            # Deterministic churn on consecutive lows
            if self._low_streak >= self.low_streak_threshold:
                terminate = True
            else:
                # Probabilistic churn when EMA is low
                # Probability grows as EMA drops below threshold
                # Use EMA only after minimum samples, otherwise use current rating
                basis = float(self._ema) if self._count >= self.ema_min_samples else float(rating)
                gap = max(0.0, self.low_threshold - basis)
                p = 1.0 - np.exp(-self.prob_scale * gap)
                # If user is recovering (several good ratings), reduce churn probability
                if self.recovery_bonus > 0 and self._since_recovery >= 2:
                    p = max(0.0, p - self.recovery_bonus)
                churn = bool(self.rng.choice([True, False], p=[p, 1 - p]))
                terminate = churn

        return float(rating), terminate

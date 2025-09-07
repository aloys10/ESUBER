import re
from typing import List

from environment.LLM import LLMRater
from environment.LLM.llm import LLM
from environment.memory import UserMovieInteraction
from environment.movies import Movie
from environment.users import User


class ThirdPersonDescriptive09_CoTLite_OurSys(LLMRater):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.system_prompt = (
            "You are a movie rating assistant. Think carefully about the movie and user preferences, but output ONLY a single digit rating from 0-9. "
            "IMPORTANT: Do your reasoning internally, then output just the number."
        )

    def adjust_rating_in(self, rating):
        return rating

    def adjust_rating_out(self, rating):
        return rating

    def adjust_text_in(self, text, do_rename=True):
        # keep numeric text as-is; optionally anonymize names if needed later
        return text

    def _get_prompt(
        self,
        user: User,
        movie: Movie,
        num_interacted: int,
        interactions: List[UserMovieInteraction],
        retrieved_items: List[Movie],
        do_rename=True,
    ):
        # Build compact history string
        history = ""
        for m, i in zip(retrieved_items, interactions):
            history += f'"{m.title}" ({int(self.adjust_rating_in(i.rating))}), '
        if len(retrieved_items) > 0:
            history = history[:-2]

        # Build genres and actors
        genres_list = "-" + "\n-".join(movie.genres) if len(movie.genres) > 0 else ""
        actors_list = ", ".join([f"{a.name} ({a.gender})" for a in movie.actors])

        overview = movie.overview if len(movie.overview) > 0 else ""

        name = user.name.split(" ")[0]

        rubric = (
            "Consider these factors in your internal reasoning:\n"
            "- Content quality: movie's overall appeal and production value\n"
            "- User preference match: how well it aligns with user's stated tastes\n"
            "- Rating guidance: 7-9 for excellent matches, 4-6 for moderate matches, 1-3 for poor matches\n"
            "IMPORTANT: Think through these points internally, then output ONLY a single digit (0-9) as the rating."
        )

        prompt = (
            f"User: {user.age}y {('man' if user.gender=='M' else 'woman')}. {self.adjust_text_in(user.description)}\n"
            + (
                f"Recent history (title, rating 0-9): {history}.\n"
                if len(retrieved_items) > 0 and len(self.previous_items_features_list) > 0
                else ""
            )
            + f"Movie: \"{movie.title}\" ({movie.release_date[:4]}).\n"
            + (f"Genres:\n{genres_list}\n" if "genres" in self.current_items_features_list and len(movie.genres) > 0 else "")
            + (f"Main actors: {actors_list}.\n" if "actors" in self.current_items_features_list and len(movie.actors) > 0 else "")
            + (f"Avg rating: {round(self.adjust_rating_in(movie.vote_average),1)} (0-9).\n" if "vote_average" in self.current_items_features_list and movie.vote_average > 0 else "")
            + rubric
        )

        initial_assistant = "Reasoning: "

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant_start", "content": initial_assistant},
        ]

    def _get_few_shot_prompts(self):
        return []

    def _get_prompt_explanation(self, prompt, rating):
        # 创建解释prompt，要求LLM提供详细的评分理由
        explanation_prompt = [
            {"role": "user", "content": f"Explain why you rated this movie {int(self.adjust_rating_in(rating))}/9. Be specific about the user's preferences and movie content. One sentence only."},
            {"role": "assistant_start", "content": f"This movie gets {int(self.adjust_rating_in(rating))}/9 because "}
        ]
        return explanation_prompt



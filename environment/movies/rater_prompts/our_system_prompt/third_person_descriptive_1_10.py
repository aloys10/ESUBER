import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re

from environment.memory import UserMovieInteraction
from environment.movies import Movie, MoviesLoader
from environment.users import User


class ThirdPersonDescriptive110_OurSys(LLMRater):
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
        self.cache_few_shot_prompts = None

        self.request_scale = "1-10"

        self.system_prompt = (
            "You are a highly sophisticated movie rating assistant with advanced reasoning capabilities. "
            "Your mission is to deliver personalized movie recommendations through careful analysis. "
            "When rating a movie, follow this Chain of Thought process:\n"
            "1. Analyze the user's personality traits and movie preferences\n"
            "2. Consider the user's viewing history and rating patterns\n"
            "3. Evaluate the movie's content, genre, and themes\n"
            "4. Synthesize all information to predict the user's rating\n\n"
            "Output: Provide only a single integer rating from 1 to 10."
        )

    def adjust_rating_in(self, rating):
        return rating

    def adjust_rating_out(self, rating):
        return rating

    def adjust_text_in(self, text, do_rename=True):
        if do_rename:
            text = text.replace("Alex", "Michael")
            text = text.replace("Nicholas", "Michael")
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
        if user.gender == "M":
            gender = "man"
            pronoun = "he"
            if int(user.age) < 18:
                gender = "boy"
        else:
            gender = "woman"
            pronoun = "she"
            if int(user.age) < 18:
                gender = "girl"

        item_interaction = ""  # NOTE it should be parametrized
        for m, i in zip(retrieved_items, interactions):
            item_interaction += (
                f'"{m.title}" ({int(self.adjust_rating_in(i.rating))}), '
            )
        if len(retrieved_items) > 0:
            item_interaction = item_interaction[:-2]  # remove last comma

        genres_list = ""
        for g in movie.genres:
            genres_list += f"-{g}\n"

        actors_list = ""
        for a in movie.actors:
            actors_list += f"{a.name} ({a.gender}), "
        if len(movie.actors) > 0:
            actors_list = actors_list[:-2]

        if len(movie.overview) > 0:
            overview = movie.overview[0].lower() + movie.overview[1:]
        else:
            overview = ""

        name = user.name.split(" ")[0]
        # NOTE: this is a hack to make sure that the name is not the same as the 2 possible names used in the few-shot prompts
        name = self.adjust_text_in(name, do_rename)

        # 添加逻辑链推理指导
        reasoning_guide = (
            "Please follow this reasoning process:\n"
            "1. Consider the user's personality and preferences\n"
            "2. Analyze their viewing history and rating patterns\n"
            "3. Evaluate how well the movie matches their interests\n"
            "4. Provide a reasoned rating prediction\n\n"
        )

        prompt = (
            f"User: {user.age}y {gender}. {self.adjust_text_in(user.description, do_rename)}\n"
            + (
                f"Recent history (title, rating 1-10): {item_interaction}.\n"
                if len(retrieved_items) > 0
                and len(self.previous_items_features_list) > 0
                else ""
            )
            + f'Movie: "{movie.title}" ({movie.release_date[:4]}).\n'
            + f"Description: {overview}\n"
            + (
                f"Genres:\n{genres_list}"
                if "genres" in self.current_items_features_list
                and len(movie.genres) > 0
                else ""
            )
            + (
                f"Main actors: {actors_list}.\n"
                if "actors" in self.current_items_features_list
                and len(movie.actors) > 0
                else ""
            )
            + (
                f"Avg rating: {round(self.adjust_rating_in(movie.vote_average), 1)} (1-10).\n"
                if "vote_average" in self.current_items_features_list
                and movie.vote_average > 0
                else ""
            )
            + reasoning_guide
            + "Output: Provide only a single integer rating from 1 to 10."
        )

        initial_assistant = "Reasoning: "

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant_start", "content": initial_assistant},
        ]

    def _get_few_shot_prompts(self):
        return []

    def _get_prompt_explanation(self, prompt, rating):
        initial_explanation = (
            f"{int(self.adjust_rating_in(rating))} on a scale of 1 to 10, because "
        )
        prompt[1]["content"] += initial_explanation
        return prompt

import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM import LLM
import re

from environment.memory import UserMovieInteraction
from environment.books import Book, BooksLoader
from environment.users import User
from environment.users.emotion_criteria import get_emotion_criteria_description


class ThirdPersonDescriptive15_OurSys(LLMRater):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        system_prompt="our_system_prompt",
        show_full_prompt=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
            show_full_prompt,
        )
        self.cache_few_shot_prompts = None

        self.system_prompt = (
            "You are a highly sophisticated book rating assistant with advanced reasoning capabilities. "
            "Your mission is to deliver personalized book recommendations through careful analysis. "
            "When rating a book, follow this Chain of Thought process:\n"
            "1. Analyze the user's personality traits and reading preferences\n"
            "2. Consider the user's reading history and rating patterns\n"
            "3. Evaluate the book's content, genre, and themes\n"
            "4. Synthesize all information to predict the user's rating\n\n"
            "Output: Provide only a single integer rating from 1 to 5."
            if system_prompt == "our_system_prompt"
            else None
        )
        self.request_scale = "1-5"

    def adjust_rating_in(self, rating):
        return rating

    def adjust_rating_out(self, rating):
        return rating

    def adjust_text_in(self, text, do_rename=True):
        # Keep original behavior
        return text

    def adjust_text_out(self, text):
        # Keep original behavior
        return text

    def number_to_rank(self, number):
        if number == 1:
            return "first"
        elif number == 2:
            return "second"
        elif number == 3:
            return "third"
        else:
            return f"{number}th"

    def _get_prompt(
        self,
        user: User,
        book: Book,
        num_interacted: int,
        interactions: List[UserMovieInteraction],
        retrieved_items: List[Book],
        do_rename=True,
    ):
        gender = "man" if user.gender == "M" else "woman"
        pronoun = "he" if user.gender == "M" else "she"

        description = book.description
        categories_list = ", ".join(book.categories)
        authors_list = ", ".join(book.authors)

        item_interaction = "; ".join(
            [
                f'"{item.title}" ({round(self.adjust_rating_in(interaction.rating),1)}/5)'
                for item, interaction in zip(retrieved_items, interactions)
            ]
        )

        name = user.name.split(" ")[0]
        name = self.adjust_text_in(name, do_rename)

        author_info = ""
        if "authors" in self.current_items_features_list and len(book.authors) > 1:
            author_info = f"The authors of the book are: {authors_list}."
        elif "authors" in self.current_items_features_list and len(book.authors) == 1:
            author_info = f"The author of the book is {authors_list}."

        # 情绪特征描述，供 LLM 参考
        traits_desc = get_emotion_criteria_description(
            getattr(user, "activity_level", 2),
            getattr(user, "conformity_level", 2),
            getattr(user, "diversity_level", 2),
        )

        # 添加逻辑链推理指导
        reasoning_guide = (
            "Please follow this reasoning process:\n"
            "1. Consider the user's personality and preferences\n"
            "2. Analyze their reading history and rating patterns\n"
            "3. Evaluate how well the book matches their interests\n"
            "4. Provide a reasoned rating prediction\n\n"
        )

        prompt = (
            f"{name} is a {user.age} years old {gender},"
            f" {pronoun} is {self.adjust_text_in(user.description, do_rename)}\n"
            + f"{traits_desc}\n"
            + (
                f"{name} has previously read the following books (in"
                f" parentheses are the ratings {pronoun} gave on a scale of 1 to 5):"
                f" {item_interaction}.\n"
                if len(retrieved_items) > 0
                and len(self.previous_items_features_list) > 0
                else ""
            )
            + f'Consider the book "{book.title}", released in'
            f" {book.published_year},"
            f" which is described as follows: {description}"
            + (
                f' The book "{book.title}" belongs to the following categories:\n'
                f"{categories_list}"
                if "categories" in self.current_items_features_list
                and len(book.categories) > 0
                else ""
            )
            + author_info
            + (
                f' On average, people rate the book "{book.title}"'
                f" {round(self.adjust_rating_in(book.vote_average), 1)} on a scale of"
                " 1 to 5."
                if "vote_average" in self.current_items_features_list
                and book.vote_average > 0
                else ""
            )
            + f' {name} reads the book "{book.title}" for the'
            f" {self.number_to_rank(num_interacted+1)} time.\n\n"
            + reasoning_guide
            + "Output: Provide only a single integer rating from 1 to 5."
        )

        return [
            {"role": "user", "content": prompt},
        ]

    def _get_few_shot_prompts(self):
        return []

    def _get_prompt_explanation(self, prompt, rating):
        return prompt

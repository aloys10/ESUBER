from typing import List

from environment.LLM import LLMRater
from environment.LLM import LLM
from environment.memory import UserMovieInteraction

from environment.users import User
from environment.users.emotion_criteria import get_emotion_criteria_description


class ThirdPersonDescriptive15_CoTLite_OurSys(LLMRater):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
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
        # Enhanced system prompt with detailed chain-of-thought reasoning guidance
        self.system_prompt = (
            "You are a book rating assistant with advanced reasoning capabilities.\n"
            "Rating reasoning process:\n"
            "1. Analyze user's personality traits and reading preferences\n"
            "2. Consider user's reading history and rating patterns\n"
            "3. Evaluate the match between book content and user interests\n"
            "4. Synthesize all information to provide rating prediction\n\n"
            "Output: Directly provide a 1-5 rating number, no other content."
        )
        self.request_scale = "1-5"

    def adjust_rating_in(self, rating):
        return rating

    def adjust_rating_out(self, rating):
        # If input is a string, try to extract rating number
        if isinstance(rating, str):
            import re
            # Find the number in the last line
            lines = rating.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.isdigit() and 1 <= int(line) <= 5:
                    return int(line)
            # If not found, try to extract from the entire text
            rating_match = re.search(r'\b([1-5])\b', rating)
            if rating_match:
                return int(rating_match.group(1))
        return rating

    def adjust_text_in(self, text, do_rename=True):
        return text

    def adjust_text_out(self, text):
        return text

    def _get_prompt(
        self,
        user: User,
        book: "Book",
        num_interacted: int,
        interactions: List[UserMovieInteraction],
        retrieved_items: List["Book"],
        do_rename=True,
    ):
        # Build auxiliary strings
        categories_list = ", ".join(book.categories)
        authors_list = ", ".join(book.authors)
        history = "; ".join(
            [
                f'"{item.title}" ({round(self.adjust_rating_in(interaction.rating),1)}/5)'
                for item, interaction in zip(retrieved_items, interactions)
            ]
        )

        # Enhanced reasoning guidance
        reasoning_guide = (
            "Reasoning requirements:\n"
            "- Consider user's emotional traits: activity level, conformity, diversity pursuit\n"
            "- Analyze rating patterns and preferences in reading history\n"
            "- Evaluate the match between book genres, themes and user interests\n"
            "- Strictly rate 1-5 integers based only on given context\n\n"
        )

        # Emotion traits description
        traits_desc = get_emotion_criteria_description(
            getattr(user, "activity_level", 2),
            getattr(user, "conformity_level", 2),
            getattr(user, "diversity_level", 2),
        )

        prompt = (
        
            f"User: {user.age}y {('man' if user.gender=='M' else 'woman')}. {self.adjust_text_in(user.description)}\n"
            + traits_desc + "\n"
            + (
                f"Recent history (title, rating 1-5): {history}.\n"
                if len(retrieved_items) > 0 and len(self.previous_items_features_list) > 0
                else ""
            )
            + f"Book: \"{book.title}\" ({book.published_year}).\n"
            + (f"Categories:\n{categories_list}\n" if "categories" in self.current_items_features_list and len(book.categories) > 0 else "")
            + (f"Authors: {authors_list}.\n" if "authors" in self.current_items_features_list and len(book.authors) > 0 else "")
            + (f"Avg rating: {round(self.adjust_rating_in(book.vote_average),1)} (1-5).\n" if "vote_average" in self.current_items_features_list and book.vote_average > 0 else "")
            + reasoning_guide
            + "Output: Directly provide a 1-5 rating number, no other content."
        )

        initial_assistant = "Reasoning: "

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant_start", "content": initial_assistant},
        ]

    def _get_few_shot_prompts(self):
        return []

    def _get_prompt_explanation(self, prompt, rating):
        # No additional explanation prompt needed as it's already required in the main prompt
        return prompt



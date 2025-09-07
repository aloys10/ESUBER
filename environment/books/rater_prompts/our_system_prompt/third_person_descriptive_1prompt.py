import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive import ThirdPersonDescriptive15_OurSys
from environment.memory import UserMovieInteraction
from environment.books import Book, BooksLoader
from environment.users import User
from environment.users.emotion_criteria import get_emotion_criteria_description


class ThirdPersonDescriptive15_1Shot_CoTLite_OurSys(ThirdPersonDescriptive15_OurSys):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        switch_user=False,
        system_prompt="our_system_prompt",
        show_full_prompt=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
            system_prompt,
            show_full_prompt,
        )
        self.cache_few_shot_prompts = None
        self.switch_user = switch_user

    def adjust_rating_out(self, rating):
        """从输出中提取评分数字"""
        # 如果输入是字符串，尝试提取评分数字
        if isinstance(rating, str):
            import re
            # 查找最后一行的数字
            lines = rating.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.isdigit() and 1 <= int(line) <= 5:
                    return int(line)
            # 如果没有找到，尝试从整个文本中提取
            rating_match = re.search(r'\b([1-5])\b', rating)
            if rating_match:
                return int(rating_match.group(1))
        return rating

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

        # 增强的推理指导 - CoT-Lite逻辑链
        reasoning_guide = (
            "推理要求：\n"
            "- 考虑用户的情感特质：活跃度、从众性、多样性追求\n"
            "- 分析阅读历史中的评分模式和偏好\n"
            "- 评估书籍类型、主题与用户兴趣的匹配度\n"
            "- 考虑书籍质量、用户可能的不喜欢因素\n"
            "- 评分标准：\n"
            "  1分：完全不匹配/用户会讨厌\n"
            "  2分：不太匹配/用户不太喜欢\n"
            "  3分：一般匹配/用户感觉一般\n"
            "  4分：较好匹配/用户会喜欢\n"
            "  5分：完美匹配/用户会非常喜欢\n"
            "- 严格按1-5整数评分，仅基于给定上下文\n\n"
        )

        # Emotion traits description
        traits_desc = get_emotion_criteria_description(
            getattr(user, "activity_level", 2),
            getattr(user, "conformity_level", 2),
            getattr(user, "diversity_level", 2),
        )

        prompt = (
            "重要：必须用中文回答，严格按照以下格式！\n"
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
            + "输出：直接给出1-5的评分数字，不要其他内容。"
        )

        initial_assistant = "Reasoning: "

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant_start", "content": initial_assistant},
        ]

    def _get_few_shot_prompts(self):
        if self.cache_few_shot_prompts is None:
            user = User(
                "Emilia",
                "F",
                20,
                (
                    "an avid reader, she spends much of her free time lost in the pages"
                    " of books, especially those filled with magical worlds, exciting"
                    " adventures and tales of elves. Her passion for the magical realms"
                    " of literature is evident in her vivid imagination and the way her"
                    " eyes light up when discussing stories. As well as reading, she"
                    " enjoys drawing, attending book club meetings, stargazing, sipping"
                    " tea on rainy days, baking and getting lost in stories about"
                    " elves."
                ),
            )
            """
            First element is the query book
            Last three are the history books
            """
            books = [
                Book(
                    id="58",
                    title="Harry Potter and the Prisoner of Azkaban",
                    description=(
                        "Harry Potter, along with his best friends, Ron and Hermione,"
                        " is about to start his third year at Hogwarts School of"
                        " Witchcraft and Wizardry. Harry can't wait to get back to"
                        " school after the summer holidays. (Who wouldn't if they lived"
                        " with the horrible Dursleys?) But when Harry gets to Hogwarts,"
                        " the atmosphere is tense. There's an escaped mass murderer on"
                        " the loose, and the sinister prison guards of Azkaban have"
                        " been called in to guard the school..."
                    ),
                    description_embedding=[],
                    authors=["J.K. Rowling"],
                    publisher="",
                    published_year="1999",
                    categories=["Fiction", "Young Adult", "Magic", "Classic"],
                    vote_average=4.58,
                ),
                Book(
                    id="58",
                    title="Harry Potter and the Chamber of Secrets",
                    description=(
                        "Ever since Harry Potter had come home for the summer, the"
                        " Dursleys had been so mean and hideous that all Harry wanted"
                        " was to get back to the Hogwarts School for Witchcraft and"
                        " Wizardry. But just as he's packing his bags, Harry receives a"
                        " warning from a strange impish creature who says that if Harry"
                        " returns to Hogwarts, disaster will strike. And strike it"
                        " does. For in Harry's second year at Hogwarts, fresh torments"
                        " and horrors arise, including an outrageously stuck-up new"
                        " professor and a spirit who haunts the girls' bathroom. But"
                        " then the real trouble begins – someone is turning Hogwarts"
                        " students to stone. Could it be Draco Malfoy, a more poisonous"
                        " rival than ever? Could it possibly be Hagrid, whose"
                        " mysterious past is finally told? Or could it be the one"
                        " everyone at Hogwarts most suspects… Harry Potter himself!"
                    ),
                    description_embedding=[],
                    authors=["J.K. Rowling"],
                    publisher="",
                    published_year="1998",
                    categories=["Fiction", "YoungAdult", "Magic", "Classic"],
                    vote_average=4.43,
                ),
                Book(
                    id="58",
                    title="Harry Potter and the Philosopher's Stone",
                    description=(
                        "Harry Potter thinks he is an ordinary boy - until he is"
                        " rescued by an owl, taken to Hogwarts School of Witchcraft and"
                        " Wizardry, learns to play Quidditch and does battle in a"
                        " deadly duel. The Reason ... HARRY POTTER IS A WIZARD!"
                    ),
                    description_embedding=[],
                    authors=["J.K. Rowling"],
                    publisher="",
                    published_year="1997",
                    categories=["Fiction", "Young Adult", "Magic", "Classic"],
                    vote_average=4.47,
                ),
                Book(
                    id="58",
                    title="Eragon",
                    description="One boy...One dragon...A world of adventure.",
                    description_embedding=[],
                    authors=["Christopher Paolini"],
                    publisher="",
                    published_year="1997",
                    categories=[
                        "Fantasy",
                        "Young Adult",
                        "Fiction",
                        "Dragons",
                        "Adventures",
                        "Magic",
                    ],
                    vote_average=4.47,
                ),
            ]
            book = books[0]
            num_interacted = 0
            interactions = [
                UserMovieInteraction(5, 0, 1),  # Harry Potter系列 - 完美匹配
                UserMovieInteraction(4, 0, 1),  # 较好匹配
                UserMovieInteraction(2, 0, 1),  # 不太匹配，增加低分示例
            ]
            retrieved_items = books[1:]
            prompt = self._get_prompt(
                user,
                book,
                num_interacted,
                interactions,
                retrieved_items,
                do_rename=False,
            )
            explanation = "5"

            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {
                    "role": "assistant",
                    "content": prompt[1]["content"] + explanation,
                },
            ]
        return self.cache_few_shot_prompts

import numpy as np
import argparse
import pandas as pd
import os

print("程序开始执行...")

from environment import LLM
from environment.books.configs import get_base_parser, get_enviroment_from_args

print("导入模块完成...")


def main():
    llm = LLM.load_LLM("deepseek-chat")

    parser = get_base_parser()
    # 附加脚本级参数
    parser.add_argument("--replay-csv", type=str, default=None, help="从CSV回放book_id进行交互")
    parser.add_argument("--output-csv", type=str, default=None, help="将交互日志写到CSV")
    parser.add_argument("--quiet", action="store_true", help="静默模式，减少打印")
    parser.add_argument("--max-steps", type=int, default=5, help="最大步数（无回放时生效）")
    parser.add_argument("--num-users", type=int, default=1, help="模拟用户数量")
    parser.add_argument("--user-start-id", type=int, default=None, help="起始用户ID（None为随机）")


    parser.set_defaults(
        llm_model="deepseek-chat",
        llm_rater="0Shot_cotlite_our",
        items_retrieval="decay_emotion_3",
        consider_arousal=True,
        reward_shaping="churn_satisfaction",
        churn_ema_alpha=0.3,
        churn_low_threshold=2.0,
        churn_low_streak_threshold=2,
        churn_min_steps=5,
        churn_prob_scale=0.3,
        churn_ema_min_samples=1,
        churn_recovery_bonus=0.0,
        book_dataset="books_amazon/postprocessed_books",
        user_dataset="detailed",
        perturbator="none",
        seed=42,
    )
    args = parser.parse_args()

    # 直接创建环境，跳过check_env
    from environment.books.books_loader import BooksLoader
    from environment.books.books_retrieval import SimpleBookRetrieval
    from environment.env import Simulatio4RecSys
    from environment.users import UsersCSVLoader
    from environment.items_selection import GreedySelector
    from environment.reward_perturbator import NoPerturbator
    from environment.reward_shaping import RewardReshapingChurnSatisfaction
    from environment.books.rater_prompts.our_system_prompt.third_person_descriptive_cotlite_1_5 import ThirdPersonDescriptive15_CoTLite_OurSys
    import os

    env = Simulatio4RecSys(
        render_mode=None,
        items_loader=BooksLoader(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "environment/books/datasets/",
                args.book_dataset + "_embeddings" + ".csv",
            )
        ),
        users_loader=UsersCSVLoader("users_600", os.path.join(os.path.dirname(__file__), "environment/books/users_generation/datasets/")),
        items_selector=GreedySelector(42),
        reward_perturbator=NoPerturbator(seed=42, stepsize=1.0, min_rating=1.0, max_rating=5.0),
        items_retrieval=SimpleBookRetrieval(3),
        llm_rater=ThirdPersonDescriptive15_CoTLite_OurSys(
            llm,
            current_items_features_list=["title", "description", "categories", "authors", "vote_average"],
            previous_items_features_list=["title", "rating"],
            llm_query_explanation=True,
        ),
        reward_shaping=RewardReshapingChurnSatisfaction(
            ema_alpha=args.churn_ema_alpha,
            low_threshold=args.churn_low_threshold,
            low_streak_threshold=args.churn_low_streak_threshold,
            min_steps=args.churn_min_steps,
            prob_scale=args.churn_prob_scale,
            ema_min_samples=args.churn_ema_min_samples,
            recovery_bonus=args.churn_recovery_bonus,
            stepsize=1.0,
            min_rating=1.0,
            max_rating=5.0,
            seed=42,
        ),
    )

    # 多用户：外层循环
    for user_idx in range(args.num_users):
        start_uid = args.user_start_id if args.user_start_id is not None else None
        current_uid = (start_uid + user_idx) if start_uid is not None else None
        obs, info = env.reset(user_id=current_uid)
        if not args.quiet:
            print(f" 书籍推荐系统启动！(用户 {user_idx+1}/{args.num_users})")
            print(f" 用户: {obs['user_name']}, 年龄: {obs['user_age']}")
            print(f" 推荐池大小: {env.action_space.n} 本书")
            print("=" * 80)

    # 回放模式：从CSV读取book_id（或item_id/id），可选user_id分组
        if args.replay_csv is not None and os.path.exists(args.replay_csv):
            df = pd.read_csv(args.replay_csv)
            candidate_cols = ["book_id", "item_id", "id"]
            col = next((c for c in candidate_cols if c in df.columns), None)
            if col is None:
                raise ValueError(f"回放CSV缺少必须列: {candidate_cols}")

            steps_done = 0
            for _, row in df.iterrows():
                # 若回放文件包含user_id且与当前用户不同，可选择跳过或切换；此处复用同一序列以便批量生成
                item_id = int(row[col])
                if item_id not in env.item_to_action:
                    if not args.quiet:
                        print(f"跳过不存在的book_id={item_id}")
                    continue
                action = env.item_to_action[item_id]

                if not args.quiet:
                    print(f"\n === 回放(用户 {user_idx+1}) 步骤 {steps_done+1} === 书籍ID {item_id} -> 动作 {action}")

                obs, reward, terminated, truncated, info = env.step(action)

                if not args.quiet:
                    # print(f"    原始评分: {info.get('LLM_rating', 'N/A')}")
                    print(f"    最终评分: {reward}/5")
                    if info.get('LLM_explanation'):
                        print(f"    解释: {info.get('LLM_explanation')[:200]}...")

                if args.output_csv:
                    out = {
                        "user_id": env._user.id,
                        "user_name": env._user.name,
                        "user_session": user_idx + 1,
                        "step": len(env._items_interact),
                        "book_id": item_id,
                        "action": action,
                        "rating": reward,
                        "ema": info.get("reward_ema"),
                        "low_streak": info.get("reward_low_streak"),
                        "terminated": terminated,
                    }
                    pd.DataFrame([out]).to_csv(
                        args.output_csv, mode="a", index=False, header=not os.path.exists(args.output_csv)
                    )

                steps_done += 1
                if terminated:
                    if not args.quiet:
                        print("用户终止，本次回放结束。")
                    break
            if not args.quiet:
                print(f"回放结束（用户 {user_idx+1}）。")
            # 回放模式下：继续下一个用户
            continue

    # 否则：演示模式，随机从动作空间采样
    max_steps = max(1, int(args.max_steps))
    for t in range(max_steps):
        action = int(np.random.randint(env.action_space.n))
        item_id = env.action_to_item[action]

        if not args.quiet:
            print(f"\n === 第 {t+1} 轮推荐（用户 {user_idx+1}） ===")
            print(f" 推荐器选择: 动作 {action} -> 书籍ID {item_id}")
            book = env.items_loader.load_items_from_ids([item_id])[0]
            print(f" 推荐书籍: 《{book.title}》")
            print(f"  简介: {book.description[:100] if book.description else '无简介'}...")
            print(f"  分类: {', '.join(book.categories) if book.categories else '未知'}")
            print(f"  作者: {', '.join(book.authors) if book.authors else '未知'}")
            print(f"  平均评分: {book.vote_average}/5")

        obs, reward, terminated, truncated, info = env.step(action)

        if not args.quiet:
            print(f"\n 用户模拟器(LLM)的回应:")
            print(f"    原始评分: {info.get('LLM_rating', 'N/A')}")
            print(f"    最终评分: {reward}/5")
            if info.get('LLM_explanation'):
                print(f"    解释: {info.get('LLM_explanation')}")
                print(f"    解释长度: {len(info.get('LLM_explanation', ''))}")
                print(f"    解释前50字符: {info.get('LLM_explanation', '')[:50]}")
                print(f"    解释后50字符: {info.get('LLM_explanation', '')[-50:]}")
            print(f"    满意度EMA: {info.get('reward_ema', 'N/A'):.2f}")
            print(f"    连续低分: {info.get('reward_low_streak', 'N/A')}")

        if args.output_csv:
            out = {
                "user_id": env._user.id,
                "user_name": env._user.name,
                "user_session": user_idx + 1,
                "step": t + 1,
                "book_id": item_id,
                "action": action,
                "rating": reward,
                "ema": info.get("reward_ema"),
                "low_streak": info.get("reward_low_streak"),
                "terminated": terminated,
            }
            pd.DataFrame([out]).to_csv(
                args.output_csv, mode="a", index=False, header=not os.path.exists(args.output_csv)
            )

        if not args.quiet:
            print(f"\n👤 用户状态:")
            print(f"    是否终止: {'是' if terminated else '否'}")
            print(f"    是否截断: {'是' if truncated else '否'}")
            print("-" * 80)

        if terminated or truncated:
            break

    if not args.quiet:
        print(f"\n 推荐会话结束！（用户 {user_idx+1}）")


if __name__ == "__main__":
    main()



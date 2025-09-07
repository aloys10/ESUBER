import numpy as np
import argparse
import pandas as pd
import os

from environment import LLM
from environment.movies.configs import get_base_parser, get_enviroment_from_args


def main():
    llm = LLM.load_LLM("glm-4.5-flash")

    parser = get_base_parser()
    # 追加本地脚本参数（不影响已有配置）
    parser.add_argument("--replay-csv", type=str, default=None, help="从CSV回放item_id进行交互")
    parser.add_argument("--output-csv", type=str, default=None, help="将交互日志写到CSV")
    parser.add_argument("--quiet", action="store_true", help="静默模式，减少打印")
    parser.add_argument("--max-steps", type=int, default=5, help="最大步数（无回放时生效）")
    parser.add_argument("--num-users", type=int, default=1, help="模拟用户数量")
    parser.add_argument("--user-start-id", type=int, default=None, help="起始用户ID（None为随机）")

    # 之前硬编码参数会忽略命令行，这里改为设置默认值+真实解析命令行
    parser.set_defaults(
        llm_model="glm-4.5-flash",
        llm_rater="0Shot_cotlite_our",
        items_retrieval="decay_emotion_3",
        consider_arousal=True,
        reward_shaping="churn_satisfaction",
        churn_ema_alpha=0.1,   # 降低EMA更新速度，让用户更稳定
        churn_low_threshold=2.0,  # 降低低分阈值，让用户更宽容
        seed=42,
    )
    args = parser.parse_args()

    env = get_enviroment_from_args(llm, args, seed=42)
    # 开启解释查询，返回评分之外的解释和HTML对话
    try:
        env.rating_prompt.llm_query_explanation = True
    except Exception:
        pass

    # 多用户模拟循环
    for user_idx in range(args.num_users):
        # 计算当前用户ID
        if args.user_start_id is not None:
            current_user_id = args.user_start_id + user_idx
        else:
            current_user_id = None  # 随机选择
        
        # 重置环境，选择新用户
        obs, info = env.reset(user_id=current_user_id)
        
        if not args.quiet:
            print(f"\n{'='*80}")
            print(f"🎬 电影推荐系统 - 用户 {user_idx + 1}/{args.num_users}")
            print(f"👤 用户: {obs['user_name']}, 年龄: {obs['user_age']}, 描述: {obs['user_description']}")
            print(f"🎯 推荐池大小: {env.action_space.n} 部电影")
            print(f"{'='*80}")
    
    # 如果指定了回放CSV，则逐行读取item_id并回放
    if args.replay_csv is not None and os.path.exists(args.replay_csv):
        df = pd.read_csv(args.replay_csv)
        # 兼容多种列名
        candidate_cols = ["item_id", "movie_id", "id"]
        col = next((c for c in candidate_cols if c in df.columns), None)
        if col is None:
            raise ValueError(f"回放CSV缺少必须列: {candidate_cols}")

        # 回放模式：为每个用户回放相同的数据
        for user_idx in range(args.num_users):
            if args.user_start_id is not None:
                current_user_id = args.user_start_id + user_idx
            else:
                current_user_id = None
            
            obs, info = env.reset(user_id=current_user_id)
            
            if not args.quiet:
                print(f"\n🔄 用户 {user_idx + 1}/{args.num_users} 开始回放...")
            
            steps_done = 0
            for idx, row in df.iterrows():
                item_id = int(row[col])
                if item_id not in env.item_to_action:
                    if not args.quiet:
                        print(f"跳过不存在的item_id={item_id}")
                    continue
                action = env.item_to_action[item_id]

                if not args.quiet:
                    print(f"\n === 回放步骤 {steps_done+1} ===")
                    print(f"🎬 回放电影ID {item_id} -> 动作 {action}")

                obs, reward, terminated, truncated, info = env.step(action)

                if args.output_csv:
                    out_dict = {
                        "user_id": env._user.id,
                        "user_name": env._user.name,
                        "user_session": user_idx + 1,
                        "step": steps_done + 1,
                        "movie_id": item_id,
                        "action": action,
                        "rating": reward,
                        "ema": info.get("reward_ema"),
                        "low_streak": info.get("reward_low_streak"),
                        "terminated": terminated,
                    }
                    pd.DataFrame([out_dict]).to_csv(
                        args.output_csv, mode="a", index=False, header=not os.path.exists(args.output_csv)
                    )

                steps_done += 1
                if terminated:
                    if not args.quiet:
                        print("用户终止，本次回放结束。")
                    break
            if not args.quiet:
                print(f"用户 {user_idx + 1} 回放完成，共执行 {steps_done} 步。")
        return

    # 否则，走演示模式（随机动作）
    max_steps = max(1, int(args.max_steps))
    
    # 为每个用户运行推荐会话
    for user_idx in range(args.num_users):
        if args.user_start_id is not None:
            current_user_id = args.user_start_id + user_idx
        else:
            current_user_id = None
        
        # 重置环境，选择新用户
        obs, info = env.reset(user_id=current_user_id)
        
        if not args.quiet:
            print(f"\n🔄 用户 {user_idx + 1}/{args.num_users} 开始推荐会话...")
        
        for t in range(max_steps):
            # 从动作空间中抽一个动作（注意：动作是连续0..N-1，不等于真实movie_id）
            action = int(np.random.randint(env.action_space.n))
            item_id = env.action_to_item[action]

            if not args.quiet:
                print(f"\n === 第 {t+1} 轮推荐 ===")
                print(f"🎬 推荐器选择: 动作 {action} -> 电影ID {item_id}")
                movie = env.items_loader.load_items_from_ids([item_id])[0]
                print(f"  推荐电影: 《{movie.title}》")
                print(f"  简介: {movie.overview[:100]}...")
                print(f"  类型: {', '.join(movie.genres) if movie.genres else '未知'}")
                print(f"  平均评分: {movie.vote_average}/10")

            obs, reward, terminated, truncated, info = env.step(action)

            if not args.quiet:
                print(f"\n用户模拟器(LLM)的回应:")
                print(f"   原始评分: {info.get('LLM_rating', 'N/A')}")
                print(f"   最终评分: {reward}/10")
                if info.get('LLM_explanation'):
                    print(f"   解释: {info.get('LLM_explanation')[:200]}...")
                print(f"   满意度EMA: {info.get('reward_ema', 'N/A'):.2f}")
                print(f"   连续低分: {info.get('reward_low_streak', 'N/A')}")

            if args.output_csv:
                out_dict = {
                    "user_id": env._user.id,
                    "user_name": env._user.name,
                    "user_session": user_idx + 1,
                    "step": t + 1,
                    "movie_id": item_id,
                    "action": action,
                    "rating": reward,
                    "ema": info.get("reward_ema"),
                    "low_streak": info.get("reward_low_streak"),
                    "terminated": terminated,
                }
                pd.DataFrame([out_dict]).to_csv(
                    args.output_csv, mode="a", index=False, header=not os.path.exists(args.output_csv)
                )

            if not args.quiet:
                print(f"\n用户状态:")
                print(f"    是否终止: {'是' if terminated else '否'}")
                print(f"   是否截断: {'是' if truncated else '否'}")
                if terminated:
                    print(f"   终止原因: 用户满意度过低，选择离开")
                print("-" * 80)

            # 调试：显示详细的终止信息
            if terminated:
                print(f"🔴 用户终止！原因：满意度驱动")
                break
            elif truncated:
                print(f"🟡 环境截断！原因：达到最大步数")
                break

        if not args.quiet:
            print(f"\n🎉 用户 {user_idx + 1} 推荐会话结束！")
            print(f"📊 总共推荐了 {t+1} 部电影")
            print(f"{'='*80}")
    
    if not args.quiet:
        print(f"\n🎊 所有用户模拟完成！共模拟了 {args.num_users} 个用户")


if __name__ == "__main__":
    main()



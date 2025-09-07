#!/usr/bin/env python3
"""
DeepSeek实验运行脚本
专门用于运行使用DeepSeek模型的书籍推荐系统实验
"""

import os
import argparse
import sys

# 检查环境变量设置
def check_environment():
    """检查必要的环境变量是否已设置"""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置DeepSeek API密钥")
        print("请设置以下环境变量之一:")
        print("  - DEEPSEEK_API_KEY: DeepSeek专用API密钥")
        print("  - OPENAI_API_KEY: OpenAI兼容的API密钥")
        print("\n设置方法:")
        print("  Windows PowerShell:")
        print("    $env:DEEPSEEK_API_KEY='your_api_key_here'")
        print("  Windows CMD:")
        print("    set DEEPSEEK_API_KEY=your_api_key_here")
        print("  Linux/Mac:")
        print("    export DEEPSEEK_API_KEY=your_api_key_here")
        return False
    
    print("✅ DeepSeek API密钥已设置")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行DeepSeek书籍推荐系统实验")
    parser.add_argument("--skip-sampling", action="store_true", default=False, 
                       help="跳过采样研究")
    
    parser.add_argument("--llm-rater", type=str, default="0Shot_cotlite_our",
                       choices=["2Shot_system_our", "1Shot_system_our", "0Shot_system_our", 
                               "0Shot_cotlite_our", "2Shot_system_default", "1Shot_system_default", 
                               "0Shot_system_default",
                               # 新增的逻辑链增强版本
                               "2Shot_cot_enhanced", "1Shot_cot_enhanced", "0Shot_cot_enhanced"],
                       help="LLM评分器类型")
    parser.add_argument("--items-retrieval", type=str, default="decay_emotion_3",
                       choices=["last_3", "most_similar_3", "none", "simple_3", "decay_emotion_3"],
                       help="物品检索策略")
    parser.add_argument("--user-dataset", type=str, default="detailed",
                       choices=["detailed", "sampled"],
                       help="用户数据集类型")
    parser.add_argument("--perturbator", type=str, default="none",
                       choices=["none", "gaussian", "greedy"],
                       help="奖励扰动器类型")
    parser.add_argument("--show-prompt", action="store_true", default=False,
                       help="显示完整的LLM prompt内容")
    
    args = parser.parse_args()
    
    # 检查环境变量
    if not check_environment():
        sys.exit(1)
    
    print("🚀 开始运行DeepSeek实验...")
    print(f"📊 实验配置:")
    print(f"  - 评分器: {args.llm_rater}")
    print(f"  - 检索策略: {args.items_retrieval}")
    print(f"  - 用户数据集: {args.user_dataset}")
    print(f"  - 扰动器: {args.perturbator}")
    print(f"  - 跳过采样: {args.skip_sampling}")
    print(f"  - 显示Prompt: {args.show_prompt}")
    
    # 设置环境变量
    os.environ["LLM_MODEL"] = "deepseek-chat"
    
    # 直接运行实验，而不是尝试导入
    try:
        print("\n🔬 正在运行实验...")
        
        # 添加项目根目录到Python路径
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        sys.path.insert(0, os.path.abspath(project_root))
        
        # 直接导入并运行实验
        from environment.books.books_loader import BooksLoader
        from ablations.books.src import (
            CategoryPreferenceStudy,
            HighRatingStudy,
            LowRatingStudy,
            BooksCollectionStudy,
            SamplingSubsetInteractionsStudy
        )
        from environment.LLM import load_LLM
        from environment.env import Simulatio4RecSys
        from environment.items_selection import GreedySelector
        from environment.reward_shaping import IdentityRewardShaping
        from environment.users import UsersLoader
        from environment.books.configs import (
            get_llm_rater,
            get_reward_perturbator,
            get_items_retrieval,
            get_user_dataset,
        )
        
        # 创建模型
        model = load_LLM("deepseek-chat")
        
        # 创建环境函数
        def create_env(item: str, user_loader_param):
            # 如果user_loader_param是字符串，则创建相应的用户加载器
            if isinstance(user_loader_param, str):
                if user_loader_param == "detailed":
                    from environment.users import UsersCSVLoader
                    users_loader = UsersCSVLoader("users_600", os.path.join(os.path.dirname(__file__), "..", "..", "environment/books/users_generation/datasets/"))
                elif user_loader_param == "sampled":
                    from environment.users import UsersCSVLoader
                    users_loader = UsersCSVLoader("users_600_sampled", os.path.join(os.path.dirname(__file__), "..", "..", "environment/books/users_generation/datasets/"))
                else:
                    # 默认使用detailed
                    from environment.users import UsersCSVLoader
                    users_loader = UsersCSVLoader("users_600", os.path.join(os.path.dirname(__file__), "..", "..", "environment/books/users_generation/datasets/"))
            else:
                users_loader = user_loader_param
            
            return Simulatio4RecSys(
                render_mode=None,
                items_loader=BooksLoader(item),
                users_loader=users_loader,
                items_selector=GreedySelector(),
                reward_perturbator=get_reward_perturbator(args.perturbator, seed=42),
                items_retrieval=get_items_retrieval(args.items_retrieval, args),
                llm_rater=get_llm_rater(
                    args.llm_rater, model, args.items_retrieval != "none", args.show_prompt
                ),
                reward_shaping=IdentityRewardShaping(seed=42),
            )
        
        # 设置实验名称
        exp_name = f"deepseek-chat-{args.llm_rater}-{args.perturbator}-{args.items_retrieval}"
        if args.user_dataset == "users_600_basic":
            exp_name += "-basic_users"
        
        print(f"🚀 [实验] 开始运行实验: {exp_name}")
        
        # 1. 运行类别偏好研究实验（跳过，因为已经运行过了）
        genre_study = CategoryPreferenceStudy(create_env, exp_name)
        genre_study.run()

        
        # 2. 运行高分研究实验
        print("\n⭐ [实验2] 高分研究实验...")
        high_ratings_study = HighRatingStudy(create_env, exp_name)
        high_ratings_study.run()
        print(f"✅ [实验2] 高分研究实验完成！")
        
        # 3. 运行低分研究实验
        print("\n📉 [实验3] 低分研究实验...")
        low_ratings_study = LowRatingStudy(create_env, exp_name)
        low_ratings_study.run()
        print(f"✅ [实验3] 低分研究实验完成！")
        
        # 4. 运行合集研究实验
        print("\n📖 [实验4] 合集研究实验...")
        books_collection_study = BooksCollectionStudy(create_env, exp_name, args.user_dataset)
        books_collection_study.run()
        print(f"✅ [实验4] 合集研究实验完成！")
        
        # 5. 运行采样分析（如果不跳过）
        # if not args.skip_sampling:
        #     print("\n🔍 [实验5] 采样分析实验...")
        #     sampling_analysis = SamplingSubsetInteractionsStudy(create_env, exp_name)
        #     sampling_analysis.run()
        #     print(f"✅ [实验5] 采样分析实验完成！")
        # else:
        #     print("\n⏭️ 跳过采样分析实验")
        
        print(f"\n🎉 所有实验完成！实验名称: {exp_name}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保在正确的目录中运行此脚本")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行实验时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

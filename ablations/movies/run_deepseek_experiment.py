#!/usr/bin/env python3
"""
DeepSeek实验运行脚本
专门用于运行使用DeepSeek模型的电影推荐系统实验
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
    # 在构建解析器前，先把项目根目录加入路径，保证本地包可被导入
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, os.path.abspath(project_root))
    from environment.movies.configs import OPTIONS_LLM_RATER

    parser = argparse.ArgumentParser(description="运行DeepSeek电影推荐系统实验")
    parser.add_argument("--skip-sampling", action="store_true", default=False, 
                       help="跳过采样研究")
    parser.add_argument("--llm-rater", type=str, default="0Shot_cotlite_our",
                       choices=OPTIONS_LLM_RATER,
                       help="LLM评分器类型")
    parser.add_argument("--items-retrieval", type=str, default="decay_emotion_3",
                       choices=["last_3", "most_similar_3", "none", "simple_3", "decay_emotion_3"],
                       help="物品检索策略")
    parser.add_argument("--user-dataset", type=str, default="detailed",
                       choices=["detailed", "basic", "sampled_genres"],
                       help="用户数据集类型")
    parser.add_argument("--perturbator", type=str, default="none",
                       choices=["none", "gaussian", "greedy"],
                       help="奖励扰动器类型")
    
    args = parser.parse_args()
    
    # 检查环境变量
    if not check_environment():
        sys.exit(1)
    
    print("🚀 开始运行DeepSeek电影推荐实验...")
    print(f"📊 实验配置:")
    print(f"  - 评分器: {args.llm_rater}")
    print(f"  - 检索策略: {args.items_retrieval}")
    print(f"  - 用户数据集: {args.user_dataset}")
    print(f"  - 扰动器: {args.perturbator}")
    print(f"  - 跳过采样: {args.skip_sampling}")
    
    # 设置环境变量
    os.environ["LLM_MODEL"] = "deepseek-chat"
    
    # 直接运行实验，而不是尝试导入
    try:
        print("\n🔬 正在运行实验...")
        
        
        # 直接导入并运行实验
        from environment.movies.movies_loader import MoviesLoader
        from ablations.movies.src import (
            CategoryPreferenceStudy,
            HighRatingStudy,
            LowRatingStudy,
            MoviesCollectionStudy,
            SamplingSubsetInteractionsStudy
        )
        from environment.LLM import load_LLM
        from environment.env import Simulatio4RecSys
        from environment.items_selection import GreedySelector
        from environment.reward_shaping import IdentityRewardShaping
        from environment.users import UsersLoader
        from environment.movies.configs import (
            get_llm_rater,
            get_reward_perturbator,
            get_items_retrieval,
            get_user_dataset,
        )
        
        # 创建模型
        model = load_LLM("deepseek-chat")
        
        # 添加API测试
        print("🔍 测试DeepSeek API连接...")
        try:
            test_response = model.query("请回复'测试成功'")
            print(f"✅ API测试成功: {test_response}")
        except Exception as e:
            print(f"❌ API测试失败: {e}")
            print("请检查API密钥和网络连接")
            sys.exit(1)
        
        print("🚀 API连接正常，开始创建实验环境...")
        
        # 创建环境函数
        def create_env(item: str, user_loader_param):
            # 如果user_loader_param是字符串，则创建相应的用户加载器
            if isinstance(user_loader_param, str):
                if user_loader_param == "detailed":
                    from environment.users import UsersCSVLoader
                    users_loader = UsersCSVLoader("user_features_600", os.path.join(os.path.dirname(__file__), "..", "..", "environment/movies/users_generation/datasets/"))
                elif user_loader_param == "basic":
                    from environment.users import UsersCSVLoader
                    users_loader = UsersCSVLoader("user_features_hard_600", os.path.join(os.path.dirname(__file__), "..", "..", "environment/movies/users_generation/datasets/"))
                elif user_loader_param == "sampled_genres":
                    from environment.users import UsersCSVLoader
                    users_loader = UsersCSVLoader("user_features_sampled_genres_600", os.path.join(os.path.dirname(__file__), "..", "..", "environment/movies/users_generation/datasets/"))
                else:
                    # 默认使用detailed
                    from environment.users import UsersCSVLoader
                    users_loader = UsersCSVLoader("user_features_600", os.path.join(os.path.dirname(__file__), "..", "..", "environment/movies/users_generation/datasets/"))
            else:
                users_loader = user_loader_param
            
            return Simulatio4RecSys(
                render_mode=None,
                items_loader=MoviesLoader(item),
                users_loader=users_loader,
                items_selector=GreedySelector(),
                reward_perturbator=get_reward_perturbator(args.perturbator, seed=42),
                items_retrieval=get_items_retrieval(args.items_retrieval, args),
                llm_rater=get_llm_rater(
                    args.llm_rater, model, args.items_retrieval != "none"
                ),
                reward_shaping=IdentityRewardShaping(seed=42),
            )
        
        # 设置实验名称
        exp_name = f"deepseek-chat-{args.llm_rater}-{args.perturbator}-{args.items_retrieval}"
        if args.user_dataset != "detailed":
            exp_name += f"-{args.user_dataset}"
        
        print(f"🚀 [实验] 开始运行实验: {exp_name}")
        
        # # 1. 运行类别偏好研究实验
        # print("\n🎬 [实验1] 类别偏好研究实验...")
        # category_study = CategoryPreferenceStudy(create_env, exp_name)
        # category_study.run()
        # print(f"✅ [实验1] 类别偏好研究实验完成！")
        
        # # 2. 运行高分研究实验
        # print("\n⭐ [实验2] 高分研究实验...")
        # high_ratings_study = HighRatingStudy(create_env, exp_name)
        # high_ratings_study.run()
        # print(f"✅ [实验2] 高分研究实验完成！")
        
        # # # 3. 运行低分研究实验
        # print("\n📉 [实验3] 低分研究实验...")
        # low_ratings_study = LowRatingStudy(create_env, exp_name)
        # low_ratings_study.run()
        # print(f"✅ [实验3] 低分研究实验完成！")
        
        # # 4. 运行合集研究实验
        # print("\n🎭 [实验4] 合集研究实验...")
        # movies_collection_study = MoviesCollectionStudy(create_env, exp_name, args.user_dataset)
        # movies_collection_study.run()
        # print(f"✅ [实验4] 合集研究实验完成！")
        
        # 5. 运行采样分析（如果不跳过）
        if not args.skip_sampling:
            print("\n🔍 [实验5] 采样分析实验...")
            sampling_analysis = SamplingSubsetInteractionsStudy(create_env, exp_name)
            sampling_analysis.run()
            print(f"✅ [实验5] 采样分析实验完成！")
        else:
            print("\n⏭️ 跳过采样分析实验")
        
        # print(f"\n🎉 所有实验完成！实验名称: {exp_name}")
        
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

#!/usr/bin/env python3
"""
分析奖励分布和评分机制
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_reward_distribution():
    """分析奖励分布"""
    print("=== 奖励分布分析 ===")
    
    # 加载数据
    df = pd.read_csv("tmp/rewards/train_rewards.csv")
    
    # 基本统计
    print(f"总训练步数: {len(df)}")
    print(f"奖励范围: {df['reward'].min()} - {df['reward'].max()}")
    print(f"平均奖励: {df['reward'].mean():.3f}")
    print(f"奖励标准差: {df['reward'].std():.3f}")
    
    # 奖励分布
    reward_counts = Counter(df['reward'])
    print(f"\n奖励分布:")
    for reward in sorted(reward_counts.keys()):
        count = reward_counts[reward]
        percentage = (count / len(df)) * 100
        print(f"  奖励 {reward}: {count} 次 ({percentage:.1f}%)")
    
    # 分析奖励趋势
    print(f"\n奖励趋势分析:")
    window_size = 1000
    for i in range(0, len(df), window_size):
        end_idx = min(i + window_size, len(df))
        chunk = df.iloc[i:end_idx]
        avg_reward = chunk['reward'].mean()
        print(f"  步数 {i+1}-{end_idx}: 平均奖励 {avg_reward:.3f}")
    
    return df

def check_rating_system():
    """检查评分系统配置"""
    print("\n=== 评分系统分析 ===")
    
    # 检查配置文件
    try:
        with open("environment/movies/configs.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # 查找评分相关的配置
        if "0Shot_cotlite_our" in content:
            print("当前使用的评分器: 0Shot_cotlite_our")
        
        if "ThirdPersonDescriptive09" in content:
            print("评分范围: 0-9 分")
        
        if "ThirdPersonDescriptive110" in content:
            print("评分范围: 1-10 分")
            
    except Exception as e:
        print(f"读取配置文件失败: {e}")
    
    # 检查评分提示词
    try:
        with open("environment/movies/rater_prompts/our_system_prompt/third_person_descriptive_cotlite_0_9.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # 查找评分相关的文本
        if "0-9" in content:
            print("评分器使用 0-9 分制")
        if "1-10" in content:
            print("评分器使用 1-10 分制")
            
        # 查找示例评分
        import re
        examples = re.findall(r'评分[：:]\s*(\d+)', content)
        if examples:
            print(f"提示词中的示例评分: {examples}")
            
    except Exception as e:
        print(f"读取评分提示词失败: {e}")

def analyze_episode_patterns(df):
    """分析Episode模式"""
    print("\n=== Episode模式分析 ===")
    
    # 分析Episode长度
    print(f"Episode长度统计:")
    print(f"  平均长度: {df['episode_length'].mean():.1f}")
    print(f"  最大长度: {df['episode_length'].max()}")
    print(f"  最小长度: {df['episode_length'].min()}")
    
    # 分析Episode结束时的奖励
    episode_ends = df[df['episode_length'] == 1]  # 新Episode开始
    if len(episode_ends) > 0:
        print(f"\nEpisode结束时的奖励分布:")
        end_rewards = df.iloc[episode_ends.index - 1]['reward'] if len(episode_ends) > 0 else []
        if len(end_rewards) > 0:
            end_reward_counts = Counter(end_rewards)
            for reward in sorted(end_reward_counts.keys()):
                count = end_reward_counts[reward]
                percentage = (count / len(end_rewards)) * 100
                print(f"  结束奖励 {reward}: {count} 次 ({percentage:.1f}%)")

def check_llm_rating_quality():
    """检查LLM评分质量"""
    print("\n=== LLM评分质量分析 ===")
    
    # 检查是否有评分解释记录
    try:
        # 这里可以检查是否有其他日志文件包含LLM的解释
        print("检查LLM评分质量...")
        print("建议检查以下方面:")
        print("1. LLM模型是否适合评分任务")
        print("2. 提示词是否清晰明确")
        print("3. 评分标准是否合理")
        print("4. 是否有评分偏差")
        
    except Exception as e:
        print(f"检查LLM评分质量失败: {e}")

def main():
    """主函数"""
    print("开始分析奖励分布和评分机制...")
    
    # 分析奖励分布
    df = analyze_reward_distribution()
    
    # 检查评分系统
    check_rating_system()
    
    # 分析Episode模式
    analyze_episode_patterns(df)
    
    # 检查LLM评分质量
    check_llm_rating_quality()
    
    print("\n=== 可能的原因分析 ===")
    print("奖励普遍只有3分左右的可能原因:")
    print("1. 评分系统使用0-9分制，3分可能是一个中等偏下的分数")
    print("2. LLM可能对推荐结果不够满意")
    print("3. 推荐算法可能还没有学到好的策略")
    print("4. 评分标准可能过于严格")
    print("5. 用户偏好与推荐内容匹配度不高")
    
    print("\n建议:")
    print("1. 检查LLM的评分提示词和标准")
    print("2. 分析推荐内容的质量")
    print("3. 考虑调整奖励塑形策略")
    print("4. 检查用户特征和电影特征的匹配度")

if __name__ == "__main__":
    main()

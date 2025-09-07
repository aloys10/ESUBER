"""
情绪判断标准字典
用于LLM给分时的情绪判断标准
"""

# 活动水平字典 - 描述用户对推荐系统的容忍度和退出倾向
ACTIVITY_DICT = {
    1: "An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
    2: "An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
    3: "A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory."
}

# 从众程度字典 - 描述用户评分时对历史评分的依赖程度
CONFORMITY_DICT = {
    1: "A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings.",
    2: "A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
    3: "A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings."
}

# 多样性偏好字典 - 描述用户对电影类型多样性的偏好程度
DIVERSITY_DICT = {
    1: "An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
    2: "A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",
    3: "A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization."
}

def get_emotion_criteria_description(activity_level, conformity_level, diversity_level):
    """
    获取用户的情绪判断标准描述
    
    Args:
        activity_level (int): 活动水平 (1-3)
        conformity_level (int): 从众程度 (1-3)
        diversity_level (int): 多样性偏好 (1-3)
    
    Returns:
        str: 完整的情绪判断标准描述
    """
    activity_desc = ACTIVITY_DICT.get(activity_level, "Unknown activity level")
    conformity_desc = CONFORMITY_DICT.get(conformity_level, "Unknown conformity level")
    diversity_desc = DIVERSITY_DICT.get(diversity_level, "Unknown diversity level")
    
    return f"""User Emotional Rating Criteria:

Activity Level ({activity_level}): {activity_desc}

Conformity Level ({conformity_level}): {conformity_desc}

Diversity Level ({diversity_level}): {diversity_desc}

Please consider these characteristics when rating items for this user."""

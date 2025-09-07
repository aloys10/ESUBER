import requests
import numpy as np
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# 使用用户提供的API密钥
TMDB_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNTY2OWIzODFjZDQ4NzdkMGI3OTUzMzIwMTU0NWFjYiIsIm5iZiI6MTc1NDQwNjYxNS4yOTIsInN1YiI6IjY4OTIxZWQ3OGJmMjVhZmMyZjdkMDA4MyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.YRTBHPh9BnBeSbxTkFArRCGbuZnVnptOvMALc3h7uYQ"

print("正在加载SentenceTransformer模型（这可能需要一些时间）...")
try:
    model = SentenceTransformer("sentence-transformers/sentence-t5-base")
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请确保已安装sentence-transformers: pip install sentence-transformers")
    exit(1)


def get_director(id):
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}",
    }
    url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=zh-CN"
    try:
        response = requests.get(url_credits, headers=headers, timeout=10)
        if response.status_code == 200:
            data_credits = response.json()
            for d in data_credits.get("crew", []):
                if d.get("job") == "Director":
                    return d.get("name")
        return "未知"
    except Exception as e:
        print(f"获取导演信息失败，电影ID {id}: {e}")
        return "未知"


def get_film(id):
    url_movie = f"https://api.themoviedb.org/3/movie/{id}?language=zh-CN"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}",
    }
    try:
        response = requests.get(url_movie, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"警告: 无法获取电影ID {id} 的数据。状态码: {response.status_code}")
            return None
        data = response.json()
    except Exception as e:
        print(f"获取电影数据时出错，ID {id}: {e}")
        return None

    # 移除不需要的字段
    movies_keys_to_remove = [
        "backdrop_path",
        "belongs_to_collection",
        "homepage",
        "poster_path",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "status",
        "tagline",
        "video",
    ]
    for k in movies_keys_to_remove:
        if k in data:
            data.pop(k)

    url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=zh-CN"
    try:
        response = requests.get(url_credits, headers=headers, timeout=10)
        if response.status_code == 200:
            data_credits = response.json()
        else:
            data_credits = {}
    except Exception as e:
        print(f"获取演职员信息时出错，ID {id}: {e}")
        data_credits = {}

    if "cast" not in data_credits:
        top_actors = []
    else:
        top_actors = data_credits["cast"][0:2]
    
    actors_keys_to_remove = [
        "adult",
        "known_for_department",
        "original_name",
        "cast_id",
        "credit_id",
        "order",
        "profile_path",
    ]

    for t in top_actors:
        for k in actors_keys_to_remove:
            if k in t:
                t.pop(k)

    data["actors"] = top_actors

    # 生成文本嵌入
    overview_text = data.get("overview", "")
    title_text = data.get("title", "")
    text_to_encode = overview_text if len(overview_text) > 0 else title_text
    
    try:
        embeddings = model.encode(text_to_encode)
        data["overview_embedding"] = embeddings.tolist()
    except Exception as e:
        print(f"为电影ID {id} 编码文本时出错: {e}")
        data["overview_embedding"] = []
    
    data["director"] = get_director(id)
    return data


# import tqdm

# # %%
# import json


def sample(ids, file_path):
    films = {}
    counter = 0
    for id in tqdm(ids):
        id = int(id)
        try:
            film_data = get_film(id)
            if film_data and film_data.get("overview") and film_data["overview"] != "":
                films[id] = film_data
            else:
                print(f"电影 {id} 没有简介信息")
        except Exception as e:
            print(f"处理电影ID {id} 时出错: {e}")
        counter += 1
        if counter % 100 == 0:
            print(f"已处理 {counter} 部电影")
        # 避免API请求过于频繁
        time.sleep(0.1)
    
    print(f"\n处理完成！共获取 {len(films)} 部电影")
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(films, outfile, ensure_ascii=False, indent=2)
    print(f"数据已保存到: {file_path}")


# # # %%
# # import pandas as pd

# # df = pd.read_csv("./ml-latest-small/links.csv")
# # df = df.dropna(axis=0, how="any")
# # df["tmdbId"] = df["tmdbId"].astype(int)
# # # %%
# # sample(df["tmdbId"].values, "./movielens_latest-small.json")
# # import requests
# # import numpy as np
# # import pandas as pd
# # import json
# # from dotenv import load_dotenv
# # import os
# # from sentence_transformers import SentenceTransformer
# # import tqdm

# # # --- MODIFICATION START ---
# # # We define the proxy here, which acts as our "door key"
# # PROXY = "http://127.0.0.1:10809"  # A common proxy port for cloud environments
# # proxies = {
# #     'http': PROXY,
# #     'https': PROXY,
# # }
# # # --- MODIFICATION END ---

# # load_dotenv()

# # print("Loading SentenceTransformer model...")
# # model = SentenceTransformer("sentence-transformers/sentence-t5-base")
# # print("Model loaded.")

# # def get_director(id):
# #     headers = {
# #         "accept": "application/json",
# #         "Authorization": "Bearer " + os.getenv("TMDB_API_KEY"),
# #     }
# #     url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=en-US"
# #     # --- MODIFICATION --- Added proxies=proxies
# #     response = requests.get(url_credits, headers=headers, proxies=proxies)
# #     data_credits = response.json()

# #     for d in data_credits.get("crew", []):
# #         if d.get("job") == "Director":
# #             return d.get("name")

# # def get_film(id):
# #     url_movie = f"https://api.themoviedb.org/3/movie/{id}?language=en-US"
# #     headers = {
# #         "accept": "application/json",
# #         "Authorization": "Bearer " + os.getenv("TMDB_API_KEY"),
# #     }
# #     # --- MODIFICATION --- Added proxies=proxies
# #     response = requests.get(url_movie, headers=headers, proxies=proxies)
# #     data = response.json()

# #     if response.status_code != 200:
# #         print(f"Warning: Could not fetch data for film ID {id}. Status: {response.status_code}. Response: {data.get('status_message', 'No message')}")
# #         return None

# #     movies_keys_to_remove = [
# #         "backdrop_path", "belongs_to_collection", "homepage", "poster_path",
# #         "production_companies", "production_countries", "spoken_languages",
# #         "status", "tagline", "video",
# #     ]
# #     for k in movies_keys_to_remove:
# #         if k in data:
# #             data.pop(k)

# #     url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=en-US"
# #     # --- MODIFICATION --- Added proxies=proxies
# #     response = requests.get(url_credits, headers=headers, proxies=proxies)
# #     data_credits = response.json()

# #     top_actors = data_credits.get("cast", [])[0:2]
# #     actors_keys_to_remove = [
# #         "adult", "known_for_department", "original_name", "cast_id",
# #         "credit_id", "order", "profile_path",
# #     ]

# #     for t in top_actors:
# #         for k in actors_keys_to_remove:
# #             if k in t:
# #                 t.pop(k)
    
# #     data["actors"] = top_actors

# #     overview_text = data.get("overview", "")
# #     title_text = data.get("title", "")
# #     text_to_encode = overview_text if len(overview_text) > 0 else title_text

# #     embeddings = model.encode(text_to_encode)
# #     data["overview_embedding"] = embeddings.tolist()
# #     data["director"] = get_director(id)
# #     return data

# # def sample(ids, file_path):
# #     films = {}
# #     for id in tqdm.tqdm(ids):
# #         id = int(id)
# #         try:
# #             film_data = get_film(id)
# #             if film_data and film_data.get("overview"):
# #                 films[id] = film_data
# #         except Exception as e:
# #             print(f"An error occurred while processing film ID {id}: {e}")
    
# #     with open(file_path, "w") as outfile:
# #         json.dump(films, outfile)
# #     print(f"\nFinished processing. Saved data for {len(films)} films to {file_path}")


# # print("Reading links.csv...")
# # df = pd.read_csv("./ml-latest-small/links.csv")
# # df = df.dropna(axis=0, how="any")
# # df = df[df['tmdbId'].notna()]
# # df["tmdbId"] = df["tmdbId"].astype(int)
# # print(f"Found {len(df)} movies to process.")

# # sample(df["tmdbId"].values, "./movielens_latest-small.json")



# 主程序执行
if __name__ == "__main__":
    print("=== TMDB电影数据下载器 ===")
    print(f"使用API密钥: {TMDB_API_KEY[:20]}...")
    
    # 检查是否有links.csv文件
    links_file_path = "./ml-latest-small/links.csv"
    if os.path.exists(links_file_path):
        print(f"找到links.csv文件，开始处理...")
        df = pd.read_csv(links_file_path)
        df = df.dropna(subset=["tmdbId"])
        df["tmdbId"] = df["tmdbId"].astype(int)
        movie_ids = df["tmdbId"].values
        print(f"找到 {len(movie_ids)} 部电影需要处理")
        
        # 下载电影数据
        sample(movie_ids, "./movielens_latest-small.json")
    else:
        print("未找到links.csv文件，将下载热门电影数据...")
        
        # 如果没有links.csv，直接下载一些热门电影
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {TMDB_API_KEY}",
        }
        
        # 获取热门电影列表
        url_popular = "https://api.themoviedb.org/3/movie/popular?language=zh-CN&page=1"
        try:
            response = requests.get(url_popular, headers=headers, timeout=10)
            if response.status_code == 200:
                popular_data = response.json()
                movies = popular_data.get("results", [])
                
                # 获取前50部热门电影的ID
                popular_ids = [movie.get("id") for movie in movies[:50] if movie.get("id")]
                print(f"获取到 {len(popular_ids)} 部热门电影")
                
                # 下载这些电影的数据
                sample(popular_ids, "./popular_movies.json")
            else:
                print(f"获取热门电影列表失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"获取热门电影列表时出错: {e}")
    
    print("\n程序执行完成！")


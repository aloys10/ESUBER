# import requests
# import numpy as np
# import pandas as pd
# import json
# from dotenv import load_dotenv

# load_dotenv()
# import os
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("sentence-transformers/sentence-t5-base")


# def get_director(id):
#     headers = {
#         "accept": "application/json",
#         "Authorization": "Bearer " + os.getenv("TMDB_API_KEY"),
#     }
#     url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=en-US"
#     response = requests.get(url_credits, headers=headers)
#     data_credits = response.json()

#     for d in data_credits["crew"]:
#         if d["job"] == "Director":
#             return d["name"]


# def get_film(id):
#     url_movie = f"https://api.themoviedb.org/3/movie/{id}?language=en-US"
#     headers = {
#         "accept": "application/json",
#         "Authorization": "Bearer " + os.getenv("TMDB_API_KEY"),
#     }
#     response = requests.get(url_movie, headers=headers)
#     data = response.json()
#     movies_keys_to_remove = [
#         "backdrop_path",
#         "belongs_to_collection",
#         "homepage",
#         "poster_path",
#         "production_companies",
#         "production_countries",
#         "spoken_languages",
#         "status",
#         "tagline",
#         "video",
#     ]
#     for k in movies_keys_to_remove:
#         if k in data:
#             data.pop(k)

#     url_credits = f"https://api.themoviedb.org/3/movie/{id}/credits?language=en-US"
#     response = requests.get(url_credits, headers=headers)
#     data_credits = response.json()
#     if "cast" not in data_credits:
#         top_actors = []
#     else:
#         top_actors = data_credits["cast"][0:2]
#     actors_keys_to_remove = [
#         "adult",
#         "known_for_department",
#         "original_name",
#         "cast_id",
#         "credit_id",
#         "order",
#         "profile_path",
#     ]

#     for t in top_actors:
#         for k in actors_keys_to_remove:
#             if k in t:
#                 t.pop(k)

#     data["actors"] = top_actors

#     embeddings = model.encode(
#         data["overview"] if len(data["overview"]) > 0 else data["title"]
#     )
#     data["overview_embedding"] = embeddings.tolist()
#     data["director"] = get_director(id)
#     return data


# import tqdm

# # %%
# import json


# def sample(ids, file_path):
#     films = {}
#     counter = 0
#     for id in tqdm.tqdm(ids):
#         id = int(id)
#         film_data = get_film(id)
#         if "overview" in film_data and film_data["overview"] != "":
#             films[id] = film_data
#         else:
#             print(f"Film {id} has no overview")
#         counter += 1
#         if counter % 100 == 0:
#             print(f"{counter} films sampled")
#     with open(file_path, "w") as outfile:
#         json.dump(films, outfile)


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
# ==============================================================================
#  全新、修复后的 tmdb_dowload_data.py
#  请用此代码完全替换你本地的同名文件
# ==============================================================================

import requests
import pandas as pd
import json
from dotenv import load_dotenv
import os
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- 1. 初始化和加载资源 ---

# 加载 .env 文件中的环境变量 (TMDB_API_KEY)
load_dotenv()

# 检查 API Key 是否存在
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in .env file. Please create a .env file with your key.")

print("Loading SentenceTransformer model (this may take a moment)...")
# 加载用于生成文本嵌入的模型
model = SentenceTransformer("sentence-transformers/sentence-t5-base")
print("Model loaded successfully.")


# --- 2. 核心数据获取函数 (已修复，更健壮) ---

def get_film_details(film_id: int, headers: dict) -> dict:
    """
    一个更健壮的函数，用于获取单个电影的详细信息。
    """
    # --- 获取电影基本信息 ---
    url_movie = f"https://api.themoviedb.org/3/movie/{film_id}?language=en-US"
    try:
        response_movie = requests.get(url_movie, headers=headers, timeout=10)
        # 如果请求失败 (例如 404 Not Found), 打印警告并返回 None
        if response_movie.status_code != 200:
            print(f"Warning: Failed to fetch movie data for ID {film_id}. Status: {response_movie.status_code}")
            return None
        data = response_movie.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie data for ID {film_id}: {e}")
        return None

    # --- 获取演职员信息 ---
    url_credits = f"https://api.themoviedb.org/3/movie/{film_id}/credits?language=en-US"
    try:
        response_credits = requests.get(url_credits, headers=headers, timeout=10)
        if response_credits.status_code == 200:
            data_credits = response_credits.json()
        else:
            data_credits = {} # 如果失败，创建一个空字典以避免后续错误
    except requests.exceptions.RequestException as e:
        print(f"Error fetching credits for ID {film_id}: {e}")
        data_credits = {}

    # --- 安全地提取和组织数据 ---

    # 提取导演
    director = "Unknown"
    for crew_member in data_credits.get("crew", []):
        if crew_member.get("job") == "Director":
            director = crew_member.get("name", "Unknown")
            break # 找到第一个导演就停止

    # 提取前两名演员
    top_actors = []
    for actor in data_credits.get("cast", [])[:2]:
        top_actors.append({
            "gender": actor.get("gender"),
            "id": actor.get("id"),
            "name": actor.get("name"),
            "popularity": actor.get("popularity"),
            "character": actor.get("character"),
        })
        
    # 确定用于编码的文本 (优先用简介，否则用标题)
    overview_text = data.get("overview", "")
    title_text = data.get("title", "Unknown Title")
    text_to_encode = overview_text if overview_text else title_text

    # 生成文本嵌入
    try:
        embeddings = model.encode(text_to_encode).tolist()
    except Exception as e:
        print(f"Error encoding text for film ID {film_id}: {e}")
        embeddings = [] # 如果编码失败，给一个空列表

    # 组装最终的电影数据字典
    film_data = {
        "id": data.get("id"),
        "title": title_text,
        "overview": overview_text,
        "genres": data.get("genres", []),
        "release_date": data.get("release_date", ""),
        "vote_average": data.get("vote_average", 0),
        "vote_count": data.get("vote_count", 0),
        "popularity": data.get("popularity", 0),
        "original_language": data.get("original_language", ""),
        "runtime": data.get("runtime", 0),
        "actors": top_actors,
        "director": director,
        "overview_embedding": embeddings,
    }
    
    # 仅当电影有标题和ID时才认为是有效数据
    if film_data["id"] and film_data["title"] != "Unknown Title":
        return film_data
    else:
        print(f"Warning: Skipping film ID {film_id} due to missing essential data (ID or Title).")
        return None

# --- 3. 主下载逻辑 (支持断点续传) ---

def download_all_movies(ids: list, output_path: str):
    """
    下载所有电影数据，并支持断点续传。
    数据以 JSON Lines 格式保存，每行一个电影的 JSON 对象。
    """
    processed_ids = set()
    
    # 检查输出文件是否已存在，如果存在，则读取已处理的电影ID
    if os.path.exists(output_path):
        print(f"Output file '{output_path}' found. Resuming download.")
        with open(output_path, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    # 每一行都是一个JSON对象
                    processed_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    # 如果行是空的或格式不正确，就忽略它
                    pass
        print(f"Found {len(processed_ids)} movies already processed. Skipping them.")

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}",
    }

    # 以追加模式(a)打开文件，这样就不会覆盖掉已有内容
    with open(output_path, "a", encoding='utf-8') as f:
        # 使用tqdm创建进度条
        pbar = tqdm(ids, desc="Downloading movie data")
        for film_id in pbar:
            film_id = int(film_id)
            
            # 如果这个ID已经处理过，就跳过
            if film_id in processed_ids:
                continue

            try:
                film_data = get_film_details(film_id, headers)
                
                # 如果get_film_details返回了有效数据，就写入文件
                if film_data:
                    # 写入一个 JSON 字符串，并加上换行符
                    f.write(json.dumps(film_data) + "\n")
                
                # 短暂休息一下，避免过于频繁地请求API导致被封禁
                time.sleep(0.1)

            except Exception as e:
                # 捕获任何其他可能的意外错误，打印并继续
                print(f"An unexpected error occurred for id {film_id}: {e}")
                time.sleep(5) # 如果发生未知错误，休息5秒再继续

    print(f"\nDownload finished. Data saved to '{output_path}'.")


# --- 4. 脚本执行入口 ---
if __name__ == "__main__":
    links_file_path = "./ml-latest-small/links.csv"
    output_json_path = "./movielens_latest-small.json"

    print(f"Reading movie IDs from '{links_file_path}'...")
    if not os.path.exists(links_file_path):
        raise FileNotFoundError(f"Could not find '{links_file_path}'. Make sure you are in the correct directory.")
        
    df = pd.read_csv(links_file_path)
    # 清理数据：删除没有 tmdbId 的行
    df = df.dropna(subset=["tmdbId"])
    # 确保 tmdbId 是整数类型
    df["tmdbId"] = df["tmdbId"].astype(int)
    
    movie_ids_to_process = df["tmdbId"].values
    
    print(f"Found {len(movie_ids_to_process)} movies to process.")
    
    download_all_movies(movie_ids_to_process, output_json_path)


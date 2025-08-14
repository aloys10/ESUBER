# ==============================================================================
#  脚本一：仅下载元数据（在本地电脑运行）
#  功能：从TMDB API获取电影信息，不进行向量化，生成中间文件。
# ==============================================================================

import requests
import pandas as pd
import json
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

# --- 1. 初始化和加载资源 ---

# 加载 .env 文件中的环境变量 (TMDB_API_KEY)
# 确保在 SUBER 项目根目录下有 .env 文件
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../../.env'))

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found. Ensure a .env file exists in the SUBER project root with your key.")

print("TMDB API Key loaded successfully.")

# --- 2. 核心数据获取函数 (已修复，更健壮) ---

def get_film_details(film_id: int, headers: dict) -> dict:
    """
    获取单个电影的详细信息（无向量化）。
    """
    # --- 获取电影基本信息 ---
    url_movie = f"https://api.themoviedb.org/3/movie/{film_id}?language=en-US"
    try:
        response_movie = requests.get(url_movie, headers=headers, timeout=15)
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
        response_credits = requests.get(url_credits, headers=headers, timeout=15)
        data_credits = response_credits.json() if response_credits.status_code == 200 else {}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching credits for ID {film_id}: {e}")
        data_credits = {}

    # --- 安全地提取和组织数据 ---
    director = "Unknown"
    for crew_member in data_credits.get("crew", []):
        if crew_member.get("job") == "Director":
            director = crew_member.get("name", "Unknown")
            break
            
    top_actors = []
    for actor in data_credits.get("cast", [])[:2]:
        top_actors.append({k: actor.get(k) for k in ["gender", "id", "name", "popularity", "character"]})

    # 组装电影数据字典 (注意：没有 embedding)
    film_data = {
        "id": data.get("id"), "title": data.get("title", "Unknown Title"),
        "overview": data.get("overview", ""), "genres": data.get("genres", []),
        "release_date": data.get("release_date", ""), "vote_average": data.get("vote_average", 0),
        "vote_count": data.get("vote_count", 0), "popularity": data.get("popularity", 0),
        "original_language": data.get("original_language", ""),"runtime": data.get("runtime", 0),
        "actors": top_actors, "director": director,
    }
    
    if film_data["id"] and film_data["title"] != "Unknown Title":
        return film_data
    else:
        print(f"Warning: Skipping film ID {film_id} due to missing essential data.")
        return None

# --- 3. 主下载逻辑 (支持断点续传) ---

def download_all_movies(ids: list, output_path: str):
    processed_ids = set()
    if os.path.exists(output_path):
        print(f"Intermediate file '{output_path}' found. Resuming download.")
        with open(output_path, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError): pass
        print(f"Found {len(processed_ids)} movies already processed. Skipping them.")

    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}

    with open(output_path, "a", encoding='utf-8') as f:
        pbar = tqdm(ids, desc="Downloading movie metadata")
        for film_id in pbar:
            if int(film_id) in processed_ids: continue
            try:
                film_data = get_film_details(int(film_id), headers)
                if film_data:
                    f.write(json.dumps(film_data) + "\n")
                time.sleep(0.05) # Be nice to the API
            except Exception as e:
                print(f"An unexpected error occurred for id {film_id}: {e}")
                time.sleep(5)
    print(f"\nMetadata download finished. Data saved to '{output_path}'.")

# --- 4. 脚本执行入口 ---
if __name__ == "__main__":
    links_file_path = "./ml-latest-small/links.csv"
    intermediate_output_path = "./metadata_without_embeddings.jsonl" # 注意文件名

    print(f"Reading movie IDs from '{links_file_path}'...")
    if not os.path.exists(links_file_path):
        raise FileNotFoundError(f"'{links_file_path}' not found. Make sure you are in '.../datasets/' directory.")
        
    df = pd.read_csv(links_file_path)
    df = df.dropna(subset=["tmdbId"])
    movie_ids_to_process = df["tmdbId"].astype(int).values
    
    print(f"Found {len(movie_ids_to_process)} movies to process.")
    download_all_movies(movie_ids_to_process, intermediate_output_path)

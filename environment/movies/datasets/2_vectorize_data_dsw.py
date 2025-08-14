# ==============================================================================
#  脚本二：仅进行向量化（在 DSW GPU 平台运行）
#  功能：读取中间文件，用GPU进行快速向量化，生成最终的JSON文件。
# ==============================================================================

import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# --- 1. 定义文件路径 ---
# 这个脚本假设它和数据文件在同一个目录下
INPUT_FILE = "./metadata_without_embeddings.jsonl"
FINAL_OUTPUT_FILE = "./movielens_latest-small.json" # 这是项目最终需要的文件名

# --- 2. 加载模型 ---
print("Loading SentenceTransformer model onto GPU...")
# `device='cuda'` 确保模型在 GPU 上运行
model = SentenceTransformer("sentence-transformers/sentence-t5-base", device='cuda')
print("Model loaded successfully.")

# --- 3. 主处理逻辑 ---
def vectorize_data(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found! Please upload it first.")

    # 读取所有电影的元数据
    movies_metadata = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            movies_metadata.append(json.loads(line))
    
    print(f"Read {len(movies_metadata)} movies from intermediate file.")

    # 准备批量处理
    # 从每部电影中提取需要编码的文本
    texts_to_encode = [
        movie.get('overview') or movie.get('title', '') for movie in movies_metadata
    ]

    print("Starting batch vectorization on GPU. This will be fast...")
    # 使用 model.encode 进行批量处理，效率极高
    embeddings = model.encode(
        texts_to_encode,
        batch_size=256, # 可根据你的GPU显存调整，256是个不错的选择
        show_progress_bar=True,
        convert_to_tensor=False # 直接输出numpy array
    )
    print("Vectorization complete.")

    # --- 4. 组装并保存最终数据 ---
    final_data_dict = {}
    for i, movie in enumerate(tqdm(movies_metadata, desc="Assembling final data")):
        # 将生成的 embedding 添加回电影数据中
        movie['overview_embedding'] = embeddings[i].tolist()
        
        # 按照原始脚本的格式，以电影ID为键存入最终的字典
        movie_id = movie.get('id')
        if movie_id:
            final_data_dict[movie_id] = movie

    print(f"Assembled data for {len(final_data_dict)} movies.")
    
    # 写入最终的、项目可用的JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data_dict, f) # 不加indent，文件更小

    print(f"Process complete! Final data saved to '{output_path}'.")
    print("You can now run the ablation studies or training scripts.")

# --- 5. 脚本执行入口 ---
if __name__ == "__main__":
    vectorize_data(INPUT_FILE, FINAL_OUTPUT_FILE)

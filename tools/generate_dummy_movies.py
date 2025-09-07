import json
import os
import random


def main():
    random.seed(42)
    num_items = 20
    embed_dim = 32

    data = {}
    for i in range(1, num_items + 1):
        movie_id = i
        title = f"Dummy Movie {i}"
        overview = f"A synthetic overview for movie {i}. It is about testing the environment pipeline."
        genres = [{"id": 1, "name": "Drama"}] if i % 2 == 0 else [{"id": 2, "name": "Action"}]
        actors = [
            {"gender": 2, "id": 1000 + i, "name": f"Actor {i}", "popularity": 1.0, "character": "Lead"}
        ]
        overview_embedding = [random.uniform(-1.0, 1.0) for _ in range(embed_dim)]
        item = {
            "id": movie_id,
            "title": title,
            "overview": overview,
            "genres": genres,
            "release_date": "2000-01-01",
            "vote_average": float(5 + (i % 5)),
            "vote_count": 10 * i,
            "popularity": float(i),
            "original_language": "en",
            "runtime": 90 + i,
            "actors": actors,
            "director": f"Director {i}",
            "overview_embedding": overview_embedding,
        }
        data[str(movie_id)] = item

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "environment", "movies", "datasets", "movielens_latest-small.json"
    )
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Wrote dummy dataset with {num_items} items to: {out_path}")


if __name__ == "__main__":
    main()



# EmoSim-RL: Emotionally Enhanced User Simulation for Recommender Systems


This repository presents EmoSim-RL, an enhanced version of the original SUBER framework with improved user simulation capabilities.

#### Our paper:



We present `EmoSim-RL`, an enhanced version of the original SUBER framework for recommender systems. Built upon the foundation of [Farama's Gymnasium](https://gymnasium.farama.org/) and compatible with [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/), EmoSim-RL introduces significant improvements to user simulation and recommendation training.

**Key Enhancements:**
- **LLM-powered User Simulator with Chain-of-Thought**: Enhanced user simulation using Large Language Models with logical reasoning chains to better mimic real user decision-making processes
- **Emotion-aware User Profiles**: Integrated emotional indicators in user profiles to capture affective states and preferences
- **Multi-factor Item Retrieval**: Advanced `decay_emotion_3` strategy that considers multiple factors including emotional decay, temporal patterns, and user preferences
- **Realistic User Behavior Modeling**: More accurate simulation of real user interactions to improve recommendation system training

EmoSim-RL provides recommendation environments for movies and books, enabling more effective training of recommendation algorithms that deliver more satisfying user experiences.

## Guide

### Requirements

**Environment Requirements:**
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.1.2
- 8GB+ GPU memory (recommended for local model inference)

**Key Dependencies:**
- `stable-baselines3==2.0.0` - Reinforcement learning algorithms
- `gymnasium==0.28.1` - Environment interface
- `transformers==4.25.1` - Pre-trained models
- `torch==2.1.2` - Deep learning framework
- `openai==1.98.0` - OpenAI API support
- `sentence-transformers==2.2.2` - Text embeddings
- `pandas==2.0.1` - Data processing
- `wandb==0.21.0` - Experiment tracking

**API Key Setup:**
- DeepSeek API: Set environment variable `DEEPSEEK_API_KEY`

Install all dependencies: `pip install -r requirements.txt`





### General environment arguments
The following arguments are available for all environments, and can be passed to ablations and RL training:

```
--llm-model: LLM model to use for the environment. We support DeepSeek models.
--llm-rater: Prompting strategy to query the LLM
--items-retrieval: Items retrieval strategy
--perturbator: Reward perturbator strategy 
--reward-shaping: Reward shaping strategy
--user-dataset: Dataset of users to use
```


### Run ablations
Run the following command to run a config of ablations for the movie environment:
``` 
python3 -m ablations.movies.run_deepseek_experiment
```

Run the following command to run a config of ablations for the book environment:
```
python3 -m ablations.books.run_deepseek_experiment
```

### Run RL training
Run the following command to run the RL training for the movie environment:
```
python3 -m algorithms.movies.CF_train_A2C --llm-model deepseek-chat --llm-rater 0Shot_cotlite_our --items-retrieval decay_emotion_3
```
Additional arguments are available, see `algorithms/movies/CF_train_A2C.py` file for more details.


## Developer Notes
We provide two main recommendation environments: movies and books. Each environment has its own folder in the `environment` folder.
Each environment has its own `configs.py` file, which contains the default configuration and arguments for the environment. 

### Implementing a new environment
To implement a new environment, you need to create a new folder in the `environment` folder, and implement the following classes:
- `Item`: extends the `Item` class in `items.py`
- `User`: extends the `User` class in `users.py`
- `Rater`: extends the `Rater` class in `rater.py`
- `Loader`: extends the `Loader` class in `loader.py`
and the following files:
- `configs.py`: default configuration and arguments for the environment

We refer to the `movies` and `books` environments for examples.

### Project structure

```
ablations/
    books/                          -- Book ablation studies
        datasets/                   -- Ablation datasets for books
        reports/                    -- Results folder (output)
        src/                        -- Source files for the ablation test cases
        run.py                      -- Run Book ablations (*)
        run_gpt.py                  -- Run Book ablations - less sampling (*)
        run_sampling_analysis.sh    -- Run sampling analysis for all reports folders
    movies/                         -- Movie ablation studies
        datasets/                   -- Ablation datasets for movies
        reports/                    -- Results folder (output)
        src/                        -- Source files for the ablation test cases
        run.py                      -- Run Movie ablations (*)
        run_gpt.py                  -- Run Movie ablations - less sampling (*)
        run_sampling_analysis.sh    -- Run sampling analysis for all reports folders
    utils/                          -- Shared utilities for ablation studies
        abstract_study.py           -- Abstract study class
        helper_functions.py         -- Helper functions

algorithms/                         -- RL training code
    books/                          -- RL training and analysis code for books
        CF_train_A2C.py            -- A2C training for book recommendations
    movies/                         -- RL training and analysis code for movies
        CF_train_A2C.py            -- A2C training for movie recommendations
        CF_train_DQN.py            -- DQN training for movie recommendations
        CF_train_PPO.py            -- PPO training for movie recommendations
        CF_train_TRPO.py           -- TRPO training for movie recommendations
        CF_analysis_*.py           -- Analysis scripts for different RL algorithms
        classical_recsys/          -- Classical recommendation system baselines
    wrappers.py                     -- Gymnasium wrappers to use Stable Baselines-3

environment/
    LLM/                            -- LLM model specific components
        guidance/                   -- Guidance model classes, used for user generation
        exllama.py                  -- ExLLAMA model class (for GPTQ inference)
        llm.py                      -- Abstract LLM model class, used by the rater
        rater.py                    -- Rater class (base class), used to rate items based on user characteristics and items features
        openai_api.py              -- OpenAI API integration
        deepseek_api.py            -- DeepSeek API integration
        std_transformers.py        -- Standard transformers integration
    books/                          -- Book specific environment components
        datasets/                   -- Book datasets (Amazon, Goodreads)
        rater_prompts/              -- Rater prompts for book environment
        users_generation/           -- Users generator for book environment
        book.py                     -- Book class for features
        books_loader.py             -- Load books from dataset to Book class
        books_retrieval.py          -- Book retrieval components
        configs.py                  -- Default configuration and arguments for the book environment
    movies/                         -- Movie specific environment components
        datasets/                   -- Movie datasets (TMDB data saved in JSON format)
        rater_prompts/              -- Rater prompts for movie environment
        users_generation/           -- Users generator for movie environment
        movie.py                    -- Movie class for features
        movies_loader.py            -- Load movies from JSON dataset to Movie class
        configs.py                  -- Default configuration and arguments for the movie environment
    users/                          -- User management components
        datasets/                   -- Dataset sampling during users generation
        user.py                     -- User class
        users_loader.py             -- UserLoaders support CSV and list of User objects
        emotion_criteria.py         -- Emotion-based criteria for user modeling
    env.py                          -- Main Gymnasium environment
    item.py                         -- Abstract class for item, all environments need to extend this class
    items_retrieval.py              -- Items retrieval components
    items_selection.py              -- Items selection components
    memory.py                       -- Memory for each user containing item_id and rating for past interactions
    reward_perturbator.py           -- Reward perturbation components
    reward_shaping.py               -- Reward shaping components
```


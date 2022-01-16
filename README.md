Code to run models and simulations for "To Recommend or Not? A Model-Based Comparison of Item-Matching Processes."

This repository contains:
- `models.py`: code to (1) plot results from Theorems 1-4, (2) run simulations with single-dimensional organic + recommender models and finite numbers of items.
- `movielens.py`: code to load MovieLens data. This assumes that the MovieLens 1M dataset is downloaded and located at the paths (PATH_TO_MOVIELENS_SMALL and PATH_TO_MOVIELENS_1M) indicated at the top of this Python file.
- `movielens_experiments.py`: code to (1) run collaborative filtering on the MovieLens data in order to learn realistic user and item embeddings, (2) run simulations of our organic and recommender models using the user and item embeddings learned from MovieLens.
- `movielens_results.ipynb`: a notebook that visualizes results from MovieLens experiments (Figures 5-7 and 2 Appendix Figures).
- `theoretical_results.ipynb`: a notebook that visualizes theoretical results (Figures 2-4).

Other notes:
- Download the MovieLens dataset here: https://grouplens.org/datasets/movielens/1m/

import pandas as pd
import numpy as np
from pathlib import Path






class DataLoader:
    def __init__(self, dataset):
        self.ds_dir = Path("Data/ds100k")
        self.usr_to_idx = {}
        self.idx_to_usr = []
        self.movie_to_idx = {}
        self.idx_to_movie = []
        
        self.user_train = []
        self.movie_train = []
        self.user_test = []
        self.movie_test = []

        # Features
        self.movie_to_features = {}  
        self._load_csv()
        self._process_features()

    def _load_csv(self):
        self.movies = pd.read_csv(self.ds_dir / "movies.csv") 
        self.ratings = pd.read_csv(self.ds_dir / "ratings.csv").drop(columns="timestamp")
        self.tags = pd.read_csv(self.ds_dir / "tags.csv").drop(columns="timestamp")

    def tt_split(self, size=0.8):
        """Train-Test Split"""
        ratings_arr = self.ratings.to_numpy()
        np.random.shuffle(ratings_arr)
        split_point = int(size * len(ratings_arr))

        for row in ratings_arr:
            usr, movie, rating = row
            if usr not in self.usr_to_idx:
                self.usr_to_idx[usr] = len(self.usr_to_idx)
                self.idx_to_usr.append(usr)
            if movie not in self.movie_to_idx:
                self.movie_to_idx[movie] = len(self.movie_to_idx)
                self.idx_to_movie.append(movie)

        num_users = len(self.idx_to_usr)
        num_movies = len(self.idx_to_movie)
        user_train = [[] for _ in range(num_users)]
        movie_train = [[] for _ in range(num_movies)]
        user_test = [[] for _ in range(num_users)]
        movie_test = [[] for _ in range(num_movies)]

        
        for index, (usr, movie, rating) in enumerate(ratings_arr):
            usr_idx = self.usr_to_idx[usr]
            movie_idx = self.movie_to_idx[movie]
            if index < split_point:
                user_train[usr_idx].append((movie_idx, float(rating)))
                movie_train[movie_idx].append((usr_idx, float(rating)))
            else:
                user_test[usr_idx].append((movie_idx, float(rating)))
                movie_test[movie_idx].append((usr_idx, float(rating)))
        return user_train, user_test, movie_train, movie_test

    def _process_features(self):
        """Processes movie features"""
        for _, row in self.movies.iterrows():
            movie_id, title, genres = row['movieId'], row['title'], row['genres']
            self.movie_to_features[movie_id] = {
                'title': title,
                'genres': genres.split('|')
            }


    def get_features(self, movie_id):
        """Fetch features from movie_id."""
        return self.movie_to_features.get(movie_id, None)

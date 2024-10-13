import numpy as np
import matplotlib.pyplot as plt
from utils import Plot

class Sparser(Plot):
    def __init__(self, ratings_data, movie_data):
        self.arr = ratings_data
        self.movie_features_data = movie_data
        self.system_user_to_user_dict = {}
        self.user_to_system_user = []
        self.system_mov_to_mov_dict = {}
        self.mov_to_system_mov = []
        self.data_by_user_train = []
        self.data_by_movie_train = []
        self.data_by_user_test = []
        self.data_by_movie_test = []
        self.split_transform()
        #features
        self.movie_to_system_mov_title_dict = {}
        self.item_to_features = []
        self.features_to_item = []
        self.genres = []
        self.features_transform()
 

    def split_transform(self):
        np.random.shuffle(self.arr)
        split_point = int(0.8 * len(self.arr))

        for index in range(len(self.arr)):
            user_sys = self.arr[index][0]
            movie_sys = self.arr[index][1]

            if user_sys not in self.system_user_to_user_dict:
                self.user_to_system_user.append(user_sys)
                self.system_user_to_user_dict[int(user_sys)] = len(self.system_user_to_user_dict)

            if movie_sys not in self.system_mov_to_mov_dict:
                self.mov_to_system_mov.append(movie_sys)
                self.system_mov_to_mov_dict[int(movie_sys)] = len(self.system_mov_to_mov_dict)

        self.data_by_user_train = [[] for i in range(len(self.user_to_system_user))]
        self.data_by_movie_train = [[] for i in range(len(self.mov_to_system_mov))]
        self.data_by_user_test = [[] for i in range(len(self.user_to_system_user))]
        self.data_by_movie_test = [[] for i in range(len(self.mov_to_system_mov))]

        for index in range(len(self.arr)):
            user_sys = self.arr[index][0]
            movie_sys = self.arr[index][1]
            rating = self.arr[index][2]

            user_index = self.system_user_to_user_dict[user_sys]
            movie_index = self.system_mov_to_mov_dict[movie_sys]

            if index < split_point:
                self.data_by_user_train[user_index].append((movie_index, float(rating)))
                self.data_by_movie_train[movie_index].append((user_index, float(rating)))
            else:
                self.data_by_user_test[user_index].append((movie_index, float(rating)))
                self.data_by_movie_test[movie_index].append((user_index, float(rating)))


    def features_transform(self):
        #every unique userId create []
        self.item_to_features = [[] for mid in np.unique(self.movie_features_data[:, 0])] 
        #every unique feature create []
        self.features_to_item = [[] for _ in np.unique(np.concatenate([entry.split('|') for entry in self.movie_features_data[:, 2]]))] 
        self.genres = np.unique(np.concatenate([entry.split('|') for entry in self.movie_features_data [:, 2]]))
        self.system_feature_to_feature = {genre: i for i, genre in enumerate(self.genres)}

        for mid, title, features in self.movie_features_data:
            movie_indx = self.system_mov_to_mov_dict.get(mid, None)
            if movie_indx is None:
                continue
            for feature in features.split("|"):
                feat_indx = self.system_feature_to_feature[feature]
                if movie_indx not in self.features_to_item[feat_indx]:
                    self.features_to_item[feat_indx].append(movie_indx)
                if feature not in self.item_to_features[movie_indx]:
                    self.item_to_features[movie_indx].append(feature)
            self.movie_to_system_mov_title_dict[movie_indx] = title 

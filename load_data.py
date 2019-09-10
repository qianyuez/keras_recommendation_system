import pandas as pd
import numpy as np


class DataManager():
    def __init__(self, file_path, user_label, object_label, score_label):
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        # randomize data
        df = df.sample(frac=1.0)
        scores = df[score_label].values
        # normalize score value to range [0, 1]
        self.scores = (scores - scores.min()) / (scores.max() - scores.min())

        users = df[user_label].values
        unique_users = set(users)
        self._max_users = len(unique_users)
        self.index_to_user, self.user_to_index = self._get_index_dict(unique_users)
        self.user_indices = self.get_user_indices(users)

        objects = df[object_label].values
        unique_objects = set(objects)
        self._max_objects = len(unique_objects)
        self.index_to_object, self.object_to_index = self._get_index_dict(unique_objects)
        self.object_indices = self.get_object_indices(objects)

    def get_user_indices(self, users):
        return list(map(lambda x: self.user_to_index[x], users))

    def get_object_indices(self, objects):
        return list(map(lambda x: self.object_to_index[x], objects))

    def get_max_users(self):
        return self._max_users

    def get_max_objects(self):
        return self._max_objects

    def get_train_test_data(self, train_split):
        size = len(self.scores)
        train_split = np.clip(train_split, 0, 1)
        index = int(size * train_split)
        train_size = index
        test_size = size - index
        train_user_indices = np.array(self.user_indices[:index]).reshape((train_size, 1))
        train_object_indices = np.array(self.object_indices[:index]).reshape((train_size, 1))
        train_scores = np.array(self.scores[:index])

        test_user_indices = np.array(self.user_indices[index:]).reshape((test_size, 1))
        test_object_indices = np.array(self.object_indices[index:]).reshape((test_size, 1))
        test_scores = np.array(self.scores[index:])

        return [train_user_indices, train_object_indices], train_scores, [test_user_indices, test_object_indices], test_scores

    def _get_index_dict(self, values):
        index_to_value = {}
        value_to_index = {}
        for index, value in enumerate(values):
            index_to_value[index] = value
            value_to_index[value] = index
        return index_to_value, value_to_index

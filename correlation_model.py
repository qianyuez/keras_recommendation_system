from keras import Model
from keras.layers import Input, Dot, Concatenate, Dense, Embedding, Flatten, Lambda
import numpy as np
import matplotlib.pyplot as plt


class CorrelationModel():
    def __init__(self, max_users, max_objects, dim=256, learn_similarity=True):
        self.max_users = max_users
        self.max_objects = max_objects
        self.dim = dim
        self.learn_similarity = learn_similarity
        if learn_similarity:
            self.score_model, self.user_similarity_model, self.object_similarity_model =\
                self._build_similarity_model()
        else:
            self.score_model = self._build_value_model()
        self.score_model.summary()
        self.score_model.compile(optimizer='adam', loss='mae')

    def fit(self, x, y, batch_size, epochs, validation_split, plot=True):
        history = self.score_model.fit(x, y, batch_size, epochs, validation_split=validation_split, verbose=2)
        if plot:
            self._plot_training_history(history.history)

    def evaluate(self, x, y):
        return self.score_model.evaluate(x, y)

    def save_model(self, path):
        self.score_model.save_weights(path)

    def load_model(self, path):
        self.score_model.load_weights(path)

    def get_similar_users(self, user_index, index_to_user, top_n=1):
        if self.learn_similarity:
            return self._get_top_n_indices(self.user_similarity_model, user_index, self.max_users, index_to_user, top_n)
        return []

    def get_similar_objects(self, object_index, index_to_object, top_n=1):
        if self.learn_similarity:
            return self._get_top_n_indices(self.object_similarity_model, object_index, self.max_objects, index_to_object, top_n)
        return []

    def _get_top_n_indices(self, model, index, size, index_dict, top_n):
        indices_similarity = []
        for compared_index in range(size):
            if index == compared_index:
                continue
            similarity = model.predict(
                [np.array([[index]]), np.array([[compared_index]])])[0][0]
            indices_similarity.append((index_dict[compared_index], similarity))
        indices_similarity.sort(key=lambda x:x[1], reverse=True)
        return indices_similarity[:min(top_n, size - 1)]

    def _build_similarity_model(self):
        user_input = Input(shape=(1,))
        user_embedding = Embedding(self.max_users, self.dim, input_length=1)
        user_vec = user_embedding(user_input)
        user_vec = Flatten()(user_vec)

        object_input = Input(shape=(1,))
        object_embedding = Embedding(self.max_objects, self.dim, input_length=1)
        object_vec = object_embedding(object_input)
        object_vec = Flatten()(object_vec)

        cosine_proximity = Dot(-1, normalize=True)([user_vec, object_vec])
        score = Lambda(lambda x: (x + 1) * 0.5)(cosine_proximity)
        score_model = Model(inputs=[user_input, object_input], outputs=score)

        compared_user_input = Input(shape=(1,))
        compared_user_vec = user_embedding(compared_user_input)
        compared_user_vec = Flatten()(compared_user_vec)
        user_cosine_proximity = Dot(-1, normalize=True)([user_vec, compared_user_vec])
        user_similarity_model = Model([user_input, compared_user_input], user_cosine_proximity)

        compared_object_input = Input(shape=(1,))
        compared_object_vec = user_embedding(compared_object_input)
        compared_object_vec = Flatten()(compared_object_vec)
        object_cosine_proximity = Dot(-1, normalize=True)([object_vec, compared_object_vec])
        object_similarity_model = Model([object_input, compared_object_input], object_cosine_proximity)

        return score_model, user_similarity_model, object_similarity_model

    def _build_value_model(self):
        user_input = Input(shape=(1,))
        user_embedding = Embedding(self.max_users, self.dim, input_length=1)
        user_vec = user_embedding(user_input)
        user_vec = Flatten()(user_vec)

        object_input = Input(shape=(1,))
        object_embedding = Embedding(self.max_objects, self.dim, input_length=1)
        object_vec = object_embedding(object_input)
        object_vec = Flatten()(object_vec)

        vec = Concatenate()([user_vec, object_vec])
        d = Dense(256, activation='relu')(vec)
        score = Dense(1, activation='sigmoid')(d)

        return Model(inputs=[user_input, object_input], outputs=score)

    def _plot_training_history(self, history_dict):
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label="Training loss")
        plt.plot(epochs, val_loss, 'b', label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()

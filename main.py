from correlation_model import CorrelationModel
from load_data import DataManager


def main():
    csv_file_path = './data/ratings.csv'
    value_model_path = './model/value_model.h5'
    similarity_model_path = './model/similarity_model.h5'
    user_label = 'user_id'
    object_label = 'book_id'
    score_label = 'rating'
    epochs = 10
    batch_size = 64
    validation_split = 0
    train_split = 0.85

    data = DataManager(csv_file_path, user_label, object_label, score_label)
    max_users = data.get_max_users()
    max_objects = data.get_max_objects()

    value_model = CorrelationModel(max_users=max_users,
                                   max_objects=max_objects,
                                   learn_similarity=False)

    similarity_model = CorrelationModel(max_users=max_users,
                                        max_objects=max_objects,
                                        learn_similarity=True)

    train_x, train_y, test_x, test_y = data.get_train_test_data(train_split)

    value_model.fit(train_x, train_y, batch_size, epochs, validation_split, plot=False)
    value_model.save_model(value_model_path)

    similarity_model.fit(train_x, train_y, batch_size, epochs, validation_split, plot=False)
    similarity_model.save_model(similarity_model_path)

    value_model_loss = value_model.evaluate(test_x, test_y)
    similarity_model_loss = similarity_model.evaluate(test_x, test_y)
    print('value model test loss: {}'.format(value_model_loss))
    print('similarity model test loss: {}'.format(similarity_model_loss))

    user_id = 1
    object_id = 1
    top_n = 10
    print('The {} most similar users for user {}:'.format(top_n, user_id))
    print(similarity_model.get_similar_users(data.get_user_indices([user_id])[0], data.index_to_user, top_n=top_n))

    print('The {} most similar objects for object {}:'.format(top_n, object_id))
    print(similarity_model.get_similar_objects(data.get_object_indices([object_id])[0], data.index_to_object, top_n=top_n))


if __name__ == '__main__':
    main()

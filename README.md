# keras_recommendation_system
Using keras to train recommendation model and learn the similarity of users or items.


# Data
goodbooks-10k

To download:

https://github.com/zygmuntz/goodbooks-10k


# Description
This recommendation model is working like word2vec algorithm, trying to learn the correlation between users and items. The model uses embedding layer in keras to transform user id and item id to 1d vector, and calculate their cosine proximity(normalized to \[0, 1]) as similarity output. When trainning, every pair of user and item index is input and the corresponding rating is normalized to range \[0, 1] as true similarity output.

In this project, there are two kinds of model. Similarity model just uses embendding layer and calculate cosine similarity, so the embedding layers also learn the similarity between users or items. Another model only focuses on output and uses a more complex network to predict rating.

# Results
Dataset is splitted to 85% train data and 15% test data. The test mean absolute error is 0.153 for the simpler model and the other is 0.15.


import tensorflow as tf


training_samples_file_path = tf.keras.utils.get_file("trainingTagSamples.csv",
                                                     "file:///Users/liling/code/SparrowRecSysV2/src/main"
                                                     "/resources/webroot/sampledata/trainingTagSamples.csv")
test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                 "file:///Users/liling/icloud/Desktop/testSamples.csv")
# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset


# split as test dataset and training dataset
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# Config
RECENT_TAGS = 3  # userRatedTag{1-3}
EMBEDDING_SIZE = 10

# define input for keras model
inputs = {
    'ctime': tf.keras.layers.Input(name='ctime', shape=(), dtype='string'),

    'entityid': tf.keras.layers.Input(name='entityid', shape=(), dtype='int32'),
    'userid': tf.keras.layers.Input(name='userid', shape=(), dtype='string'),
    'cate1': tf.keras.layers.Input(name='cate1', shape=(), dtype='string'),
    'cate2': tf.keras.layers.Input(name='cate2', shape=(), dtype='int32'),
    'cate3': tf.keras.layers.Input(name='cate3', shape=(), dtype='string'),

    'actiontype': tf.keras.layers.Input(name='actiontype', shape=(), dtype='string'),
    'usertag1': tf.keras.layers.Input(name='usertag1', shape=(), dtype='int32'),
    'usertag2': tf.keras.layers.Input(name='usertag2', shape=(), dtype='int32'),
    'usertag3': tf.keras.layers.Input(name='usertag3', shape=(), dtype='int32'),
}

tag_id = tf.feature_column.categorical_column_with_identity(key='entityid', num_buckets=160000)
tag_emb_id = tf.feature_column.embedding_column(tag_id, 10)

# cate2 embedding feature     二级类目共
cate2_id = tf.feature_column.categorical_column_with_identity(key='cate2', num_buckets=403648)
cate2_emb_id = tf.feature_column.embedding_column(cate2_id, 10)


# genre features vocabulary
# genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
#                'Sci-Fi', 'Drama', 'Thriller',
#                'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
# user genre embedding feature
user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
                                                                           vocabulary_list=genre_vocab)
user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, EMBEDDING_SIZE)
# item genre embedding feature
item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
                                                                           vocabulary_list=genre_vocab)
item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, EMBEDDING_SIZE)

# user behaviors
recent_rate_col = [
    tf.feature_column.numeric_column(key='usertag1', default_value=0),
    tf.feature_column.numeric_column(key='usertag2', default_value=0),
    tf.feature_column.numeric_column(key='usertag3', default_value=0),
 ]

# user profile
user_profile = [

    cate2_emb_id,
]

# context features
# context_features = [
#     item_genre_emb_col,
#     tf.feature_column.numeric_column('releaseYear'),
#     tf.feature_column.numeric_column('movieRatingCount'),
#     tf.feature_column.numeric_column('movieAvgRating'),
#     tf.feature_column.numeric_column('movieRatingStddev'),
# ]

candidate_emb_layer = tf.keras.layers.DenseFeatures([movie_emb_col])(inputs)
user_behaviors_layer = tf.keras.layers.DenseFeatures(recent_rate_col)(inputs)
user_profile_layer = tf.keras.layers.DenseFeatures(user_profile)(inputs)
context_features_layer = tf.keras.layers.DenseFeatures(context_features)(inputs)

# Activation Unit
user_behaviors_emb_layer = tf.keras.layers.Embedding(input_dim=1001,
                                                     output_dim=EMBEDDING_SIZE,
                                                     mask_zero=True)(user_behaviors_layer)  # mask zero
repeated_candidate_emb_layer = tf.keras.layers.RepeatVector(RECENT_MOVIES)(candidate_emb_layer)

activation_sub_layer = tf.keras.layers.Subtract()([user_behaviors_emb_layer,
                                                   repeated_candidate_emb_layer])  # element-wise sub
activation_product_layer = tf.keras.layers.Multiply()([user_behaviors_emb_layer,
                                                       repeated_candidate_emb_layer])  # element-wise product

activation_all = tf.keras.layers.concatenate([activation_sub_layer, user_behaviors_emb_layer,
                                              repeated_candidate_emb_layer, activation_product_layer], axis=-1)

activation_unit = tf.keras.layers.Dense(32)(activation_all)
activation_unit = tf.keras.layers.PReLU()(activation_unit)
activation_unit = tf.keras.layers.Dense(1, activation='sigmoid')(activation_unit)
activation_unit = tf.keras.layers.Flatten()(activation_unit)  # 将模型展平
activation_unit = tf.keras.layers.RepeatVector(EMBEDDING_SIZE)(activation_unit)
activation_unit = tf.keras.layers.Permute((2, 1))(activation_unit)  # 置换输入的第1个和第2个维度
activation_unit = tf.keras.layers.Multiply()([user_behaviors_emb_layer, activation_unit])

# sum pooling
user_behaviors_pooled_layers = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(activation_unit)

# fc layer
concat_layer = tf.keras.layers.concatenate([user_profile_layer, user_behaviors_pooled_layers,
                                            candidate_emb_layer, context_features_layer])

output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)

model = tf.keras.Model(inputs, output_layer)

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')]
)

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('Test Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
" | Actual rating label: ",
("Good Rating" if bool(goodRating) else "Bad Rating"))

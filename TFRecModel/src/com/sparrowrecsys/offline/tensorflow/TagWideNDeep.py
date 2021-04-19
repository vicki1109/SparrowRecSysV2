import tensorflow as tf

training_samples_file_path = tf.keras.utils.get_file("trainingTagSamples.csv",
                                                     "file:///Users/liling/code/SparrowRecSysV2/src/main"
                                                     "/resources/webroot/sampledata/trainingTagSamples.csv")
test_samples_file_path = tf.keras.utils.get_file("tagtestlabel.csv",
                                                 "file:///Users/liling/code/SparrowRecSysV2/src/main"
                                                 "/resources/webroot/sampledata/tagtestlabel.csv")

# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True
    )
    return dataset

# split as test dataset and training dataset
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# genre features vocalulary
# genre_vocab = ['File-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
#                'Sci-Fi', 'Drama', 'Thriller',
#                'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
#
# GENRE_FEATURES = {
#     'userGenre1': genre_vocab,
#     'userGenre2': genre_vocab,
#     'userGenre3': genre_vocab,
#     'userGenre4': genre_vocab,
#     'userGenre5': genre_vocab,
#     'movieGenre1': genre_vocab,
#     'movieGenre2': genre_vocab,
#     'movieGenre3': genre_vocab
# }
#
# # all categorical features
categorical_columns = []
# for feature, vocab in GENRE_FEATURES.items():
#     cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
#         key=feature, vocabulary_list=vocab)
#     emb_col = tf.feature_column.embedding_column(cat_col, 10)
#     categorical_columns.append(emb_col)
# tag id embedding feature       非top19标签
tag_id = tf.feature_column.categorical_column_with_identity(key='entityid', num_buckets=160000)
tag_emb_id = tf.feature_column.embedding_column(tag_id, 10)
categorical_columns.append(tag_emb_id)

# cate2 embedding feature     二级类目共
cate2_id = tf.feature_column.categorical_column_with_identity(key='cate2', num_buckets=403648)
cate2_emb_id = tf.feature_column.embedding_column(cate2_id, 10)
categorical_columns.append(cate2_emb_id)


# user id embedding feature     近三个月有行为用户
# user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
# user_emb_col = tf.feature_column.embedding_column(user_col, 10)
# categorical_columns.append(user_emb_col)

# all numerical features
# numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
#                      tf.feature_column.numeric_column('movieRatingCount'),
#                      tf.feature_column.numeric_column('movieAvgRating'),
#                      tf.feature_column.numeric_column('movieRatingStddev'),
#                      tf.feature_column.numeric_column('userRatingCount'),
#                      tf.feature_column.numeric_column('userAvgRating'),
#                      tf.feature_column.numeric_column('userRatingStddev')]

# -----------------same as MLP-------------------------

# cross feature between current tag and user historical tag
# “用户已点击标签”和“当前标签”组成的交叉特征
rated_tag = tf.feature_column.categorical_column_with_identity(key='usertag1', num_buckets=30001)
crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([tag_id, rated_tag], 10000))

# define input for keras model
inputs = {
    # 'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
    # 'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
    # 'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
    # 'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
    # 'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
    # 'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
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
    # 'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
    # 'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
    # 'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
    # 'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
}

# wide and deep model architecture
# deep part for all input features
deep = tf.keras.layers.DenseFeatures(categorical_columns)(inputs)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
# wide part for cross feature
wide = tf.keras.layers.DenseFeatures(crossed_feature)(inputs)
both = tf.keras.layers.concatenate([deep, wide])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
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

# print same predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
" | Actual rating label: ",
("Good Rating" if bool(goodRating) else "Bad Rating"))






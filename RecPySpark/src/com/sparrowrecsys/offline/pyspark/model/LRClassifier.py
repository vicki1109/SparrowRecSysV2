# import tensorflow as tf
#
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# # 定义模型结构和模型参数
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     #输出层采用softmax模型，处理多分类问题
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# #定义模型的优化方法(adam), 损失函数（sparse_categorical_crossentropy）和评估指标(accuracy)
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test, verbose=2)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import collections
import tensorflow as tf

defaults = collections.OrderedDict([
    ("label", [0]),
    ("infoId2", [""]),
    ("infoCityId1", [""]),
    ("infoCateId1", [""]),
    ("infoCateId2", [""]),
    ("deviceType", [""]),
    ("postDayScore", [0.0]),
    ("createDayScore", [0.0]),
    ("businessScore", [0.0]),
    ("cvrScore", [0.0]),
    ("listCtrScore", [0.0]),
    ("detailCtrScore", [0.0]),
    ("callUvScore", [0.0]),
    ("ratingScore", [0.0]),
    ("business", [""]),
    ("contentScore", [0.0]),
    ("picScore", [0.0]),
    ("areaScore", [0.0]),
    ("hasSearchText", [""]),
    ("searchTextTermCount", [0.0]),
    ("overlapTermCount", [0.0]),
    ("overlapRatioBySearchText", [0.0]),
    ("overlapRatioByInfoTitle", [0.0]),
    ("highCommentScore", [0.0]),
    ("midCommentScore", [0.0]),
    ("lowCommentScore", [0.0]),
    ("allCommentScore", [0.0]),
    ("bizRank", [0])
])

def dataset(datapath_train, datapath_test, y_name="label"):
    def decode_line(line):
        items = tf.decode_csv(line, list(defaults.values()), field_delim="\t")
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)
        label = features_dict.pop(y_name)
        return features_dict, label

    base_dataset = tf.data.TextLineDataset(datapath_train)
    train = (base_dataset.map(decode_line).repeat(1))

    base_dataset = tf.data.TextLineDataset(datapath_test)
    test = (base_dataset.map(decode_line).repeat(1))

    return train, test

def classify(datapath_train, datapath_test, steps, learn_rate, ):
    (train, test) = dataset(datapath_train, datapath_test)

    def input_train():
        return (train.shuffle(150000).batch(128).make_one_shot_iterator().get_next())
    def input_test():
        return (test.shuffle(150000).batch(128).make_one_shot_iterator().get_next())

    feature_column_infoCityId1 = tf.feature_column.categorical_column_with_hash_bucket(key="infoCityId1", hash_bucket_size=700)
    feature_column_infoCateId1 = tf.feature_column.categorical_column_with_hash_bucket(key="infoCateId1", hash_bucket_size=20)
    feature_column_infoCateId2 = tf.feature_column.categorical_column_with_hash_bucket(key="infoCateId2", hash_bucket_size=280)

    feature_column_deviceType = tf.feature_column.categorical_column_with_vocabulary_list(key="deviceType", vocabulary_list=["1", "2", "3"])
    feature_column_business = tf.feature_column.categorical_column_with_vocabulary_list(key="business", vocabulary_list=["0", "1"])
    feature_column_hasSearchText = tf.feature_column.categorical_column_with_vocabulary_list(key="hasSearchText", vocabulary_list=["0", "1"])

    feature_columns = [
        tf.feature_column.embedding_column(feature_column_infoCityId1, 200),
        tf.feature_column.embedding_column(feature_column_infoCateId1, 200),
        tf.feature_column.embedding_column(feature_column_infoCateId2, 200),

        tf.feature_column.numeric_column(key="postDayScore"),
        tf.feature_column.numeric_column(key="createDayScore"),
        tf.feature_column.numeric_column(key="businessScore"),
        tf.feature_column.numeric_column(key="cvrScore"),
        tf.feature_column.numeric_column(key="listCtrScore"),
        tf.feature_column.numeric_column(key="detailCtrScore"),
        tf.feature_column.numeric_column(key="callUvScore"),
        tf.feature_column.numeric_column(key="ratingScore"),

        tf.feature_column.numeric_column(key="contentScore"),
        tf.feature_column.numeric_column(key="picScore"),
        tf.feature_column.numeric_column(key="areaScore"),

        tf.feature_column.numeric_column(key="highCommentScore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="midCommentScore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="lowCommentScore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="allCommentScore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="bizRank", dtype=tf.int64),

        tf.feature_column.indicator_column(feature_column_deviceType),
        tf.feature_column.indicator_column(feature_column_business),
        tf.feature_column.indicator_column(feature_column_hasSearchText)
    ]

    model = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=learn_rate,
            l1_regularization_strength=0.001
        )
    )

    model.train(input_fn=input_train, steps=steps)
    eval_result = model.evaluate(input_fn=input_test)
    print("------------------------------------------------------")
    print("steps: " + str(steps) + " model evaluate result: " + str(eval_result))
    print("------------------------------------------------------")

    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    modelPath = '/workspace/lbg/firstcate/v1/'
    model.export_savedmodel(modelPath, serving_input_receiver_fn)

if __name__ == "__main__":
    datapath_train = sys.argv[1]    # lbg_lj_dajia_sort_guesslike_v1_train
    datapath_test = sys.argv[2]     # lbg_lj_dajia_sort_guesslike_v1_test
    steps = int(sys.argv[3])        # 1000
    learn_rate = float(sys.argv[4]) # 0.03
    classify(datapath_train, datapath_test, steps, learn_rate)
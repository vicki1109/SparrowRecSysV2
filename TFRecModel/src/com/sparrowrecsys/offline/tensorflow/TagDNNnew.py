from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

import tensorflow as tf

defaults = collections.OrderedDict([
    ("label", [0]),
    ("tagid", [""]),
    ("showcityid", [""]),
    ("showcate1", [""]),
    ("showcate2", [""]),
    ("showcate3", [""]),
    ("clicktag", [""]),
    ("searchkey", [""]),
    ("tagcate1", [""]),
    ("tagcate2", [""]),
    ("tagcate3", [""]),
    ("has2dispcate", [0]),
    ("has3dispcate", [0]),
    ("cpcandiscore", [0.0]),
    ("ctrcandiscore", [0.0]),
    ("querycandiscore", [0.0]),
    ("weightcandiscore", [0.0]),
    ("contained_60", [0]),
    ("contained_40", [0]),
    ("isdefaultcandi", [0]),
    ("isnewtagcandi", [0]),
    ("UserClickTagsHistory", [0]),
    ("TagUserActionsCf", [0.0]),
    ("TagUserActionsCfRnn", [0.0]),
    ("User24HActions", [0.0]),
    ("User24HActionCF", [0.0]),
    ("TagW2VScore", [0.0]),
    ("SearchW2VScore", [0.0])
])

def dataset(datapath_train, datapath_test, y_name="label"):
    def decode_line(line):
        """Convert a csv line into a (features_dict,label) pair."""
        # Decode the line to a tuple of items based on the types of
        # csv_header.values().
        items = tf.decode_csv(line, list(defaults.values()), field_delim="\t")

        # Convert the keys and items to a dict.
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)

        # Remove the label from the features_dict
        label = features_dict.pop(y_name)

        return features_dict, label

    base_dataset = tf.data.TextLineDataset(datapath_train)
    train = (base_dataset.map(decode_line).repeat(1))

    base_dataset = tf.data.TextLineDataset(datapath_test)
    test = (base_dataset.map(decode_line).repeat(1))

    return train, test


def classify(datapath_train, datapath_test, steps):
    (train, test) = dataset(datapath_train, datapath_test)

    def input_train():
        return (train.shuffle(100000).batch(128).make_one_shot_iterator().get_next())

    def input_test():
        return (test.shuffle(100000).batch(128).make_one_shot_iterator().get_next())

    showcityid = tf.feature_column.categorical_column_with_hash_bucket(key="showcityid", hash_bucket_size=300)
    showcate1 = tf.feature_column.categorical_column_with_hash_bucket(key="showcate1", hash_bucket_size=100)
    showcate2 = tf.feature_column.categorical_column_with_hash_bucket(key="showcate2", hash_bucket_size=200)
    showcate3 = tf.feature_column.categorical_column_with_hash_bucket(key="showcate3", hash_bucket_size=1000)

    clicktag = tf.feature_column.categorical_column_with_hash_bucket(key="clicktag", hash_bucket_size=2000)
    searchkey = tf.feature_column.categorical_column_with_hash_bucket(key="searchkey", hash_bucket_size=20000)

    tagcate1 = tf.feature_column.categorical_column_with_hash_bucket(key="tagcate1", hash_bucket_size=100)
    tagcate2 = tf.feature_column.categorical_column_with_hash_bucket(key="tagcate2", hash_bucket_size=200)
    tagcate3 = tf.feature_column.categorical_column_with_hash_bucket(key="tagcate3", hash_bucket_size=1000)

    emb_showcityid = tf.feature_column.embedding_column(showcityid, 50)
    emb_showcate1 = tf.feature_column.embedding_column(showcate1, 25)
    emb_showcate2 = tf.feature_column.embedding_column(showcate2, 50)
    emb_showcate3 = tf.feature_column.embedding_column(showcate3, 500)

    emb_clicktag = tf.feature_column.embedding_column(clicktag, 250)
    emb_searchkey = tf.feature_column.embedding_column(searchkey, 2500)

    emb_tagcate1 = tf.feature_column.embedding_column(tagcate1, 25)
    emb_tagcate2 = tf.feature_column.embedding_column(tagcate2, 50)
    emb_tagcate3 = tf.feature_column.embedding_column(tagcate3, 500)

    feature_columns = [

        emb_showcityid,
        emb_showcate1,
        emb_showcate2,
        emb_showcate3,

        emb_clicktag,
        emb_searchkey,
        emb_tagcate1,
        emb_tagcate2,
        emb_tagcate3,

        tf.feature_column.numeric_column(key="cpcandiscore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="ctrcandiscore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="querycandiscore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="weightcandiscore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="TagUserActionsCf", dtype=tf.float32),
        tf.feature_column.numeric_column(key="TagUserActionsCfRnn", dtype=tf.float32),
        tf.feature_column.numeric_column(key="User24HActions", dtype=tf.float32),
        tf.feature_column.numeric_column(key="User24HActionCF", dtype=tf.float32),
        tf.feature_column.numeric_column(key="TagW2VScore", dtype=tf.float32),
        tf.feature_column.numeric_column(key="SearchW2VScore", dtype=tf.float32),

        tf.feature_column.numeric_column(key="contained_60"),
        tf.feature_column.numeric_column(key="contained_40"),
        tf.feature_column.numeric_column(key="UserClickTagsHistory"),
        tf.feature_column.numeric_column(key="has2dispcate"),
        tf.feature_column.numeric_column(key="has3dispcate"),
        tf.feature_column.numeric_column(key="isdefaultcandi"),
        tf.feature_column.numeric_column(key="isnewtagcandi")
    ]

    # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
    # defined above as input.
    model = tf.estimator.DNNClassifier(
        hidden_units=[64, 8],
        feature_columns=feature_columns,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.02,
            l1_regularization_strength=0.01
        )
    )

    model.train(input_fn=input_train, steps=steps)
    eval_result = model.evaluate(input_fn=input_test)
    print("------------- steps:" + str(steps) + " -------------")
    print(eval_result)


if __name__ == "__main__":

    datapath_train = sys.argv[1]
    datapath_predict = sys.argv[2]
    modelpath = sys.argv[3]

    for step in range(560, 900, 40):
        print("step: " + str(step))
        classify(datapath_train, datapath_predict, step)
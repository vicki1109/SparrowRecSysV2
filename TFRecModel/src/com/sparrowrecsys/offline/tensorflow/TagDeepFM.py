import tensorflow as tf

training_samples_file_path = tf.keras.utils.get_file("tagtrainlabel.csv",
                                                     "file:///Users/liling/code/SparrowRecSysV2/src/main"
                                                     "/resources/webroot/sampledata/tagtrainlabel.csv")
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

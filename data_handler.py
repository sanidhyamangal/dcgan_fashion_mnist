"""
Author: Sanidhya Mangal
github: sanidhyamangal
"""
import glob  # for glob based operations
import pathlib  # for path based ops

import pandas as pd  # for data frame based ops
import tensorflow as tf  # for deep learning and data processing

train_path = "/home/sanidhya/Dataset/fashion-mnist_train.csv"
# test_path = "/home/sanidhya/Dataset/fashion-mnist_test.csv"


def data_loader_csv(df_path: str, batch_size=64, shuffle=True):

    data = pd.read_csv(df_path)

    data_set_labels, data_set_data = tf.convert_to_tensor(
        data.iloc[:, 0],
        tf.float32), tf.convert_to_tensor(data.iloc[:, 1:], tf.float32)

    # free up the memory for train set test_set
    del data

    # create a train and test dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (data_set_data, data_set_labels)).shuffle(60000).batch(64)

    del data_set_data
    del data_set_labels

    # return dataset
    return dataset


def data_loader_csv_unsupervisied(df_path: str, batch_size=64, shuffle=True):

    data = pd.read_csv(df_path)

    data_set_data = tf.convert_to_tensor(data.iloc[:, 1:], tf.float64)

    # free up the memory for train set test_set
    del data

    # create a train and test dataset
    dataset = tf.data.Dataset.from_tensor_slices((data_set_data)).shuffle(60000).batch(batch_size).map(lambda x: tf.reshape((x-127.5)/127.5, shape=[-1, 28, 28, 1]))

    del data_set_data

    # return dataset
    return dataset

train_path = "/home/sanidhya/Dataset/fashion-mnist_train.csv"

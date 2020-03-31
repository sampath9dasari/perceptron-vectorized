import numpy as np


def make_batches(data, batch_size, axis=0):
    """
    Function to make batches, given an numpy array, batch size and the axis for splitting

    :param data: Array of input data to be split into batches
    :param axis: An integer value giving the Axis to split the data on
    :param batch_size: An integer value giving the size of each batch
    :return: An array containing arrays of input data, each array being a different batch
    """

    num_batches = data.shape[axis] // batch_size
    residue = data.shape[axis] % batch_size
    batches = np.split(data[:data.shape[axis] - residue], num_batches)
    if residue != 0:
        batches = batches + [data[data.shape[axis] - residue:]]

    return batches


def accuracy(pred_labels, actual_labels):
    """
    Calculates the accuracy between the predicted label and actual labels.

    :param pred_labels: Array of predicted output labels of data set.
    :param actual_labels: Array of actual output labels of data set.
    :return: A float value giving the accuracy.
    """

    return np.array([pred_labels == actual_labels]).mean()
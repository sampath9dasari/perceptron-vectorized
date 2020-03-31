import numpy as np


def make_batches(data, axis, batch_size):
    """
    Function to make batches, given an numpy array, batch size and the axis for splitting

    :param data: Array of input data to be split into batches
    :param axis: An integer value giving the Axis to split the data on
    :param batch_size: An integer value giving the size of each batch
    :return: An array containing arrays of input data, each array being a different batch
    """
    num_batches = data.shape[axis] // batch_size
    residue = data.shape[0] % batch_size
    batches = np.split(data[:data.shape[axis] - residue], num_batches)
    if residue != 0:
        batches = batches + [data[data.shape[axis] - residue:]]

    return batches

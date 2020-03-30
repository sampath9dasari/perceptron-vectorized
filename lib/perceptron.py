import numpy as np


class perceptron():

    def __init__(self, epoch_threshold=100, l_rate=1, batch_size=1):
        self.weights = np.ones(4)
        self.epoch_threshold = epoch_threshold
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.axis = 0
        pass


    ### Function to make batches, given an numpy array, batch size and the axis for splitting
    def make_batches(self, data, axis, batch_size = None):
        if batch_size == None :
            batch_size = self.batch_size

        num_batches = data.shape[axis] // batch_size
        residue = data.shape[0] % batch_size
        batches = np.split(data[:data.shape[axis] - residue], num_batches)
        if residue != 0: batches = batches + [data[data.shape[axis] - residue:]]
        return (batches)


    ### Function to compute the product between weights, feature vector and true label
    def compute(self, x, y, weights):
        y_compute = np.dot(x, weights) * y
        return (y_compute)


    ### Function to train the perceptron model and update the weights
    def train_perceptron(self, train, labels, weights, l_rate=1, batch_size=1, epoch_threshold=100):
        epoch = 0
        train_batches = self.make_batches(train, batch_size, 0)
        label_batches = self.make_batches(labels, batch_size, 0)
        print("Number of batches formed: ", len(train_batches))
        print()
        while (True):
            total_miss_class = 0
            for tbatch, lbatch, batch_num in zip(train_batches, label_batches, range(len(train_batches) + 1)):
                computed_value = compute(tbatch, lbatch, weights)
                total_miss_class += len(tbatch[computed_value < 0])

                ###         Filtering for the missclassified features using computed_value
                weights_update = l_rate * np.dot(lbatch[computed_value < 0], tbatch[computed_value < 0])
                weights = weights + weights_update
            #             print("---> epoch : %d | batch : %d | Misclassified : %d" % (epoch, batch_num, miss_class))
            print("Summary of epoch : %d | Total batches : %d | Total Misclassified : %d" \
                  % (epoch, batch_num + 1, total_miss_class))
            if total_miss_class == 0: break
            #         if epoch > epoch_threshold : break
            epoch += 1
        print()
        print("Final adjusted weights: ", list(weights))
        print("\nFinal Hyperplane : %.2fx + %.2fy + %.2fz + %.2f" % tuple(weights))
        return (weights, epoch + 1)

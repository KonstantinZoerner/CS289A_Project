import numpy as np

class Data:

    def __init__(self, dataset="cancer"):
        # find path to dataset
        if dataset == "cancer":
            path = "datasets/breast_cancer/data.csv"
        elif dataset == "diabetes":
            path = "datasets/diabetes/diabetes_binary_health_indicators_BRFSS2015.csv"
        else:
            raise NotImplementedError("This dataset does not exist")

        # load data
        data = np.genfromtxt(path, delimiter=',', dtype=None)
        
        # split labels and features
        if dataset == "cancer":
            self.labels = data[:, [1]]
            self.features = data[:, 2:]
        else:
            self.labels = data[:, [0]]
            self.features = data[:, 1:]


if __name__ == "__main__":
    data = Data(dataset = "cancer")
    print(data.features.shape)
    print(data.labels.shape)
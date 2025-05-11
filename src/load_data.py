import numpy as np
import time

class Data:

    def __init__(self, dataset="cancer", verbose=False):
        print(f"Loading dataset: {dataset}")
        t_0 = time.perf_counter()
        # find path to dataset
        if dataset == "cancer":
            path = "datasets/breast_cancer/data.csv"
        elif dataset == "diabetes":
            path = "datasets/diabetes/diabetes_binary_health_indicators_BRFSS2015.csv"
        elif dataset == "diabetes_50_50":
            path = "datasets/diabetes/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
        else:
            raise NotImplementedError("This dataset does not exist")
        
        self.name = dataset

        # load data
        data = np.genfromtxt(path, delimiter=',', dtype=None)

        # split labels and features
        if dataset == "cancer":

            self.labels = (data[1:, 1] == b'M').astype(float)
            self.features = data[1:, 2:].astype(float)
        else:
            self.labels = data[1:, 0].astype(float)
            self.features = data[1:, 1:].astype(float)
    
        # compute constants
        self.n_samples, self.n_features = self.features.shape

        t_1 = time.perf_counter()
        print(f"Finished loading data in {t_1 - t_0} sec")
         

    def split_by_ratio(self, r_train, r_val, r_test, shuffel=True, rng = None):
        assert r_train + r_val + r_test <= 1

        n_train = int(np.round(r_train*self.n_samples))
        n_val = int(np.round((r_train + r_val)*self.n_samples) - n_train)
        n_test = int(np.round((r_train + r_val + r_test)*self.n_samples) - n_val - n_train)

        self.split_by_number(n_train, n_val, n_test, shuffel=shuffel, rng = rng)


    def split_by_number(self, n_train, n_val, n_test, shuffel=True, rng = None):
        assert n_train + n_val + n_test <= self.n_samples

        if rng == None:
            rng = np.random.default_rng()

        if shuffel:
            temp = np.hstack((self.labels.reshape(-1, 1), self.features))
            rng.shuffle(temp)
            self.labels = temp[:, 0]
            self.features = temp[:, 1:]
        
        self.train_features = self.features[:n_train, :]
        self.val_features = self.features[n_train:n_train + n_val, :]
        self.test_features = self.features[n_train + n_val:, :]

        self.train_labels = self.labels[:n_train]
        self.val_labels = self.labels[n_train:n_train + n_val]
        self.test_labels = self.labels[n_train + n_val:]
        

if __name__ == "__main__":
    data = Data(dataset = "cancer")
  

    data.split_by_ratio(0.6, 0.4, 0, shuffel=False)
    print(data.train_features)
    print(data.train_labels)
import sys
import os

import sklearn.model_selection
sys.path.append(os.path.abspath('.'))

import scipy.stats
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from src.losses import One_Zero_Loss
import numpy as np
import matplotlib.pyplot as plt
import src.load_data as load_data
import scipy
import tqdm

#TODO: add weighted knn

class KNN(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform'):
        super().__init__(n_neighbors=n_neighbors, weights=weights)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def model_size(self):
        return None # non-parametric model
    
    def grid_search_k(self, features, labels, 
                      ks = list(range(1, 21)),
                      loss = One_Zero_Loss(), verbose=False, k_fold = True):

        errors = np.zeros(len(ks))
        if k_fold:
            split = sklearn.model_selection.KFold(n_splits=5)
            indices = split.split(features)
        else:
            indices = ((np.arange(0, int(len(features) * 0.8)), np.arange(int(len(features) * 0.8), len(features))),)
        for (train_idx, val_idx) in tqdm.tqdm(indices): 
            train_features = features[train_idx]
            train_labels = labels[train_idx]
            val_features = features[val_idx]
            val_labels = labels[val_idx]

            self.fit(train_features, train_labels)
            for i, k in tqdm.tqdm(enumerate(ks)):
                self.n_neighbors = k
                pred = self.predict(val_features)
                errors[i] += loss(pred, val_labels)

        idx = np.argmin(errors)

        if verbose:
            print(f'Best k: {ks[idx]}, Error: {errors[idx]:.4f}')

        # Plot error vs k for k Nearest Neighbors
        if verbose:
            plt.plot(ks, errors)
            plt.xlabel('k')
            plt.ylabel('loss')
            plt.title('loss vs. k in kNN')
            plt.savefig(f"figures/knn/grid_search_k_knn.pdf", bbox_inches="tight")
            plt.show()

        self.n_neighbors = ks[idx]

            


    


if __name__ == "__main__":
    np.random.seed(0)

    data = load_data.Data("diabetes")
    data.split_by_ratio(0.8, 0.0, 0.2)
    knn = KNN(n_neighbors=5)
    knn.grid_search_k(data.train_features, data.train_labels, verbose=True, k_fold=False, ks=list(range(1, 60, 5)))
    knn.fit(data.train_features, data.train_labels)
    print(knn.score(data.test_features, data.test_labels))
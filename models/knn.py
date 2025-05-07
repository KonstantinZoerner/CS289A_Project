import sys
import os
sys.path.append(os.path.abspath('.'))

import scipy.stats
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from src.losses import One_Zero_Loss
import numpy as np
import matplotlib.pyplot as plt
import src.load_data as load_data
import scipy


class KNN(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform'):
        super().__init__(n_neighbors=n_neighbors, weights=weights)

    def model_size(self):
        return self.n_neighbors * self.n_features_in_ + 1


def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, loss = One_Zero_Loss(), verbose=True):
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')

    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)
        pred = scipy.stats.mode(train_labels[indices], axis=1)[0]
        pred = pred.reshape((-1, 1))
        print(pred.shape)
        error = loss(pred, test_labels)
        mean_errors.append(error)
        if verbose:
            print(f'Error: {error:.1f}')

    idx = np.argmin(mean_errors)
    print(f"best k = {ks[idx]}, mean displacement error (miles) : {mean_errors[idx]}")

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error (miles) vs. k in kNN')
        plt.savefig(f"grid_search_(is_weighted={is_weighted}).pdf", bbox_inches="tight")
        plt.show()

    return ks[idx]

if __name__ == "__main__":
    data = load_data.Data()
    data.split_by_ratio(0.8, 0, 0.2)
    print(data.test_labels)
    grid_search(data.train_features, data.train_labels, data.test_features, data.test_labels)
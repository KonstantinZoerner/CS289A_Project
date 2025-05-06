from sklearn.neighbors import NearestNeighbors
from src.losses import One_Zero_Loss
import numpy as np
import matplotlib.pyplot as plt
import src.load_data as load_data

def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')

    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)
        error = One_Zero_Loss(indices, train_labels, test_labels, distances, is_weighted)
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
    
import numpy as np

def sort_np_arrays_by_first_column(first_array, *arrays):
    combined = np.column_stack((first_array, *arrays))
    sorted_combined = combined[combined[:, 0].argsort()]
    idx = np.argsort(combined[:, 0])
    sorted_combined = combined[idx]
    # Extract the sorted arrays
    sorted_arrays = [sorted_combined[:, i] for i in range(1, combined.shape[1])]
    sorted_first_array = sorted_combined[:, 0]
    return sorted_first_array, *sorted_arrays

if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.randint(0, 10, 10)
    B = np.random.randint(0, 10, 10)
    C = np.random.randint(0, 10, 10)
    
    print("A", A)
    print("B", B)
    print("C", C)
    A_sorted, B_sorted, C_sorted = sort_np_arrays_by_first_column(A, B, C)
    print("A_sorted", A_sorted)
    print("B_sorted", B_sorted)
    print("C_sorted", C_sorted)
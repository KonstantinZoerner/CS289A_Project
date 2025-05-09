from sklearn.svm import LinearSVC
from sklearn.svm import SVC

class SVM(LinearSVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, dual="auto")

    def model_size(self):
        return self.n_features_in_*2
    
class non_linear_SVM(SVC):
    def __init__(self, **kwargs):
        if "kernel" not in kwargs:
            kwargs["kernel"] = "rbf"
        super().__init__(**kwargs)

    def model_size(self):
        return 0 # TODO: figure this out
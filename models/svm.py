from sklearn.svm import LinearSVC

class SVM(LinearSVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, dual="auto")

    def model_size(self):
        return self.n_features_in_*2
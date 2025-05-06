from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

class QDA(QuadraticDiscriminantAnalysis):
    def __init__(self, **kwargs):
        # Force store_covariance=True, override if user provides a different value
        kwargs.setdefault('store_covariance', True)
        super().__init__(**kwargs)

    def model_size(self):
        return self.means_.size + self.covariance_.size

class LDA(LinearDiscriminantAnalysis):
    def __init__(self, **kwargs):
        # Force store_covariance=True, override if user provides a different value
        kwargs.setdefault('store_covariance', True)
        super().__init__(**kwargs)

    def model_size(self):
        return self.means_.size + self.covariance_.size
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import numpy as np

class QDA(QuadraticDiscriminantAnalysis):
    def __init__(self, **kwargs):
        # Force store_covariance=True, override if user provides a different value
        kwargs.setdefault('store_covariance', True)
        super().__init__(**kwargs)

    def model_size(self):
        return np.sum([mean.size for mean in self.means_]) + np.sum([covariance.size for covariance in self.covariance_])

class LDA(LinearDiscriminantAnalysis):
    def __init__(self, **kwargs):
        # Force store_covariance=True, override if user provides a different value
        kwargs.setdefault('store_covariance', True)
        super().__init__(**kwargs)

    def model_size(self):
        return np.sum([mean.size for mean in self.means_]) + np.sum([covariance.size for covariance in self.covariance_])
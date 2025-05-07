from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoost(AdaBoostClassifier):
    def __init__(self, **kwargs):
        if 'estimator' not in kwargs:
            kwargs['estimator'] = DecisionTreeClassifier(max_depth=2)
        super().__init__(**kwargs)

    def model_size(self):
        #TODO what about leaf nodes
        return (1 + np.sum([estimator.tree_.node_count * 2 for estimator in self.estimator_]))
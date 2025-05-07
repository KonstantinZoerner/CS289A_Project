from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class DecicisonTree(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_size(self):
        #TODO what about leaf nodes
        return self.tree_.node_count * 2
    
class RandomForest(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_size(self):
        #TODO what about leaf nodes
        return np.sum([estimator.tree_.node_count * 2 for estimator in self.estimators_])
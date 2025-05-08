import numpy as np
import scipy
import scipy.stats

class EnsembleLearner:

    def __init__(self, *learners):
        self.learners = learners
        self.n_learners = len(learners)

    def fit(self, X, y):
        for learner in self.learners:
            learner.fit(X, y)

    def predict(self, X):
        predictions = []
        for learner in self.learners:
            predictions.append(learner.predict(X))
        predictions = np.array(predictions)
        prediction = scipy.stats.mode(predictions, axis=0, keepdims=False)[0].flatten()
        return prediction
    
    def model_size(self):
        size = 0
        for learner in self.learners:
            size += learner.model_size()
        return size
    

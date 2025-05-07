from sklearn.tree import DecisionTreeClassifier

class DecicisonTree(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_size(self):
        #TODO what about leaf nodes
        return self.tree_.node_count * 2
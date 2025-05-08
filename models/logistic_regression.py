from sklearn.linear_model import LogisticRegression


class LogisticRegression(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_size(self):
        #TODO
        pass
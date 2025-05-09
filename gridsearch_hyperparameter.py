from sklearn.model_selection import GridSearchCV
import numpy as np
import src.load_data as load_data

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib as mpl
from src.utils import sort_np_arrays_by_first_column


class HyperparameterTuning:
    def __init__(self, model, param_grid, data, runs=1):
        self.model = model
        self.param_grid = param_grid
        self.data = data
        self.name = model.__class__.__name__
        self.runs = runs

    def tune_hyperparameters(self, verbose=True):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=5, n_jobs=-1,)

        self.mean_test_scores = []
        for i in tqdm(range(self.runs)):
            self.data.split_by_ratio(0.8, 0.2, 0, shuffel=True)
            grid_search.fit(self.data.train_features , self.data.train_labels)
            self.mean_test_scores.append(grid_search.cv_results_['mean_test_score'])
        
        self.mean_test_scores = np.mean(self.mean_test_scores, axis=0)
        print(self.mean_test_scores.shape)
        idx = np.argmax(self.mean_test_scores)

        self.best_params = grid_search.cv_results_['params'][idx]
        self.best_score = self.mean_test_scores[idx]
        self.params = grid_search.cv_results_['params']


        print("Best hyperparameters:", self.best_params)
        print("Best score:", self.best_score)

    def plot_results_1D(self, param_name, x_scale='log'):
        fig_width = 3.25
        fig_height = fig_width * 0.75
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1,
            "lines.markersize": 4,
            "figure.figsize": [fig_width, fig_height],
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })


        param = [params[param_name] for params in self.params]

        unique_params = list(set(param))
        avg_scores = [np.mean([self.mean_test_scores[i] for i in range(len(param)) if param[i] == up]) for up in unique_params]
        unique_params, avg_scores = sort_np_arrays_by_first_column(np.array(unique_params), np.array(avg_scores))

        plt.plot(unique_params, avg_scores, marker='o')
        plt.xscale(x_scale)
        plt.xlabel(param_name)
        plt.ylabel('Mean Accuracy')
        plt.grid(True)
        plt.savefig(f"grid_search_data/{self.name}_{self.runs}_{param_name}.pdf", bbox_inches='tight')
        plt.show()

    def save_results(self):
        results = {
        "params": self.params,
        "mean_test_scores": self.mean_test_scores.tolist(),
        "best_params": self.best_params,
        "best_score": self.best_score
        }

        with open(f"grid_search_data/{self.name}_{self.runs}.json", "w") as f:
            json.dump(results, f, indent=4)

def tune_SVC():
    model = SVC()
    tuner = HyperparameterTuning(model, param_grid={'C': [0.1, 1, 10, 100, 1000, 10000], 'kernel': ['linear']}, data=data)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='C', x_scale='log')

    # "C" : 100, kernel = "linear"

def tune_DecisionTree(data):
    model = DecisionTreeClassifier()
    tuner = HyperparameterTuning(model, param_grid={'max_depth': range(1, 20)}, data=data, runs=1000)
    tuner.tune_hyperparameters(verbose=True)
    tuner.save_results()
    tuner.plot_results_1D(param_name='max_depth', x_scale='linear')

    # {'max_depth': 4}
    # Save results to a file

def tune_RandomForest(data):
    model = RandomForestClassifier()
    tuner = HyperparameterTuning(model, param_grid={'n_estimators': [10, 50, 100, 200, 300, 400, 500], 'max_depth': range(1, 11)}, data=data, runs=5)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='max_depth', x_scale='linear')
    tuner.plot_results_1D(param_name='n_estimators', x_scale='linear')
    tuner.save_results()

def tune_LogisticRegression(data):
    model = LogisticRegression()
    tuner = HyperparameterTuning(model, param_grid={'C': [0.1, 1, 10, 100, 1000, 10000]}, data=data)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='C', x_scale='log')

def tune_KNeighborsClassifier(data):
    model = KNeighborsClassifier()
    tuner = HyperparameterTuning(model, param_grid={'n_neighbors': range(1, 20)}, data=data)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='n_neighbors', x_scale='linear')

def tune_AdaBoostClassifier(data):
    model = AdaBoostClassifier()
    tuner = HyperparameterTuning(model, param_grid={'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1, 10]}, data=data)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='n_estimators', x_scale='linear')
    tuner.plot_results_1D(param_name='learning_rate', x_scale='linear')
    tuner.save_results()

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    data = load_data.Data(dataset="cancer")

    # tune_SVC(data)
    # tune_DecisionTree(data)
    tune_RandomForest(data)

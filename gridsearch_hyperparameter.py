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

# ----------------------------------------------------
#                       Constants
# ----------------------------------------------------  

DIABETES_RATIO = 0.2
fig_width = 3.25
fig_height = fig_width * 0.75
FULL = {
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
        }
HALF = {
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
            "figure.figsize": [0.5 * fig_width, 0.5 * fig_height],
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        }

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
            if self.data.name == "cancer":
                self.data.split_by_ratio(0.8, 0.2, 0, shuffel=True)
            else:
                self.data.split_by_ratio(0.8*DIABETES_RATIO, 0.2, 0, shuffel=True)
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

    def plot_results_1D(self, param_name, x_scale='log', 
                        show=False, save=True, close=True, 
                        graph_settings=FULL, filename=None, ax= None, 
                        label=None, yrange=None):
        mpl.rcParams.update(graph_settings)
        if ax is None:
            fig, ax = plt.subplots()

        param = [params[param_name] for params in self.params]

        unique_params = list(set(param))
        avg_scores = [np.mean([self.mean_test_scores[i] for i in range(len(param)) if param[i] == up]) for up in unique_params]
        unique_params, avg_scores = sort_np_arrays_by_first_column(np.array(unique_params), np.array(avg_scores))

        ax.plot(unique_params, avg_scores, marker='o', label=label)
        if label is not None:
            ax.legend()
        ax.set_xscale(x_scale)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Mean Accuracy')
        if yrange is not None:
            ax.set_ylim(yrange)
        ax.grid(True)
        if save and filename is None:
            plt.savefig(f"grid_search_data/{self.data.name}_{self.name}_{self.runs}_{param_name}.pdf", bbox_inches='tight')
        elif save and filename is not None:
            plt.savefig(f"grid_search_data/{filename}.pdf", bbox_inches='tight')
        if show:
            plt.show()
        if close:
            plt.close()
        return ax

    def save_results(self, filename=None):
        results = {
        "params": self.params,
        "mean_test_scores": self.mean_test_scores.tolist(),
        "best_params": self.best_params,
        "best_score": self.best_score
        }
        if filename is None:
            filename = f"{self.data.name}_{self.name}_{self.runs}"
        with open(f"grid_search_data/{filename}.json", "w") as f:
            json.dump(results, f, indent=4)

def tune_SVC(data):
    model = SVC()
    if data.name == "cancer":
        n_runs = 10
    elif data.name == "diabetes":
        n_runs = 1
    else:
        raise NotImplementedError(f"Dataset {data.name} is not implemented")
    #tuner = HyperparameterTuning(model, param_grid={'C': [0.1, 1, 10, 20, 50, 100, 200, 500, 1000], 'kernel': ['linear']}, data=data, runs=n_runs)
    tuner = HyperparameterTuning(model, param_grid={'C': [0.1, 1, 10, 20], 'kernel': ['rbf','linear']}, data=data, runs=n_runs)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='C', x_scale='log')
    tuner.save_results()
    # "C" : 100, kernel = "linear"

def tune_DecisionTree(data):
    model = DecisionTreeClassifier()
    if data.name == "cancer":
        n_runs = 1000
    elif data.name == "diabetes":
        n_runs = 100
    else:
        raise NotImplementedError(f"Dataset {data.name} is not implemented")
    tuner = HyperparameterTuning(model, param_grid={'max_depth': range(1, 20)}, data=data, runs=n_runs)
    tuner.tune_hyperparameters(verbose=True)
    tuner.save_results()
    tuner.plot_results_1D(param_name='max_depth', x_scale='linear')

    # {'max_depth': 4}
    # Save results to a file

def tune_RandomForest(data):
    model = RandomForestClassifier()
    if data.name == "cancer":
        n_runs = 30
    elif data.name == "diabetes":
        n_runs = 5
    else:
        raise NotImplementedError(f"Dataset {data.name} is not implemented")

    #tuner = HyperparameterTuning(model, param_grid={'n_estimators': [10, 50, 100, 200, 300, 400, 500], 'max_depth': range(1, 16)}, data=data, runs=n_runs)
    tuner = HyperparameterTuning(model, param_grid={'n_estimators': [10, 50, 100, 200, 300, 400, 500], 'max_depth': range(1, 16, 2)}, data=data, runs=n_runs)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='max_depth', x_scale='linear')
    tuner.plot_results_1D(param_name='n_estimators', x_scale='linear')
    tuner.save_results()

def tune_LogisticRegression(data):
    model = LogisticRegression()
    tuner = HyperparameterTuning(model, param_grid={'C': [0.1, 1, 10, 100, 1000, 10000]}, data=data)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='C', x_scale='log')
    tuner.save_results()

def tune_KNeighborsClassifier(data):
    model = KNeighborsClassifier()
    tuner = HyperparameterTuning(model, param_grid={'n_neighbors': range(1, 40,2)}, data=data)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='n_neighbors', x_scale='linear')
    tuner.save_results()

def tune_AdaBoostClassifier(data, depth=2):
    filename=f"Addaboost_Diabetes_{depth}_few_estimators"
    model = RandomForestClassifier()
    if data.name == "cancer":
        n_runs = 10
    elif data.name == "diabetes":
        n_runs = 10
    else:
        raise NotImplementedError(f"Dataset {data.name} is not implemented")
    
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth))
    tuner = HyperparameterTuning(model, param_grid={'n_estimators': range(1, 41)},
                                data=data, runs=n_runs)
    tuner.tune_hyperparameters(verbose=True)
    tuner.plot_results_1D(param_name='n_estimators', x_scale='linear', filename=filename)
    tuner.save_results(filename=filename)

def load_past_data(dataset, file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    tuner = HyperparameterTuning(dataset, None, None)
    tuner.mean_test_scores = np.array(data["mean_test_scores"])
    tuner.params = data["params"]
    tuner.best_params = data["best_params"]
    tuner.best_score = data["best_score"]
    return tuner

def plot_both_trees():
    tuner_cancer = load_past_data("cancer", "grid_search_data/cancer_DecisionTreeClassifier_1000.json")
    tuner_diabetes = load_past_data("diabetes", "grid_search_data/diabetes_DecisionTreeClassifier_100.json")

    ax1 = tuner_cancer.plot_results_1D(param_name='max_depth', x_scale='linear', show=False, save=False, close=False, label="Breast Cancer")
    tuner_diabetes.plot_results_1D(param_name='max_depth', x_scale='linear', show=True, save=True, close=True, filename="trees_combined_(1000,100)", ax=ax1, label="Diabetes")

def plot_both_forests():
    tuner_cancer = load_past_data("cancer", "grid_search_data/cancer_RandomForestClassifier_30.json")
    tuner_diabetes = load_past_data("diabetes", "grid_search_data/diabetes_RandomForestClassifier_5.json")

    ax1 = tuner_cancer.plot_results_1D(param_name='max_depth', x_scale='linear', show=False, save=False, close=False, label="Breast Cancer")
    tuner_diabetes.plot_results_1D(param_name='max_depth', x_scale='linear', show=True, save=True, close=True, filename="forests_combined_max_depth_(30,5)", ax=ax1.twinx(), label="Diabetes", yrange=(0.86, 0.867))

    ax1 = tuner_cancer.plot_results_1D(param_name='n_estimators', x_scale='linear', show=False, save=False, close=False, label="Breast Cancer")
    tuner_diabetes.plot_results_1D(param_name='n_estimators', x_scale='linear', show=True, save=True, close=True, filename="forests_combined_n_estimators_(30,5)", ax=ax1.twinx(), label="Diabetes")



if __name__ == "__main__":
    rng = np.random.default_rng(1)
    data = load_data.Data(dataset="diabetes")

    # tune_SVC(data)
    # tune_DecisionTree(data)
    # tune_RandomForest(data)
    # tune_KNeighborsClassifier(data)
    # plot_both_trees()
    # plot_both_forests()
    tune_AdaBoostClassifier(data, 2)

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd 
import src.utils as utils

model_to_color = {
    "QDA": "C0",     
    "LDA": "C1",     
    "Decision Tree": "C2",  
    "Decicision Tree": "C2",  
    "Random Forest": "C3",  
    "Random Forrest": "C3",  
    "Random Forest (No Bootstrap)": "C4",  
    "SVM": "C4",     
    "AdaBoost": "C0", 
    "kNN": "C1",     
    "Logistic Regression": "C2",  
    "Neural Network": "C3" 
}

model_to_style = {
    "QDA": "-",     
    "LDA": "-",     
    "Decision Tree": "-",  
    "Decicision Tree": "-",  
    "Random Forest": "-",  
    "Random Forrest": "-",  
    "Random Forest (No Bootstrap)": "-",  
    "SVM": "-",     
    "AdaBoost": "--",  
    "kNN": "--",     
    "Logistic Regression": "--",  
    "Neural Network": "--"  
}

model_to_marker = {
    "QDA": "+",     
    "LDA": "+",     
    "Decision Tree": "+",  
    "Decicision Tree": "+",  
    "Random Forest": "+",  
    "Random Forrest": "+",  
    "Random Forest (No Bootstrap)": "+",  
    "SVM": "+",     
    "AdaBoost": "2",  
    "kNN": "2",     
    "Logistic Regression": "2",  
    "Neural Network": "2" 
}



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
        "figure.figsize": [0.5*fig_width, 0.5*fig_height],
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }

# NAME = "tuned_cancer_[0.1, 1.05, 0.05]_100"
NAME = "nacht_tuned_diabetes_log[-1, 0, 50]_100"
FILENAME = "runtime_analysis_" + NAME

# ------------------------------------------------------------
# Scatter plot of two variables 
# ------------------------------------------------------------
def plot_analysis_x_vs_y(dataset="diabetes", x="model_size", y="error", 
                         models=None, remove_modles=None, 
                         x_label="Number of Parameters", y_label="Error",
                         title=None, x_scale="log", y_scale="log", train_ratio=1.0, figure_settings=FULL,
                         legend=False):
    
    runtime_results = pd.read_csv(f"analysis_data/{FILENAME}.csv", index_col=[0, 1, 2])
    if models is None:
        models = list(runtime_results.index.get_level_values("model")) 
    if remove_modles is not None:    
        models = [x for x in models if x not in remove_modles]   
    x_data = runtime_results.loc[(dataset, train_ratio, models), x].to_numpy()
    y_data = runtime_results.loc[(dataset, train_ratio, models), y].to_numpy()
    
    # plot
    mpl.rcParams.update(figure_settings)

    for (y_data, complexity, label) in zip(y_data, x_data, models):
        plt.scatter(complexity, y_data, label=label, color=model_to_color[label], marker=model_to_marker[label], s=40)
    if legend:
        plt.legend()
        l = "l"
    else:
        l = ""
    plt.grid(which='both', axis='y')
    plt.grid(which='major', axis='x')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # NO titles to comply with ICML format guidelines
    # if type(title) == str:
    #     plt.title(title)
    # elif title is not None:
    #     plt.title(f"{x_label} vs {y_label} for {dataset} dataset")
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.savefig(f"figures/{x}_vs_{y}_{dataset}_{NAME}{l}.pdf", bbox_inches="tight")
    plt.show()

def plot_model_complexity_vs_accuracy(dataset="diabetes"):
    plot_analysis_x_vs_y(dataset=dataset, x="model_size", y="error",
                         x_label="Number of Parameters", y_label="Error",
                         title=True)

def plot_runtime_vs_accuracy(dataset="diabetes", **kwargs):
    plot_analysis_x_vs_y(dataset=dataset, x="training_time", y="error",
                         x_label="Training Time (s)", y_label="Error",
                         title=True, **kwargs)

def plot_runtime_vs_model_complexity(dataset="diabetes"):
    plot_analysis_x_vs_y(dataset=dataset, x="training_time", y="model_size",
                         x_label="Training Time (s)", y_label="Number of Parameters",
                         title=True)
    
def plot_runtime_vs_predict_time(dataset="diabetes"):
    plot_analysis_x_vs_y(dataset=dataset, x="training_time", y="predict_time",
                         x_label="Training Time (s)", y_label="Prediction Time (s)",
                         title=True)
    
# ------------------------------------------------------------
# Development of variable vs training data ratio
# ------------------------------------------------------------
def plot_x_vs_training_data_ratio(dataset="diabetes", y="error", 
                         models=None, remove_models=None, y_label="Error",
                         title=None, x_scale="log", y_scale="linear", 
                         train_ratios=None, remove_ratios=None, figure_settings=FULL, legend=False):

    runtime_results = pd.read_csv(f"analysis_data/{FILENAME}.csv", index_col=[0, 1, 2])
    if models is None:
        models = list(runtime_results.index.get_level_values("model").unique()) 
    if remove_models is not None:    
        models = [x for x in models if x not in remove_models]  
    if train_ratios is None:
        train_ratios = runtime_results.index.get_level_values("training_ratio").unique()
    else:
        train_ratios = [x for x in train_ratios if x not in remove_ratios]
         
    # plot
    print("models", models)
    mpl.rcParams.update(figure_settings)

    for i, label in enumerate(models):
        y_data = runtime_results.loc[(dataset, train_ratios, label), y].to_numpy()
        sorted_train_ratios, sorted_y_data = utils.sort_np_arrays_by_first_column(train_ratios, y_data)
        
        plt.plot(sorted_train_ratios, sorted_y_data, label=label, color=model_to_color[label], linestyle = model_to_style[label])
    if legend:
        plt.legend()
        l = "l"
    else:
        l = ""
    plt.grid(which="both")
    plt.xlabel("Ratio of used Training Data")
    plt.ylabel(y_label)
    # No plot title to comply with ICML format guidelines
    #if type(title) == str:
    #    plt.title(title)
    #elif title is not None:
    #    plt.title(f"{y_label} vs Ratio of used Training Data for {dataset} dataset")
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.savefig(f"figures/{y}_vs_training_ratio_{dataset}_{NAME}{l}.pdf", bbox_inches="tight")
    plt.show()

def plot_error_vs_training_data_ratio(dataset="diabetes", *args, **kwargs):
    plot_x_vs_training_data_ratio(dataset=dataset, y="error", 
                         y_label="Error", title=True, remove_models=["SVM"], *args, **kwargs)


if __name__ == "__main__":
    NAME = "nacht_tuned_diabetes_log[-3_5, 0, 100]_10"
    FILENAME = "runtime_analysis_" + NAME
    plot_error_vs_training_data_ratio(dataset="diabetes", x_scale="log", y_scale="log", legend=True)
    plot_error_vs_training_data_ratio(dataset="diabetes", x_scale="log", y_scale="log", legend=False)
    plot_runtime_vs_accuracy(dataset="diabetes", train_ratio=1.0, remove_modles=["QDA"], legend=True)
    plot_runtime_vs_accuracy(dataset="diabetes", train_ratio=1.0, remove_modles=["QDA"], legend=False)


    NAME = "nacht_tuned_cancer_log[-1, 0, 100]_100"
    FILENAME = "runtime_analysis_" + NAME
    plot_error_vs_training_data_ratio(dataset="cancer", x_scale="linear", legend=True)
    plot_error_vs_training_data_ratio(dataset="cancer", x_scale="linear", legend=False)
    plot_runtime_vs_accuracy(dataset="cancer", train_ratio=1.0, remove_modles=["QDA"], legend=True)
    plot_runtime_vs_accuracy(dataset="cancer", train_ratio=1.0, remove_modles=["QDA"], legend=False)


    # NAME = "tuned_diabetes_[00.0005, 0.001, 0.0005]_100"
    # FILENAME = "runtime_analysis_" + NAME
    # plot_error_vs_training_data_ratio(dataset="diabetes", x_scale="linear", legend=True)
    # plot_error_vs_training_data_ratio(dataset="diabetes", x_scale="linear", legend=False)
    
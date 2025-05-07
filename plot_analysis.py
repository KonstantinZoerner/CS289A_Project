import matplotlib.pyplot as plt
import pandas as pd 

def plot_analysis_x_vs_y(dataset="diabetes", x="model_size", y="error", 
                         models=None, remove_modles=None, 
                         x_label="Number of Parameters", y_label="Error",
                         title=None, x_scale="log", y_scale="log"):
    
    runtime_results = pd.read_csv("analysis_data/runtime_analysis.csv", index_col=[0, 1])
    if models is None:
        models = list(runtime_results.index.get_level_values("model")) 
    elif models is not None:    
        models = [x for x in models if x not in remove_modles]   
    x_data = runtime_results.loc[(dataset, models), x].to_numpy()
    y_data = runtime_results.loc[(dataset, models), y].to_numpy()
    
    # plot
    for (y_data, complexity, label) in zip(y_data, x_data, models):
        plt.scatter(complexity, y_data, label=label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if type(title) == str:
        plt.title(title)
    elif title is not None:
        plt.title(f"{x_label} vs {y_label} for {dataset} dataset")
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.savefig(f"figures/{x}_vs_{y}_{dataset}.pdf", bbox_inches="tight")
    plt.show()


def plot_model_complexity_vs_accuracy(dataset="diabetes"):
    plot_analysis_x_vs_y(dataset=dataset, x="model_size", y="error",
                         x_label="Number of Parameters", y_label="Error",
                         title=True)

def plot_runtime_vs_accuracy(dataset="diabetes"):
    plot_analysis_x_vs_y(dataset=dataset, x="training_time", y="error",
                         x_label="Training Time (s)", y_label="Error",
                         title=True)

def plot_runtime_vs_model_complexity(dataset="diabetes"):
    plot_analysis_x_vs_y(dataset=dataset, x="training_time", y="model_size",
                         x_label="Training Time (s)", y_label="Number of Parameters",
                         title=True)
    
def plot_runtime_vs_predict_time(dataset="diabetes"):
    plot_analysis_x_vs_y(dataset=dataset, x="training_time", y="predict_time",
                         x_label="Training Time (s)", y_label="Prediction Time (s)",
                         title=True)
    


if __name__ == "__main__":
    plot_model_complexity_vs_accuracy()
    plot_runtime_vs_accuracy()
    plot_runtime_vs_model_complexity()
    plot_runtime_vs_predict_time()
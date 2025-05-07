import matplotlib.pyplot as plt
import pandas as pd 

def plot_model_complexity_vs_accuracy():
    runtime_results = pd.read_csv("analysis_data/runtime_analysis.csv", index_col=[0, 1])
    print(runtime_results)


if __name__ == "__main__":
    plot_model_complexity_vs_accuracy()
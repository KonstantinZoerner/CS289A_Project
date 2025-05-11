import pandas as pd
import numpy as np

def generate_latex_table_full(cancer_path, diabetes_path, cancer_ratio, diabetes_ratio, decimal_places, label, data):
    # Load CSV with MultiIndex
    df_cancer = pd.read_csv(cancer_path, index_col=[0, 1, 2])
    df_diabetes = pd.read_csv(diabetes_path, index_col=[0, 1, 2])
    models_cancer = df_cancer.index.get_level_values("model").unique().tolist()
    models_diabetes = df_diabetes.index.get_level_values("model").unique().tolist()
    common_models = list(set(models_cancer).intersection(models_diabetes))

    err_cancer = df_cancer.loc[("cancer", cancer_ratio, common_models), data].to_numpy()

    err_diabetes = df_diabetes.loc[("diabetes", diabetes_ratio, common_models), data].to_numpy()
    
    combined = pd.DataFrame(data = np.vstack((err_cancer,err_diabetes)).T, index = common_models, columns = ["cancer", "diabetes"])

    combined = combined.sort_values(by="cancer", ascending=True)

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"    \caption{Validation errors for all models on both datasets. Sorted by cancer validation error.}",
        r"    \label{sample-table}",
        r"    \vskip 0.15in",
        r"    \begin{center}",
        r"    \begin{small}",
        r"    \begin{sc}",
        r"    \sisetup{table-format=2.2}",
        r"    \begin{tabular}{lSS}",
        r"        \toprule",
        r"        \multirow{2}{*}{Model} & \multicolumn{2}{c}{"+label+r"}\\",
        r"        & {Cancer} & {Diabetes} \\",
        r"        \midrule"
    ]

    for model_name, row in combined.iterrows():
        if data == "error":
            factor = 100
        elif data == "training_time":
            factor = 1000
        else:
            factor = 1
            
        value_cancer = factor * row["cancer"]
        value_diabetes = factor * row["diabetes"]
        latex_lines.append(f"        {model_name} & {value_cancer:.{decimal_places}f} & {value_diabetes:.{decimal_places}f}\\\\")

    latex_lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \end{sc}",
        r"    \end{small}",
        r"    \end{center}",
        r"    \vskip -0.1in",
        r"\end{table}"
    ]

    return "\n".join(latex_lines)

# Example usage:
latex_code = generate_latex_table_full(
    cancer_path="analysis_data/runtime_analysis_nacht_tuned_cancer_log[-1, 0, 100]_100.csv",
    diabetes_path="analysis_data/runtime_analysis_nacht_tuned_diabetes_log[-3_5, 0, 100]_10.csv",
    cancer_ratio=1.0,
    diabetes_ratio = 1.0,
    decimal_places = 2,
    label = r"Training Time [ms]",
    #label = r"Error [\%]",
    data = "training_time"
)

print(latex_code)
import pandas as pd
import numpy as np

def generate_latex_table_full(cancer_path, diabetes_path, cancer_ratio, diabetes_ratio):
    # Load CSV with MultiIndex
    df_cancer = pd.read_csv(cancer_path, index_col=[0, 1, 2])
    df_diabetes = pd.read_csv(diabetes_path, index_col=[0, 1, 2])
    models_cancer = df_cancer.index.get_level_values("model").unique().tolist()
    models_diabetes = df_diabetes.index.get_level_values("model").unique().tolist()
    common_models = list(set(models_cancer).intersection(models_diabetes))

    err_cancer = df_cancer.loc[("cancer", cancer_ratio, common_models), "error"].to_numpy()

    err_diabetes = df_diabetes.loc[("diabetes", diabetes_ratio, common_models), "error"].to_numpy()
    
    combined = pd.DataFrame(data = np.vstack((err_cancer,err_diabetes)).T, index = common_models, columns = ["cancer", "diabetes"])

    combined = combined.sort_values(by="cancer", ascending=True)

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[t]",
        r"    \caption{Validation errors for all models on both datasets. Sorted by cancer validation error.}",
        r"    \label{sample-table}",
        r"    \vskip 0.15in",
        r"    \begin{center}",
        r"    \begin{small}",
        r"    \begin{sc}",
        r"    \sisetup{table-format=2.2}",
        r"    \begin{tabular}{lSS}",
        r"        \toprule",
        r"        \multirow{2}{*}{Model} & \multicolumn{2}{c}{Error [\%]}\\",
        r"        & {Cancer} & {Diabetes} \\",
        r"        \midrule"
    ]

    for model_name, row in combined.iterrows():
        value_cancer = 100 * row["cancer"]
        value_diabetes = 100 * row["diabetes"]
        latex_lines.append(f"        {model_name} & {value_cancer:.2f} & {value_diabetes:.2f}\\\\")

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
    cancer_path="analysis_data/runtime_analysis_cancer_100_runs_[0.05-1.0, 0.5].csv",
    diabetes_path="analysis_data/runtime_analysis_tuned_diabetes_[0.01, 0.105, 0.005]_100.csv",
    cancer_ratio=1,
    diabetes_ratio = 0.01
)

print(latex_code)
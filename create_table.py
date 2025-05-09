import pandas as pd

def generate_latex_table_full(csv_path, dataset_name, category, target_ratio, decimal_places=1):
    # Load CSV with MultiIndex
    df = pd.read_csv(csv_path, index_col=[0, 1, 2])
    df.index.set_names(['dataset', 'training_ratio', 'model'], inplace=True)
    df = df.sort_index()

    # Filter for specific dataset and ratio
    df_filtered = df.loc[(dataset_name, target_ratio)]

    # Sort by error (optional, for cleaner table)
    df_filtered = df_filtered.sort_values(category)

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[t]",
        f"    \\caption{{Classification errors for all models on the {dataset_name} validation dataset. Trained on {target_ratio*100:.0f}\% of the training data and sorted by validation error.}}",
        r"    \label{sample-table}",
        r"    \vskip 0.15in",
        r"    \begin{center}",
        r"    \begin{small}",
        r"    \begin{sc}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        Model & Error [\%] \\",
        r"        \midrule"
    ]

    for model_name, row in df_filtered.iterrows():
        value = row[category]
        if category == "error":
            value_str = f"{100 * value:.{decimal_places}f}"
        elif category == "training_time":
            value_str = f"{1000 * value:.{decimal_places}f}"
        else:
            value_str = f"{value:.0f}"
        latex_lines.append(f"        {model_name} & {value_str}\\\\")

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
    csv_path="analysis_data/runtime_analysis_cancer_100_runs_[0.05-1.0, 0.5].csv",
    dataset_name="cancer",
    category = "training_time",
    target_ratio=1,
    decimal_places=2
)
print(latex_code)

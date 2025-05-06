import time
import numpy as np
import pandas as pd
import src.load_data as load_data
import models.GDA as GDA

models = {"GDA": GDA.QDA(),
          "LDA": GDA.LDA()}

datasets = ["cancer", "diabetes"]

index = pd.MultiIndex.from_product([datasets, models.keys()], names=["dataset", "model"])

training_times = pd.DataFrame(index=index, columns=["time"])
predict_times = pd.DataFrame(index=index, columns=["time"])

for dataset in datasets:
    data = load_data.Data(dataset)
    data.split_by_ratio(0.6, 0.4, 0.0)

    for name, model in models.items():
        print(f"Analysing {name}")
        t_0 = time.perf_counter()
        model.fit(data.train_features, data.train_labels)
        t_1 = time.perf_counter()

        training_times.loc[(dataset, name), "time"] = t_1 - t_0


        predict_times.loc[(dataset, name), "time"] = t_1 - t_0

print(training_times)
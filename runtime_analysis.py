import time
import numpy as np
import pandas as pd
import src.load_data as load_data
import models.GDA as GDA
import models.decision_trees as dt
import models.knn as knn
from src.losses import One_Zero_Loss

rng = np.random.default_rng(1)

models = {"GDA": GDA.QDA(reg_param = 1e-4),
          "LDA": GDA.LDA(),
          "Decicision Tree": dt.DecicisonTree(),
          "Random Forrest": dt.RandomForest(),
          "Random Forrest (No Bootstrap)": dt.RandomForest(bootstrap = False),}
          #"kNN": knn.KNN()

datasets = ["cancer", "diabetes"]

index = pd.MultiIndex.from_product([datasets, models.keys()], names=["dataset", "model"])

training_times = pd.DataFrame(index=index, columns=["time"])
predict_times = pd.DataFrame(index=index, columns=["time"])
error = pd.DataFrame(index=index, columns=["error"])

for dataset in datasets:
    data = load_data.Data(dataset, verbose=True)
    data.split_by_ratio(0.6, 0.4, 0.0, rng)

    for name, model in models.items():
        print(f"Analysing {name}")
        t_0 = time.perf_counter()
        model.fit(data.train_features, data.train_labels)

        t_1 = time.perf_counter()
        training_times.loc[(dataset, name), "time"] = t_1 - t_0

        y_pred = model.predict(data.val_features)

        t_2 = time.perf_counter()
        predict_times.loc[(dataset, name), "time"] = t_2 - t_1
        
        loss = One_Zero_Loss()
        error.loc[(dataset, name), "error"] = loss(y_pred, data.val_labels)

print(f"\nTRAINING TIMES\n{training_times}")
print(f"\nPREDICT TIMES\n{predict_times}")
print(f"\nERROR\n{error}")
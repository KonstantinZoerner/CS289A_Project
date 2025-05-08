import time
import numpy as np
import pandas as pd
import src.load_data as load_data
import models.GDA as GDA
import models.decision_trees as dt
import models.knn as knn
import models.svm as svm
import models.ada_boost as ada_boost
from src.losses import One_Zero_Loss
import os

#-------------------------------------------------------------
#                           Setteings
#-------------------------------------------------------------

FILENAME = "runtime_analysis_test"
LOAD_DATA = True
SAVE_DATA = True

# datasets = ["cancer", "diabetes"]
datasets = ["diabetes"]

#ratio_training_data = np.arange(0.1, 1.1, 0.1)
# ratio_training_data = np.logspace(-3, 0, 10, base=10)
ratio_training_data = np.array([0.25, 0.5])
print("ratios:", ratio_training_data)

models = {"QDA": GDA.QDA(reg_param = 1e-4),
          "LDA": GDA.LDA(),
          "Decicision Tree": dt.DecicisonTree(),
          "Random Forrest": dt.RandomForest(),
          "Random Forrest (No Bootstrap)": dt.RandomForest(bootstrap = False),
          "SVM": svm.SVM(),
          "AdaBoost": ada_boost.AdaBoost()}
          #"kNN": knn.KNN()

#-------------------------------------------------------------
#                           Main
#-------------------------------------------------------------

rng = np.random.default_rng(1)
index = pd.MultiIndex.from_product([datasets, ratio_training_data, models.keys()], names=["dataset", "training_ratio", "model"])
if os.path.exists("analysis_data/{FILENAME}.csv") and LOAD_DATA:
    old_runtime_results = pd.read_csv("analysis_data/{FILENAME}.csv", index_col=[0, 1, 2])
    new_runtime_results = pd.DataFrame(index=index, columns=["training_time", "predict_time", "error", "model_size"])
    runtime_results = pd.concat([old_runtime_results, new_runtime_results])
    runtime_results = runtime_results.drop_duplicates()
else:
    runtime_results = pd.DataFrame(index=index, columns=["training_time", "predict_time", "error", "model_size"])

for dataset in datasets:
    data = load_data.Data(dataset, verbose=True)

    for ratio in ratio_training_data:
        print("ratio:", ratio)
        data.split_by_ratio(0.8*ratio, 0.2, 0.0, rng)

        for name, model in models.items():
            print(f"Analysing {name}")
            t_0 = time.perf_counter()
            model.fit(data.train_features, data.train_labels)

            t_1 = time.perf_counter()
            runtime_results.loc[(dataset, ratio, name), "training_time"] = t_1 - t_0

            y_pred = model.predict(data.val_features)

            t_2 = time.perf_counter()
            runtime_results.loc[(dataset, ratio, name), "predict_time"] = t_2 - t_1
            
            loss = One_Zero_Loss()
            runtime_results.loc[(dataset, ratio, name), "error"] = loss(y_pred, data.val_labels)

            runtime_results.loc[(dataset, ratio, name), "model_size"] = model.model_size()

print(runtime_results)
if SAVE_DATA:
    runtime_results.to_csv(f"analysis_data/{FILENAME}.csv")
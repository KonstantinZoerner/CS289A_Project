import time
import numpy as np
import pandas as pd
import src.load_data as load_data
import models.GDA as GDA
import models.decision_trees as dt
import models.knn as knn
import models.svm as svm
import models.ada_boost as ada_boost
import models.neural_network as nn
import models.ensemble_learners as ensemble
from src.losses import One_Zero_Loss
from tqdm import tqdm
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# set model
model = nn.NeuralNetwork(verbose=True)

# datasets = ["cancer", "diabetes"]
# runs = [100, 1]
datasets = ["cancer"]
runs = [1]
ratio = 1
loss = One_Zero_Loss()

for i, dataset in enumerate(datasets):
    training_time = 0
    predict_time = 0
    error = 0
    model_size = 0
    data = load_data.Data(dataset, verbose=True)
    for run in tqdm(range(runs[i])):
        data.split_by_ratio(0.8 * ratio, 0.2, 0.0, rng)
        t_0 = time.perf_counter()
        model.fit(data.train_features, data.train_labels, epochs=20, model=None,X_val=data.val_features,y_val=data.val_labels)
        t_1 = time.perf_counter()
        y_pred = model.predict(data.val_features)  
        t_2 = time.perf_counter()
        
        training_time += t_1 - t_0
        predict_time += t_2 - t_1
        error += loss(y_pred, data.val_labels)
        model_size += model.model_size()

    print(f"training_time = {training_time / runs[i]:.4f}")
    print(f"predict_time = {predict_time / runs[i]:.4f}")
    print(f"error = {error / runs[i]:.4f}")
    print(f"model_size = {model_size / runs[i]:.4f}")

    plt.plot(model.acc_train)
    plt.plot(model.acc_val)
    plt.show()
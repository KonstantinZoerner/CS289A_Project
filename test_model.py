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
from src.losses import One_Zero_Loss


rng = np.random.default_rng(1)

model = nn.NeuralNetwork()

datasets = ["cancer", "diabetes"]

for dataset in datasets:
    data = load_data.Data(dataset, verbose=True)

    data.split_by_ratio(0.4, 0.2, 0.0, rng)

    t_0 = time.perf_counter()
    model.fit(data.train_features, data.train_labels)

    t_1 = time.perf_counter()
    print(f"Train time = {t_1 - t_0}")

    y_pred = model.predict(data.val_features)

    t_2 = time.perf_counter()
    print(f"Predict time = {t_2 - t_1}")
    
    loss = One_Zero_Loss()
    
    print(f"Error = {loss(y_pred, data.val_labels)}")
    print(f"Model Size = {model.model_size()}")
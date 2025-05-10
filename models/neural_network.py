import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.losses import One_Zero_Loss

torch.manual_seed(1)
torch.cuda.manual_seed(1)

class NeuralNetwork(nn.Module):
    def __init__(self, verbose = False):
        super().__init__()
        self.model = None  # will be created during fit
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.device = "cpu"
        self.verbose = verbose
        
    def fit(self, X, y, epochs=20, model = None, X_val = None, y_val = None):
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        in_size = X_tensor.shape[1]
        out_size = int(torch.max(y_tensor).item()) + 1

        if model == None:
            self.model = nn.Sequential(
                        nn.Linear(in_size, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        #nn.Dropout(0.2),
                        nn.Linear(256,256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        #nn.Dropout(0.2),
                        nn.Linear(256,out_size)
                    ).to(self.device)
        else:
            self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
            

        dataset = TensorDataset(X_tensor, y_tensor)
        if y_tensor.shape[0] >= 10000:
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=True)
        else:
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True)

        self.model.train()

        self.acc_train = []
        self.acc_val = []

        self.acc_train.append(self.eval(X, y))
        if X_val is not None:
                self.acc_val.append(self.eval(X_val, y_val))

        for epoch in range(epochs):
            if self.verbose:
                iterator = tqdm(dataloader, unit="batch")
            else:
                iterator = dataloader

            for x_batch, y_batch in iterator:
                x_batch, y_batch = x_batch.float().to(self.device), y_batch.long().to(self.device)
                self.optimizer.zero_grad() # Remove the gradients from the previous step
                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            
            self.acc_train.append(self.eval(X, y))
            if X_val is not None:
                self.acc_val.append(self.eval(X_val, y_val))

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(next(self.model.parameters()).device)
            pred = self.model(X_tensor)
            y_pred = torch.argmax(pred, dim=1)
        
        self.model.train()
        return  y_pred.cpu().numpy()
    
    def model_size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def eval(self, X, y):
        y_pred = self.predict(X)
        loss = One_Zero_Loss()
        return loss(y_pred, y)